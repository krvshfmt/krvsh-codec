//! Quantization + the byte serialization that feeds the range coder.
//!
//! Pipeline (krvsh-codec-spec.md §6, §7), per {k,v} tensor's projected coefficients:
//!   coef (f32, R per token) --quant--> q (i32) --zigzag--> u (u32) --varint--> bytes
//!   bytes --range-code--> stream   (range coder lives in rangecoder.zig)
//! and the exact inverse on decode.
//!
//! Determinism is the whole point: encode on the cloud and decode on the edge must
//! agree bit-for-bit. So:
//!   - the quant step Delta is the fp16-rounded value that actually ships in the frame
//!     (`scale_q`); encoder and decoder both divide/multiply by the SAME bits. Callers
//!     get Delta through `fp16RoundTripped` so the fp32 they quantize with equals the
//!     fp16 they serialize.
//!   - rounding is round-half-to-even (banker's), stated explicitly, not left to a
//!     platform default.
//! Locked by golden vectors (added with the end-to-end codec).

const std = @import("std");

// ---------------------------------------------------------------------------
// fp16 scale handling
// ---------------------------------------------------------------------------

/// Round an f32 through fp16 and back to f32. The frame stores Delta as fp16
/// (`scale_q`); the encoder MUST quantize with this exact value so the decoder,
/// which only ever sees the fp16 bits, reconstructs with the identical step.
pub fn fp16RoundTripped(x: f32) f32 {
    const h: f16 = @floatCast(x);
    return @floatCast(h);
}

pub fn f32ToFp16Bits(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

pub fn fp16BitsToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

// ---------------------------------------------------------------------------
// quantize / dequantize
// ---------------------------------------------------------------------------

/// Round half to even. Zig's @round is round-half-away-from-zero, so we implement
/// banker's rounding explicitly for cross-platform determinism.
fn roundHalfEven(x: f64) f64 {
    const fl = @floor(x);
    const diff = x - fl;
    if (diff < 0.5) return fl;
    if (diff > 0.5) return fl + 1;
    // exactly .5 -> pick the even neighbor
    const fl_i: i64 = @intFromFloat(fl);
    return if (@mod(fl_i, 2) == 0) fl else fl + 1;
}

/// q[i] = round_half_even(coef[i] / delta). `delta` must be the fp16-roundtripped
/// step (see `fp16RoundTripped`). `delta` > 0 required. out.len == coef.len.
pub fn quantize(coef: []const f32, delta: f32, out: []i32) void {
    std.debug.assert(delta > 0);
    std.debug.assert(out.len == coef.len);
    const d: f64 = delta;
    for (coef, out) |c, *o| {
        const r = roundHalfEven(@as(f64, c) / d);
        o.* = @intFromFloat(r);
    }
}

/// coef_hat[i] = q[i] * delta. Same `delta` the encoder used.
pub fn dequantize(q: []const i32, delta: f32, out: []f32) void {
    std.debug.assert(out.len == q.len);
    for (q, out) |qi, *o| {
        o.* = @as(f32, @floatFromInt(qi)) * delta;
    }
}

// ---------------------------------------------------------------------------
// zig-zag
// ---------------------------------------------------------------------------

/// Map signed to unsigned so small-magnitude values (either sign) become small
/// unsigned values: 0,-1,1,-2,2,... -> 0,1,2,3,4,...
pub fn zigzag(q: i32) u32 {
    const v: u32 = @bitCast(q);
    // (q << 1) ^ (q >> 31), with >> 31 being an arithmetic (sign) shift.
    const sign: u32 = @bitCast(q >> 31);
    return (v << 1) ^ sign;
}

pub fn unzigzag(u: u32) i32 {
    const half: u32 = u >> 1;
    const neg: u32 = 0 -% (u & 1); // all-ones if u was odd, else zero
    return @bitCast(half ^ neg);
}

// ---------------------------------------------------------------------------
// LEB128 varint
// ---------------------------------------------------------------------------

/// Append the LEB128 varint of `u` to `out`. 7 payload bits per byte, high bit set
/// on all but the last byte.
pub fn varintEncode(out: *std.array_list.Managed(u8), u: u32) !void {
    var v = u;
    while (true) {
        var byte: u8 = @intCast(v & 0x7F);
        v >>= 7;
        if (v != 0) byte |= 0x80;
        try out.append(byte);
        if (v == 0) break;
    }
}

pub const VarintError = error{ Truncated, Overflow };

/// Decode one LEB128 varint from `buf` starting at `pos.*`, advancing `pos.*`.
/// A u32 is at most 5 varint bytes (5*7 = 35 >= 32 bits); the 5th byte may carry at
/// most 4 payload bits.
pub fn varintDecode(buf: []const u8, pos: *usize) VarintError!u32 {
    var result: u64 = 0;
    var shift: u6 = 0;
    var nbytes: usize = 0;
    while (true) {
        if (pos.* >= buf.len) return error.Truncated;
        const byte = buf[pos.*];
        pos.* += 1;
        nbytes += 1;
        const payload: u64 = byte & 0x7F;
        // 5th byte of a u32 varint may only contribute the low 4 bits.
        if (nbytes == 5 and payload > 0xF) return error.Overflow;
        if (nbytes > 5) return error.Overflow;
        result |= payload << shift;
        if (byte & 0x80 == 0) break;
        shift += 7;
    }
    if (result > std.math.maxInt(u32)) return error.Overflow;
    return @intCast(result);
}

/// Encode a whole q[] as zig-zag varints into `out`.
pub fn encodeSymbols(out: *std.array_list.Managed(u8), q: []const i32) !void {
    for (q) |qi| try varintEncode(out, zigzag(qi));
}

/// Decode `n` zig-zag varints from `buf` into `out` (out.len == n).
pub fn decodeSymbols(buf: []const u8, out: []i32) VarintError!void {
    var pos: usize = 0;
    for (out) |*o| {
        const u = try varintDecode(buf, &pos);
        o.* = unzigzag(u);
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const t = std.testing;

test "quant roundtrip within delta/2" {
    const delta = fp16RoundTripped(0.05);
    var prng = std.Random.DefaultPrng.init(0xBEEF);
    const rand = prng.random();
    var coef: [4096]f32 = undefined;
    for (&coef) |*c| c.* = (rand.float(f32) - 0.5) * 20.0; // range ~[-10,10]

    var q: [4096]i32 = undefined;
    quantize(&coef, delta, &q);
    var hat: [4096]f32 = undefined;
    dequantize(&q, delta, &hat);

    for (coef, hat) |c, h| {
        try t.expect(@abs(c - h) <= delta / 2.0 + 1e-6);
    }
}

test "quant round-half-to-even" {
    // delta=1 so coef values land exactly on .5 boundaries.
    const delta: f32 = 1.0;
    const coef = [_]f32{ 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5 };
    var q: [coef.len]i32 = undefined;
    quantize(&coef, delta, &q);
    // half-to-even: 0.5->0, 1.5->2, 2.5->2, 3.5->4, -0.5->0, -1.5->-2, -2.5->-2
    const want = [_]i32{ 0, 2, 2, 4, 0, -2, -2 };
    try t.expectEqualSlices(i32, &want, &q);
}

test "quant zero coef gives zero" {
    const delta = fp16RoundTripped(0.1);
    const coef = [_]f32{ 0, 0, 0 };
    var q: [3]i32 = undefined;
    quantize(&coef, delta, &q);
    try t.expectEqualSlices(i32, &[_]i32{ 0, 0, 0 }, &q);
}

test "fp16 roundtrip is idempotent" {
    const a = fp16RoundTripped(0.0531);
    const b = fp16RoundTripped(a);
    try t.expectEqual(a, b);
    // and the bits form the inverse pair
    try t.expectEqual(a, fp16BitsToF32(f32ToFp16Bits(a)));
}

test "zigzag maps small magnitudes to small unsigned" {
    try t.expectEqual(@as(u32, 0), zigzag(0));
    try t.expectEqual(@as(u32, 1), zigzag(-1));
    try t.expectEqual(@as(u32, 2), zigzag(1));
    try t.expectEqual(@as(u32, 3), zigzag(-2));
    try t.expectEqual(@as(u32, 4), zigzag(2));
}

test "zigzag roundtrip over range incl extremes" {
    const vals = [_]i32{ 0, 1, -1, 2, -2, 1000, -1000, std.math.maxInt(i32), std.math.minInt(i32) };
    for (vals) |v| try t.expectEqual(v, unzigzag(zigzag(v)));
}

test "varint roundtrip boundaries" {
    const vals = [_]u32{ 0, 1, 127, 128, 300, 16383, 16384, 0xFFFF_FFFF };
    for (vals) |v| {
        var out = std.array_list.Managed(u8).init(t.allocator);
        defer out.deinit();
        try varintEncode(&out, v);
        var pos: usize = 0;
        const got = try varintDecode(out.items, &pos);
        try t.expectEqual(v, got);
        try t.expectEqual(out.items.len, pos); // consumed exactly
    }
}

test "varint small values are one byte" {
    var out = std.array_list.Managed(u8).init(t.allocator);
    defer out.deinit();
    try varintEncode(&out, 0);
    try varintEncode(&out, 127);
    try t.expectEqual(@as(usize, 2), out.items.len);
}

test "varint decode truncated errors" {
    const buf = [_]u8{0x80}; // continuation bit set, no following byte
    var pos: usize = 0;
    try t.expectError(error.Truncated, varintDecode(&buf, &pos));
}

test "varint decode overflow errors" {
    // Six 0x80 bytes then 0x01 => shift past u32.
    const buf = [_]u8{ 0x80, 0x80, 0x80, 0x80, 0x80, 0x01 };
    var pos: usize = 0;
    try t.expectError(error.Overflow, varintDecode(&buf, &pos));
}

test "symbol stream roundtrip" {
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    var q: [2048]i32 = undefined;
    for (&q) |*x| {
        // peaky: mostly 0, some +-small, rare large
        const r = rand.float(f32);
        x.* = if (r < 0.8) 0 else if (r < 0.97) rand.intRangeAtMost(i32, -3, 3) else rand.intRangeAtMost(i32, -500, 500);
    }
    var out = std.array_list.Managed(u8).init(t.allocator);
    defer out.deinit();
    try encodeSymbols(&out, &q);

    var dec: [2048]i32 = undefined;
    try decodeSymbols(out.items, &dec);
    try t.expectEqualSlices(i32, &q, &dec);
}

test "full pipeline: coef -> quant -> symbols -> unquant" {
    const delta = fp16RoundTripped(0.08);
    var prng = std.Random.DefaultPrng.init(321);
    const rand = prng.random();
    var coef: [1024]f32 = undefined;
    for (&coef) |*c| c.* = (rand.float(f32) - 0.5) * 6.0;

    var q: [1024]i32 = undefined;
    quantize(&coef, delta, &q);

    var bytes = std.array_list.Managed(u8).init(t.allocator);
    defer bytes.deinit();
    try encodeSymbols(&bytes, &q);

    var q2: [1024]i32 = undefined;
    try decodeSymbols(bytes.items, &q2);
    try t.expectEqualSlices(i32, &q, &q2); // lossless through the byte layer

    var hat: [1024]f32 = undefined;
    dequantize(&q2, delta, &hat);
    for (coef, hat) |c, h| try t.expect(@abs(c - h) <= delta / 2.0 + 1e-6);
}
