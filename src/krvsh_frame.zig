//! krvsh frame (`kfrm`) payload: per-tensor encode/decode.
//!
//! For each tensor X of shape (T, D) the pipeline (krvsh-codec-spec.md §5-§7):
//!   coef = (X - mean) @ basis          # (T, R), basis stored as Vt[:R] (R,D)
//!   rms  = sqrt(mean(coef^2))
//!   Delta = fp16( s * rms )            # the fp16 that ships as scale_q
//!   q = round_half_even(coef / Delta)
//!   stream = rangecode( zigzag_varint(q) )
//! and the inverse:
//!   coef_hat = q * Delta
//!   X_hat = coef_hat @ basis.T + mean
//!
//! The init stores `basis` as Vt[:R], row-major (R, D), fp16. We decode it to f32 for
//! the projection (basis precision is not the fidelity bottleneck; the quantizer is).
//! Little-endian throughout.

const std = @import("std");
const quant = @import("quant.zig");
const rc = @import("rangecoder.zig");
const init_mod = @import("krvsh_init.zig");

pub const CODER_RANGE: u8 = 1;

/// One tensor's fpay layout prefix (krvsh-codec-spec.md §5), before the stream bytes:
///   scale_q u16 (fp16) | n_symbols u32 | coder_id u8 | stream_len u32
pub const TENSOR_PREFIX_LEN: usize = 2 + 4 + 1 + 4;

pub const EncodeError = std.mem.Allocator.Error;
pub const DecodeError = error{
    ShortBuffer,
    UnknownCoder,
    SymbolCountMismatch,
} || quant.VarintError || std.mem.Allocator.Error;

/// Decode an fp16[R*D] basis (row-major Vt[:R]) and fp16[D] mean to f32 scratch.
/// Caller owns the returned slices.
fn decodeBasisMean(
    alloc: std.mem.Allocator,
    mean_fp16: []const u8,
    basis_fp16: []const u8,
    d: usize,
    r: usize,
) !struct { mean: []f32, basis: []f32 } {
    const mean = try alloc.alloc(f32, d);
    errdefer alloc.free(mean);
    for (0..d) |i| mean[i] = init_mod.fp16LEToF32(mean_fp16[i * 2 .. i * 2 + 2][0..2]);

    const basis = try alloc.alloc(f32, r * d);
    errdefer alloc.free(basis);
    for (0..r * d) |i| basis[i] = init_mod.fp16LEToF32(basis_fp16[i * 2 .. i * 2 + 2][0..2]);

    return .{ .mean = mean, .basis = basis };
}

/// Project: coef[t,r] = sum_d (X[t,d]-mean[d]) * Vt[r,d]. X is row-major (T,D).
const VLEN = 8;
const Vf = @Vector(VLEN, f32);

/// Vectorized dot product of two equal-length f32 slices.
///
/// Determinism across architectures: per-lane `+`/`*` on `@Vector` are elementwise
/// IEEE ops, bit-identical on any conforming ISA. `@reduce(.Add, v)` is NOT safe here:
/// it may lower to different summation trees on NEON vs AVX, giving different f32
/// rounding at large D. So we keep 8 lane accumulators and fold them in a FIXED,
/// explicit left-to-right order, which both ISAs compute identically. The scalar
/// remainder is likewise fixed-order. (Verified local==mosk hash.)
fn dot(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.strict);
    const n = a.len;
    const nv = n / VLEN;
    var accv: Vf = @splat(0);
    var k: usize = 0;
    while (k < nv) : (k += 1) {
        const av: Vf = a[k * VLEN ..][0..VLEN].*;
        const bv: Vf = b[k * VLEN ..][0..VLEN].*;
        const prod: Vf = av * bv; // separate mul; strict mode forbids fusing into FMA
        accv += prod;
    }
    // Fixed-order horizontal fold of the lanes (not @reduce, which varies by ISA).
    var acc: f32 = accv[0];
    inline for (1..VLEN) |lane| acc += accv[lane];
    var i = nv * VLEN;
    while (i < n) : (i += 1) acc += a[i] * b[i];
    return acc;
}

/// Fixed-order horizontal fold of a lane vector (deterministic across ISAs).
inline fn hsum(v: Vf) f32 {
    var s: f32 = v[0];
    inline for (1..VLEN) |l| s += v[l];
    return s;
}

/// Project a 4-token x 2-basis-row register tile. `xc` is the centered X (T,D),
/// `basis` is (R,D). Writes coef[(ti+i)*R + (ri+j)] for i in 0..4, j in 0..2.
/// Branch-free, named accumulators (no indexed array) so LLVM keeps all 8 in vector
/// registers -- verified no stack spills. strict float: cross-arch bit-exact.
inline fn tile4x2(xc: []const f32, basis: []const f32, d: usize, r: usize, coef: []f32, ti: usize, ri: usize) void {
    @setFloatMode(.strict);
    const nv = d / VLEN;
    const x0 = xc[(ti + 0) * d ..];
    const x1 = xc[(ti + 1) * d ..];
    const x2 = xc[(ti + 2) * d ..];
    const x3 = xc[(ti + 3) * d ..];
    const bA = basis[(ri + 0) * d ..];
    const bB = basis[(ri + 1) * d ..];
    var a0A: Vf = @splat(0);
    var a1A: Vf = @splat(0);
    var a2A: Vf = @splat(0);
    var a3A: Vf = @splat(0);
    var a0B: Vf = @splat(0);
    var a1B: Vf = @splat(0);
    var a2B: Vf = @splat(0);
    var a3B: Vf = @splat(0);
    var k: usize = 0;
    while (k < nv) : (k += 1) {
        const o = k * VLEN;
        const va: Vf = bA[o..][0..VLEN].*;
        const vb: Vf = bB[o..][0..VLEN].*;
        const v0: Vf = x0[o..][0..VLEN].*;
        const v1: Vf = x1[o..][0..VLEN].*;
        const v2: Vf = x2[o..][0..VLEN].*;
        const v3: Vf = x3[o..][0..VLEN].*;
        a0A += v0 * va;
        a1A += v1 * va;
        a2A += v2 * va;
        a3A += v3 * va;
        a0B += v0 * vb;
        a1B += v1 * vb;
        a2B += v2 * vb;
        a3B += v3 * vb;
    }
    // scalar-tail fold identical to dot()'s remainder, per (token,row).
    var s0A = hsum(a0A);
    var s1A = hsum(a1A);
    var s2A = hsum(a2A);
    var s3A = hsum(a3A);
    var s0B = hsum(a0B);
    var s1B = hsum(a1B);
    var s2B = hsum(a2B);
    var s3B = hsum(a3B);
    var i = nv * VLEN;
    while (i < d) : (i += 1) {
        const ba = bA[i];
        const bb = bB[i];
        s0A += x0[i] * ba;
        s1A += x1[i] * ba;
        s2A += x2[i] * ba;
        s3A += x3[i] * ba;
        s0B += x0[i] * bb;
        s1B += x1[i] * bb;
        s2B += x2[i] * bb;
        s3B += x3[i] * bb;
    }
    coef[(ti + 0) * r + ri + 0] = s0A;
    coef[(ti + 1) * r + ri + 0] = s1A;
    coef[(ti + 2) * r + ri + 0] = s2A;
    coef[(ti + 3) * r + ri + 0] = s3A;
    coef[(ti + 0) * r + ri + 1] = s0B;
    coef[(ti + 1) * r + ri + 1] = s1B;
    coef[(ti + 2) * r + ri + 1] = s2B;
    coef[(ti + 3) * r + ri + 1] = s3B;
}

/// coef = (X - mean) @ basis^T, where basis is Vt[:R] (R,D). Centers all T rows once,
/// then runs a branch-free 4-token x 2-row register-tiled matmul with scalar remainders
/// for T%4 and R%2. The per-(token,row) reduction order is identical to dot(), so the
/// tiled and remainder paths agree bit-for-bit and the result is cross-arch deterministic.
fn project(alloc: std.mem.Allocator, x: []const f32, mean: []const f32, basis: []const f32, t: usize, d: usize, r: usize, coef: []f32) !void {
    // Center all T rows once into a resident (T,D) scratch.
    const xc = try alloc.alloc(f32, t * d);
    defer alloc.free(xc);
    for (0..t) |ti| {
        const xrow = x[ti * d .. ti * d + d];
        const crow = xc[ti * d .. ti * d + d];
        for (0..d) |di| crow[di] = xrow[di] - mean[di];
    }

    const t_blk = t - (t % 4);
    const r_blk = r - (r % 2);

    var ti: usize = 0;
    while (ti < t_blk) : (ti += 4) {
        var ri: usize = 0;
        while (ri < r_blk) : (ri += 2) tile4x2(xc, basis, d, r, coef, ti, ri);
        // R remainder (odd last column) for these 4 tokens
        while (ri < r) : (ri += 1) {
            const brow = basis[ri * d .. ri * d + d];
            coef[(ti + 0) * r + ri] = dot(xc[(ti + 0) * d .. (ti + 0) * d + d], brow);
            coef[(ti + 1) * r + ri] = dot(xc[(ti + 1) * d .. (ti + 1) * d + d], brow);
            coef[(ti + 2) * r + ri] = dot(xc[(ti + 2) * d .. (ti + 2) * d + d], brow);
            coef[(ti + 3) * r + ri] = dot(xc[(ti + 3) * d .. (ti + 3) * d + d], brow);
        }
    }
    // T remainder (last <4 tokens): plain dot over all R.
    while (ti < t) : (ti += 1) {
        const xrow = xc[ti * d .. ti * d + d];
        for (0..r) |ri| coef[ti * r + ri] = dot(xrow, basis[ri * d .. ri * d + d]);
    }
}

/// Reconstruct: X_hat[t,d] = mean[d] + sum_r coef_hat[t,r] * Vt[r,d].
fn reconstruct(coef: []const f32, mean: []const f32, basis: []const f32, t: usize, d: usize, r: usize, xhat: []f32) void {
    @setFloatMode(.strict);
    const nv = d / VLEN;
    for (0..t) |ti| {
        const crow = coef[ti * r .. ti * r + r];
        const xrow = xhat[ti * d .. ti * d + d];
        @memcpy(xrow, mean[0..d]);
        for (0..r) |ri| {
            const c = crow[ri];
            if (c == 0) continue; // post-quant sparsity: most coeffs are zero
            const brow = basis[ri * d .. ri * d + d];
            const cv: Vf = @splat(c);
            var k: usize = 0;
            while (k < nv) : (k += 1) {
                const base = k * VLEN;
                var xv: Vf = xrow[base..][0..VLEN].*;
                const bv: Vf = brow[base..][0..VLEN].*;
                xv += cv * bv;
                xrow[base..][0..VLEN].* = xv;
            }
            var i = nv * VLEN;
            while (i < d) : (i += 1) xrow[i] += c * brow[i];
        }
    }
}

/// Encode one tensor's fpay (prefix + range-coded stream) into `out`.
/// `x` is row-major (T, D). `mean`/`basis` are the init's fp16 blocks for this tensor.
/// `s` is the rate knob; Delta = fp16(s * rms). Returns bytes appended.
pub fn encodeTensor(
    alloc: std.mem.Allocator,
    out: *std.array_list.Managed(u8),
    x: []const f32,
    mean_fp16: []const u8,
    basis_fp16: []const u8,
    d: usize,
    r: usize,
    s: f32,
) EncodeError!void {
    const t = x.len / d;
    const bm = try decodeBasisMean(alloc, mean_fp16, basis_fp16, d, r);
    defer alloc.free(bm.mean);
    defer alloc.free(bm.basis);

    const coef = try alloc.alloc(f32, t * r);
    defer alloc.free(coef);
    try project(alloc, x, bm.mean, bm.basis, t, d, r, coef);

    // rms over all coefficients
    var sumsq: f64 = 0;
    for (coef) |c| sumsq += @as(f64, c) * c;
    const rms: f32 = if (coef.len == 0) 0 else @floatCast(@sqrt(sumsq / @as(f64, @floatFromInt(coef.len))));
    const delta = quant.fp16RoundTripped(s * rms);
    // Guard: a degenerate all-zero tensor -> delta 0; use 1.0 so quant is well-defined
    // (all q become 0 anyway).
    const delta_safe = if (delta > 0) delta else 1.0;

    const q = try alloc.alloc(i32, t * r);
    defer alloc.free(q);
    quant.quantize(coef, delta_safe, q);

    // symbols -> bytes -> range-coded stream
    var symbols = std.array_list.Managed(u8).init(alloc);
    defer symbols.deinit();
    try quant.encodeSymbols(&symbols, q);

    var stream = std.array_list.Managed(u8).init(alloc);
    defer stream.deinit();
    try rc.encodeBuf(&stream, symbols.items);

    // prefix
    var prefix: [TENSOR_PREFIX_LEN]u8 = undefined;
    std.mem.writeInt(u16, prefix[0..2], quant.f32ToFp16Bits(delta_safe), .little);
    std.mem.writeInt(u32, prefix[2..6], @intCast(t * r), .little);
    prefix[6] = CODER_RANGE;
    std.mem.writeInt(u32, prefix[7..11], @intCast(stream.items.len), .little);
    try out.appendSlice(&prefix);
    try out.appendSlice(stream.items);
}

/// A parsed tensor fpay (prefix fields + stream slice borrowed from the buffer).
pub const TensorPayload = struct {
    scale_q_bits: u16,
    n_symbols: u32,
    coder_id: u8,
    stream: []const u8,
    /// total bytes this tensor occupied in the buffer
    consumed: usize,

    pub fn delta(self: TensorPayload) f32 {
        return quant.fp16BitsToF32(self.scale_q_bits);
    }
};

/// Parse one tensor's fpay prefix+stream starting at buf[0].
pub fn parseTensor(buf: []const u8) DecodeError!TensorPayload {
    if (buf.len < TENSOR_PREFIX_LEN) return error.ShortBuffer;
    const scale_q_bits = std.mem.readInt(u16, buf[0..2], .little);
    const n_symbols = std.mem.readInt(u32, buf[2..6], .little);
    const coder_id = buf[6];
    const stream_len = std.mem.readInt(u32, buf[7..11], .little);
    if (coder_id != CODER_RANGE) return error.UnknownCoder;
    const end = TENSOR_PREFIX_LEN + stream_len;
    if (buf.len < end) return error.ShortBuffer;
    return .{
        .scale_q_bits = scale_q_bits,
        .n_symbols = n_symbols,
        .coder_id = coder_id,
        .stream = buf[TENSOR_PREFIX_LEN..end],
        .consumed = end,
    };
}

/// Decode one tensor's fpay into X_hat (row-major (T, D), caller-allocated len == t*d).
pub fn decodeTensor(
    alloc: std.mem.Allocator,
    payload: TensorPayload,
    mean_fp16: []const u8,
    basis_fp16: []const u8,
    d: usize,
    r: usize,
    t: usize,
    xhat: []f32,
) DecodeError!void {
    if (payload.n_symbols != t * r) return error.SymbolCountMismatch;
    std.debug.assert(xhat.len == t * d);

    // Range-decode and parse exactly n_symbols zig-zag varints, pulling bytes through
    // the range decoder on demand (we stored symbol count, not decoded byte length).
    const q = try alloc.alloc(i32, @intCast(payload.n_symbols));
    defer alloc.free(q);
    try decodeSymbolsFromStream(payload.stream, q);

    const delta = payload.delta();
    const coef = try alloc.alloc(f32, @intCast(payload.n_symbols));
    defer alloc.free(coef);
    quant.dequantize(q, delta, coef);

    const bm = try decodeBasisMean(alloc, mean_fp16, basis_fp16, d, r);
    defer alloc.free(bm.mean);
    defer alloc.free(bm.basis);
    reconstruct(coef, bm.mean, bm.basis, t, d, r, xhat);
}

/// Range-decode `stream` and parse exactly `out.len` zig-zag varints from the decoded
/// byte sequence. We don't know the decoded byte length up front, so we decode bytes on
/// demand through the range decoder as the varint parser requests them.
fn decodeSymbolsFromStream(stream: []const u8, out: []i32) !void {
    var model = rc.Model.init();
    var dec = rc.Decoder.init(stream);
    for (out) |*o| {
        var u: u32 = 0;
        var shift: u6 = 0;
        var nbytes: usize = 0;
        while (true) {
            const byte = dec.decode(&model);
            nbytes += 1;
            const payload: u32 = byte & 0x7F;
            if (nbytes > 5) return error.Overflow;
            u |= payload << @as(u5, @intCast(shift));
            if (byte & 0x80 == 0) break;
            shift += 7;
        }
        o.* = quant.unzigzag(u);
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const ztest = std.testing;

/// Build identity-ish fp16 basis/mean for a tiny D=R case to test the pipeline.
fn fp16Slice(alloc: std.mem.Allocator, vals: []const f32) ![]u8 {
    const out = try alloc.alloc(u8, vals.len * 2);
    for (vals, 0..) |v, i| init_mod.f32ToFp16LE(v, out[i * 2 .. i * 2 + 2][0..2]);
    return out;
}

test "encode/decode tensor roundtrip within quant tolerance (identity basis)" {
    const alloc = ztest.allocator;
    const D = 4;
    const R = 4;
    const T = 3;
    // identity basis (R==D): Vt[:R] = I, so coef == X - mean.
    var basis_vals: [R * D]f32 = @splat(0);
    for (0..R) |i| basis_vals[i * D + i] = 1.0;
    const basis = try fp16Slice(alloc, &basis_vals);
    defer alloc.free(basis);
    const mean = try fp16Slice(alloc, &[_]f32{ 0, 0, 0, 0 });
    defer alloc.free(mean);

    var x: [T * D]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    const rand = prng.random();
    for (&x) |*v| v.* = (rand.float(f32) - 0.5) * 4.0;

    const s: f32 = 0.5;
    var out = std.array_list.Managed(u8).init(alloc);
    defer out.deinit();
    try encodeTensor(alloc, &out, &x, mean, basis, D, R, s);

    const payload = try parseTensor(out.items);
    try ztest.expectEqual(@as(u32, T * R), payload.n_symbols);
    try ztest.expectEqual(CODER_RANGE, payload.coder_id);
    try ztest.expectEqual(out.items.len, payload.consumed);

    var xhat: [T * D]f32 = undefined;
    try decodeTensor(alloc, payload, mean, basis, D, R, T, &xhat);

    // With identity basis, reconstruction error is purely quantization: <= delta/2.
    const delta = payload.delta();
    for (x, xhat) |xi, hi| try ztest.expect(@abs(xi - hi) <= delta / 2.0 + 1e-4);
}

test "all-zero tensor encodes and decodes to zero" {
    const alloc = ztest.allocator;
    const D = 4;
    const R = 2;
    const T = 2;
    var basis_vals: [R * D]f32 = @splat(0);
    basis_vals[0] = 1;
    basis_vals[D + 1] = 1;
    const basis = try fp16Slice(alloc, &basis_vals);
    defer alloc.free(basis);
    const mean = try fp16Slice(alloc, &[_]f32{ 0, 0, 0, 0 });
    defer alloc.free(mean);
    const x: [T * D]f32 = @splat(0);

    var out = std.array_list.Managed(u8).init(alloc);
    defer out.deinit();
    try encodeTensor(alloc, &out, &x, mean, basis, D, R, 1.0);
    const payload = try parseTensor(out.items);
    var xhat: [T * D]f32 = undefined;
    try decodeTensor(alloc, payload, mean, basis, D, R, T, &xhat);
    for (xhat) |h| try ztest.expectEqual(@as(f32, 0), h);
}

test "parse rejects unknown coder" {
    var buf: [TENSOR_PREFIX_LEN]u8 = undefined;
    std.mem.writeInt(u16, buf[0..2], 0, .little);
    std.mem.writeInt(u32, buf[2..6], 4, .little);
    buf[6] = 99; // bad coder
    std.mem.writeInt(u32, buf[7..11], 0, .little);
    try ztest.expectError(error.UnknownCoder, parseTensor(&buf));
}

test "parse rejects short stream" {
    var buf: [TENSOR_PREFIX_LEN]u8 = undefined;
    std.mem.writeInt(u16, buf[0..2], 0, .little);
    std.mem.writeInt(u32, buf[2..6], 4, .little);
    buf[6] = CODER_RANGE;
    std.mem.writeInt(u32, buf[7..11], 100, .little); // claims 100 stream bytes, none present
    try ztest.expectError(error.ShortBuffer, parseTensor(&buf));
}

test "decode rejects symbol-count mismatch" {
    const alloc = ztest.allocator;
    const D = 4;
    const R = 4;
    var basis_vals: [R * D]f32 = @splat(0);
    for (0..R) |i| basis_vals[i * D + i] = 1.0;
    const basis = try fp16Slice(alloc, &basis_vals);
    defer alloc.free(basis);
    const mean = try fp16Slice(alloc, &[_]f32{ 0, 0, 0, 0 });
    defer alloc.free(mean);
    const x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    var out = std.array_list.Managed(u8).init(alloc);
    defer out.deinit();
    try encodeTensor(alloc, &out, &x, mean, basis, D, R, 0.5);
    const payload = try parseTensor(out.items);

    // ask to decode with wrong T (expects T*R symbols != payload.n_symbols)
    var xhat: [3 * D]f32 = undefined;
    try ztest.expectError(error.SymbolCountMismatch, decodeTensor(alloc, payload, mean, basis, D, R, 3, &xhat));
}

test "two tensors concatenate and parse sequentially" {
    const alloc = ztest.allocator;
    const D = 2;
    const R = 2;
    const T = 2;
    var basis_vals: [R * D]f32 = .{ 1, 0, 0, 1 };
    const basis = try fp16Slice(alloc, &basis_vals);
    defer alloc.free(basis);
    const mean = try fp16Slice(alloc, &[_]f32{ 0, 0 });
    defer alloc.free(mean);
    const x1 = [_]f32{ 1, 2, 3, 4 };
    const x2 = [_]f32{ -1, -2, -3, -4 };

    var out = std.array_list.Managed(u8).init(alloc);
    defer out.deinit();
    try encodeTensor(alloc, &out, &x1, mean, basis, D, R, 0.5);
    const first_len = out.items.len;
    try encodeTensor(alloc, &out, &x2, mean, basis, D, R, 0.5);

    const p1 = try parseTensor(out.items);
    try ztest.expectEqual(first_len, p1.consumed);
    const p2 = try parseTensor(out.items[p1.consumed..]);
    try ztest.expectEqual(out.items.len - first_len, p2.consumed);

    var h1: [T * D]f32 = undefined;
    var h2: [T * D]f32 = undefined;
    try decodeTensor(alloc, p1, mean, basis, D, R, T, &h1);
    try decodeTensor(alloc, p2, mean, basis, D, R, T, &h2);
    for (x1, h1) |xi, hi| try ztest.expect(@abs(xi - hi) <= p1.delta() / 2.0 + 1e-4);
    for (x2, h2) |xi, hi| try ztest.expect(@abs(xi - hi) <= p2.delta() / 2.0 + 1e-4);
}
