//! KV source/output dtype conversions (krvsh-codec-spec.md §2.1.1).
//!
//! The codec computes in f32; this module converts the model's KV representation to/from
//! f32 at the encode/decode boundary. Conversions are defined bit-exactly and are
//! deterministic on any machine (manual bit manipulation for bf16/fp8, which Zig lacks
//! as native types; native @floatCast for fp16/fp32).
//!
//! All multi-byte dtypes little-endian.

const std = @import("std");

pub const Dtype = enum(u8) {
    bf16 = 0,
    fp16 = 1,
    fp32 = 2,
    fp8_e4m3 = 3,
    fp8_e5m2 = 4,
    int8 = 5, // RESERVED, not implemented in v0

    pub fn byteSize(self: Dtype) !usize {
        return switch (self) {
            .bf16, .fp16 => 2,
            .fp32 => 4,
            .fp8_e4m3, .fp8_e5m2 => 1,
            .int8 => error.UnsupportedDtype,
        };
    }
};

pub const DtypeError = error{ UnsupportedDtype, ShortBuffer, SizeMismatch };

/// Decode `src` (raw bytes of `dt`) into `out` f32. out.len == src.len / byteSize.
pub fn toF32(dt: Dtype, src: []const u8, out: []f32) DtypeError!void {
    const bs = try dt.byteSize();
    if (src.len != out.len * bs) return error.SizeMismatch;
    switch (dt) {
        .fp32 => for (out, 0..) |*o, i| {
            o.* = @bitCast(std.mem.readInt(u32, src[i * 4 ..][0..4], .little));
        },
        .fp16 => for (out, 0..) |*o, i| {
            const h: f16 = @bitCast(std.mem.readInt(u16, src[i * 2 ..][0..2], .little));
            o.* = @floatCast(h);
        },
        .bf16 => for (out, 0..) |*o, i| {
            const bits: u32 = @as(u32, std.mem.readInt(u16, src[i * 2 ..][0..2], .little)) << 16;
            o.* = @bitCast(bits);
        },
        .fp8_e4m3 => for (out, 0..) |*o, i| {
            o.* = e4m3ToF32(src[i]);
        },
        .fp8_e5m2 => for (out, 0..) |*o, i| {
            o.* = e5m2ToF32(src[i]);
        },
        .int8 => return error.UnsupportedDtype,
    }
}

/// Encode `in` f32 into `out` raw bytes of `dt`. out.len == in.len * byteSize.
pub fn fromF32(dt: Dtype, in: []const f32, out: []u8) DtypeError!void {
    const bs = try dt.byteSize();
    if (out.len != in.len * bs) return error.SizeMismatch;
    switch (dt) {
        .fp32 => for (in, 0..) |v, i| {
            std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(v), .little);
        },
        .fp16 => for (in, 0..) |v, i| {
            const h: f16 = @floatCast(v);
            std.mem.writeInt(u16, out[i * 2 ..][0..2], @bitCast(h), .little);
        },
        .bf16 => for (in, 0..) |v, i| {
            std.mem.writeInt(u16, out[i * 2 ..][0..2], f32ToBf16(v), .little);
        },
        .fp8_e4m3 => for (in, 0..) |v, i| {
            out[i] = f32ToE4m3(v);
        },
        .fp8_e5m2 => for (in, 0..) |v, i| {
            out[i] = f32ToE5m2(v);
        },
        .int8 => return error.UnsupportedDtype,
    }
}

// ---------------------------------------------------------------------------
// bf16: top 16 bits of f32, round-to-nearest-even.
// ---------------------------------------------------------------------------

pub fn f32ToBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    // NaN: force a quiet NaN that survives truncation.
    if ((bits & 0x7F80_0000) == 0x7F80_0000 and (bits & 0x007F_FFFF) != 0) {
        return @intCast((bits >> 16) | 0x0040);
    }
    // round-to-nearest-even on the discarded low 16 bits
    const rounding_bias: u32 = 0x7FFF + ((bits >> 16) & 1);
    return @intCast((bits + rounding_bias) >> 16);
}

// ---------------------------------------------------------------------------
// fp8 e4m3 (1-4-3, bias 7). No infinities; S.1111.111 is NaN. Max finite = 448.
// ---------------------------------------------------------------------------

fn e4m3ToF32(b: u8) f32 {
    const sign: u32 = @as(u32, b >> 7) << 31;
    const exp: u32 = (b >> 3) & 0x0F;
    const man: u32 = b & 0x07;
    if (exp == 0 and man == 0) return @bitCast(sign); // +/- 0
    if (exp == 0x0F and man == 0x07) { // NaN
        return @bitCast(sign | 0x7FC0_0000);
    }
    if (exp == 0) {
        // subnormal: value = man/8 * 2^(1-7)
        const f = @as(f32, @floatFromInt(man)) / 8.0 * exp2f(1 - 7);
        return if (sign != 0) -f else f;
    }
    // normal: (1 + man/8) * 2^(exp-7)
    const f = (1.0 + @as(f32, @floatFromInt(man)) / 8.0) * exp2f(@as(i32, @intCast(exp)) - 7);
    return if (sign != 0) -f else f;
}

fn f32ToE4m3(v: f32) u8 {
    return floatToFp8(v, 4, 3, 7, 448.0, true);
}

// ---------------------------------------------------------------------------
// fp8 e5m2 (1-5-2, bias 15). IEEE-like: inf = S.11111.00, NaN = S.11111.{nonzero}.
// Max finite = 57344.
// ---------------------------------------------------------------------------

fn e5m2ToF32(b: u8) f32 {
    const sign: u32 = @as(u32, b >> 7) << 31;
    const exp: u32 = (b >> 2) & 0x1F;
    const man: u32 = b & 0x03;
    if (exp == 0x1F) {
        if (man == 0) return @bitCast(sign | 0x7F80_0000); // inf
        return @bitCast(sign | 0x7FC0_0000); // NaN
    }
    if (exp == 0 and man == 0) return @bitCast(sign);
    if (exp == 0) {
        const f = @as(f32, @floatFromInt(man)) / 4.0 * exp2f(1 - 15);
        return if (sign != 0) -f else f;
    }
    const f = (1.0 + @as(f32, @floatFromInt(man)) / 4.0) * exp2f(@as(i32, @intCast(exp)) - 15);
    return if (sign != 0) -f else f;
}

fn f32ToE5m2(v: f32) u8 {
    return floatToFp8(v, 5, 2, 15, 57344.0, false);
}

// ---------------------------------------------------------------------------
// shared fp8 encoder: round-to-nearest-even, saturate to max finite.
// ---------------------------------------------------------------------------

fn exp2f(e: i32) f32 {
    // exact power of two via bit construction for the exponent range we use.
    return std.math.ldexp(@as(f32, 1.0), e);
}

fn floatToFp8(v: f32, comptime exp_bits: u3, comptime man_bits: u3, comptime bias: i32, max_finite: f32, comptime e4m3: bool) u8 {
    _ = e4m3;
    const bits: u32 = @bitCast(v);
    const sign: u8 = @intCast((bits >> 31) << 7);
    const abs: f32 = @abs(v);

    if (std.math.isNan(v)) return sign | 0x7F;
    if (abs > max_finite) return sign | maxFiniteBits(exp_bits, man_bits);
    if (abs == 0) return sign;

    const e_min: i32 = 1 - bias; // smallest normal exponent
    // decompose abs into mantissa in [1,2) and exponent: abs = mant * 2^exp2val
    const fe = std.math.frexp(abs); // abs = sig * 2^exp, 0.5 <= sig < 1
    const exp2val: i32 = fe.exponent - 1;
    var mant: f32 = fe.significand * 2.0; // [1,2)

    const man_scale: f32 = @floatFromInt(@as(u32, 1) << man_bits);
    if (exp2val < e_min) {
        // subnormal: shift mantissa to the e_min exponent, exp field = 0
        const shift = e_min - exp2val;
        mant = mant / exp2f(shift);
        const q = roundEven(mant * man_scale);
        const mq: u32 = @intFromFloat(q);
        return sign | @as(u8, @intCast(mq & ((@as(u32, 1) << man_bits) - 1)));
    }

    // normal: encode fractional mantissa (mant - 1)
    var mq = roundEven((mant - 1.0) * man_scale);
    var biased_exp: i32 = exp2val + bias;
    if (mq == man_scale) { // mantissa rounded up to 2.0 -> bump exponent
        mq = 0;
        biased_exp += 1;
    }
    const max_exp: i32 = (@as(i32, 1) << exp_bits) - 1;
    if (biased_exp >= max_exp) return sign | maxFiniteBits(exp_bits, man_bits);
    const e_field: u32 = @intCast(biased_exp);
    const m_field: u32 = @intFromFloat(mq);
    return sign | @as(u8, @intCast((e_field << man_bits) | m_field));
}

fn maxFiniteBits(comptime exp_bits: u3, comptime man_bits: u3) u8 {
    // e4m3: exp=1110(14) man=111 -> but max finite is exp=1111,man=110 (448). e5m2: exp=11110 man=11.
    if (exp_bits == 4 and man_bits == 3) return 0x7E; // S.1111.110 = 448
    if (exp_bits == 5 and man_bits == 2) return 0x7B; // S.11110.11 = 57344
    unreachable;
}

fn roundEven(x: f32) f32 {
    const fl = @floor(x);
    const diff = x - fl;
    if (diff < 0.5) return fl;
    if (diff > 0.5) return fl + 1;
    const fi: i64 = @intFromFloat(fl);
    return if (@mod(fi, 2) == 0) fl else fl + 1;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const t = std.testing;

test "fp32 identity roundtrip" {
    const vals = [_]f32{ 0, 1, -1, 3.14159, -2.5e10, 1e-20 };
    var bytes: [vals.len * 4]u8 = undefined;
    try fromF32(.fp32, &vals, &bytes);
    var back: [vals.len]f32 = undefined;
    try toF32(.fp32, &bytes, &back);
    try t.expectEqualSlices(f32, &vals, &back);
}

test "fp16 roundtrip within half precision" {
    const vals = [_]f32{ 0, 1, -1, 0.5, 100.0, -0.0625 };
    var bytes: [vals.len * 2]u8 = undefined;
    try fromF32(.fp16, &vals, &bytes);
    var back: [vals.len]f32 = undefined;
    try toF32(.fp16, &bytes, &back);
    for (vals, back) |v, b| try t.expectApproxEqRel(v, b, 0.001);
}

test "bf16 roundtrip and known encodings" {
    // 1.0 = 0x3F80_0000 -> bf16 0x3F80
    try t.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0));
    // -2.0 = 0xC000_0000 -> 0xC000
    try t.expectEqual(@as(u16, 0xC000), f32ToBf16(-2.0));
    const vals = [_]f32{ 0, 1, -1, 3.14159, 100.0, -0.001 };
    var bytes: [vals.len * 2]u8 = undefined;
    try fromF32(.bf16, &vals, &bytes);
    var back: [vals.len]f32 = undefined;
    try toF32(.bf16, &bytes, &back);
    // bf16 has ~2-3 decimal digits; relative error <~1%
    for (vals, back) |v, b| {
        if (v == 0) {
            try t.expectEqual(@as(f32, 0), b);
        } else {
            try t.expectApproxEqRel(v, b, 0.01);
        }
    }
}

test "bf16 round-to-nearest-even direction" {
    // A value whose low 16 bits are exactly 0x8000 rounds to even.
    // 1.0 + 1 ulp-of-bf16/2 boundary: construct bits = 0x3F80_8000 -> ties to even (0x3F80).
    const v: f32 = @bitCast(@as(u32, 0x3F80_8000));
    try t.expectEqual(@as(u16, 0x3F80), f32ToBf16(v)); // even neighbor
    const v2: f32 = @bitCast(@as(u32, 0x3F81_8000));
    try t.expectEqual(@as(u16, 0x3F82), f32ToBf16(v2)); // rounds up to even
}

test "fp8 e4m3 roundtrip representable values" {
    // Exactly representable e4m3 values: 1.0, 2.0, 0.5, 448 (max), 0.0
    const vals = [_]f32{ 0, 1, 2, 0.5, -1, 448, -448 };
    var bytes: [vals.len]u8 = undefined;
    try fromF32(.fp8_e4m3, &vals, &bytes);
    var back: [vals.len]f32 = undefined;
    try toF32(.fp8_e4m3, &bytes, &back);
    try t.expectEqualSlices(f32, &vals, &back);
}

test "fp8 e4m3 saturates above max" {
    var bytes: [1]u8 = undefined;
    try fromF32(.fp8_e4m3, &[_]f32{1000.0}, &bytes);
    var back: [1]f32 = undefined;
    try toF32(.fp8_e4m3, &bytes, &back);
    try t.expectEqual(@as(f32, 448), back[0]);
}

test "fp8 e5m2 roundtrip representable values" {
    const vals = [_]f32{ 0, 1, 2, 0.5, -1, 4, -0.25 };
    var bytes: [vals.len]u8 = undefined;
    try fromF32(.fp8_e5m2, &vals, &bytes);
    var back: [vals.len]f32 = undefined;
    try toF32(.fp8_e5m2, &bytes, &back);
    try t.expectEqualSlices(f32, &vals, &back);
}

test "int8 is unsupported" {
    try t.expectError(error.UnsupportedDtype, Dtype.byteSize(.int8));
    var out: [1]f32 = undefined;
    try t.expectError(error.UnsupportedDtype, toF32(.int8, &[_]u8{0}, &out));
}

test "size mismatch rejected" {
    var out: [2]f32 = undefined;
    try t.expectError(error.SizeMismatch, toF32(.fp32, &[_]u8{ 0, 0, 0, 0 }, &out));
}
