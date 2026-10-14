//! Golden vectors: fixed inputs -> fixed committed output bytes.
//!
//! These lock byte-exact determinism of the codec across machines and Zig versions.
//! A cloud encoder and an edge decoder (and this repo across platforms, incl mosk)
//! MUST produce identical bytes. If a golden assertion fails, the codec changed its
//! wire output -- either an intended format change (regenerate + bump) or a bug.
//!
//! Golden bytes were generated once with tools/gen_golden logic and pasted here as hex.
//! Regenerate deliberately if the format changes; do not "fix" by editing goldens to
//! match a changed implementation without understanding why they moved.

const std = @import("std");
const krvsh = @import("krvsh");
const rc = krvsh.rangecoder;
const quant = krvsh.quant;
const kframe = krvsh.krvsh_frame;
const kinit = krvsh.krvsh_init;
const kbox = krvsh.krvsh_box;

const t = std.testing;

/// Parse a compile-time hex string into bytes at runtime into `out` (out.len*2 == hex.len).
fn hexInto(hex: []const u8, out: []u8) void {
    std.debug.assert(hex.len == out.len * 2);
    for (out, 0..) |*b, i| {
        b.* = (nibble(hex[i * 2]) << 4) | nibble(hex[i * 2 + 1]);
    }
}
fn nibble(c: u8) u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => unreachable,
    };
}

fn expectHex(golden_hex: []const u8, actual: []const u8) !void {
    const golden = try t.allocator.alloc(u8, golden_hex.len / 2);
    defer t.allocator.free(golden);
    hexInto(golden_hex, golden);
    try t.expectEqualSlices(u8, golden, actual);
}

test "golden: range coder" {
    const input = [_]u8{ 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 0, 1, 5, 0, 0, 0 };
    var out = std.array_list.Managed(u8).init(t.allocator);
    defer out.deinit();
    try rc.encodeBuf(&out, &input);
    try expectHex("0003aff53f627fb0e0fe00", out.items);

    // and it must decode back to the exact input
    const dec = try t.allocator.alloc(u8, input.len);
    defer t.allocator.free(dec);
    rc.decodeBuf(out.items, dec);
    try t.expectEqualSlices(u8, &input, dec);
}

test "golden: quantize" {
    const coef = [_]f32{ 0.5, 1.5, -2.5, 3.14159, -0.07, 0.0, 100.0, -0.33 };
    const delta = quant.fp16RoundTripped(0.1);
    try t.expectEqual(@as(u16, 0x2e66), quant.f32ToFp16Bits(delta));
    var q: [coef.len]i32 = undefined;
    quant.quantize(&coef, delta, &q);
    const want = [_]i32{ 5, 15, -25, 31, -1, 0, 1000, -3 };
    try t.expectEqualSlices(i32, &want, &q);
}

test "golden: frame fpay (identity basis)" {
    const D = 4;
    const R = 4;
    var basis_vals: [R * D]f32 = @splat(0);
    for (0..R) |i| basis_vals[i * D + i] = 1.0;
    const basis = try t.allocator.alloc(u8, R * D * 2);
    defer t.allocator.free(basis);
    for (basis_vals, 0..) |v, i| kinit.f32ToFp16LE(v, basis[i * 2 .. i * 2 + 2][0..2]);
    const mean = try t.allocator.alloc(u8, D * 2);
    defer t.allocator.free(mean);
    for (0..D) |i| kinit.f32ToFp16LE(0.0, mean[i * 2 .. i * 2 + 2][0..2]);

    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0 };
    var fpay = std.array_list.Managed(u8).init(t.allocator);
    defer fpay.deinit();
    try kframe.encodeTensor(t.allocator, &fpay, &x, mean, basis, D, R, 0.5);
    try expectHex("7a3d08000000010a0000000205c191fd6c94d181f7", fpay.items);
}

test "golden: frame fpay (vectorized path, D=R=16)" {
    // D=16 exercises the VLEN=8 vectorized dot/axpy (2 lanes). Locks the fixed
    // reduction order so a change to vectorization is caught across machines.
    const D = 16;
    const R = 16;
    var basis_vals: [R * D]f32 = @splat(0);
    for (0..R) |i| basis_vals[i * D + i] = 1.0;
    const basis = try t.allocator.alloc(u8, R * D * 2);
    defer t.allocator.free(basis);
    for (basis_vals, 0..) |v, i| kinit.f32ToFp16LE(v, basis[i * 2 .. i * 2 + 2][0..2]);
    const mean = try t.allocator.alloc(u8, D * 2);
    defer t.allocator.free(mean);
    for (0..D) |i| kinit.f32ToFp16LE(0.0, mean[i * 2 .. i * 2 + 2][0..2]);

    var x: [2 * D]f32 = undefined;
    for (0..2 * D) |i| x[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16)) * 0.5;
    var fpay = std.array_list.Managed(u8).init(t.allocator);
    defer fpay.deinit();
    try kframe.encodeTensor(t.allocator, &fpay, &x, mean, basis, D, R, 0.5);
    try expectHex("a04020000000011600000005050450a76146948a1d13361a2ffbf007ef40cd5fe8", fpay.items);
}

test "golden: cnit box + init_id" {
    const h = kinit.InitHeader{
        .model_fingerprint = @splat(0),
        .dtype = .bf16,
        .n_layers = 1,
        .d = 2,
        .r = 1,
    };
    const mean = try t.allocator.alloc(u8, 4);
    defer t.allocator.free(mean);
    kinit.f32ToFp16LE(0.5, mean[0..2]);
    kinit.f32ToFp16LE(-0.5, mean[2..4]);
    const basis = try t.allocator.alloc(u8, 4);
    defer t.allocator.free(basis);
    kinit.f32ToFp16LE(1.0, basis[0..2]);
    kinit.f32ToFp16LE(0.0, basis[2..4]);
    const blocks = [_]kinit.TensorBlock{
        .{ .mean = mean, .basis = basis, .sigma = null },
        .{ .mean = mean, .basis = basis, .sigma = null },
    };
    const cpay = try kinit.buildPayload(t.allocator, h, "{}", &blocks);
    defer t.allocator.free(cpay);

    try expectHex(
        "000000000000000000000000000000000000000000000000000000000000000001000000010002000200000001000000020000007b7d003800b8003c0000003800b8003c0000",
        cpay,
    );
    const id = kinit.initId(cpay);
    try expectHex("a215ba1f33739e660d19b7b25e06b710", &id);

    var cbox = std.array_list.Managed(u8).init(t.allocator);
    defer cbox.deinit();
    try kbox.writeCnit(&cbox, cpay);
    try expectHex(
        "00000090636e6974010000010000001463646369010000016b727673010000000000001e696e6964010000010110a215ba1f33739e660d19b7b25e06b710000000526370617901000001000000000000000000000000000000000000000000000000000000000000000001000000010002000200000001000000020000007b7d003800b8003c0000003800b8003c0000",
        cbox.items,
    );
}
