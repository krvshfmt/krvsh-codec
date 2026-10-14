//! krvsh codec-init (`cnit`) payload: build and parse.
//!
//! The container `cnit` box (container-spec.md §5) holds child boxes `cdci` (codec id),
//! `inid` (init id), `cpay` (codec-opaque payload). This module owns the *krvsh*
//! `cpay` payload (krvsh-codec-spec.md §2-§4): fixed fields + JSON metadata + per-tensor
//! basis blocks, and the `init_id` content hash over that payload.
//!
//! Little-endian, per the krvsh codec convention. The container framing around this
//! (box headers, cdci/inid/cpay) is assembled in krvsh_box.zig.

const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const CODEC_FOURCC: [4]u8 = .{ 'k', 'r', 'v', 's' };
pub const CODEC_VERSION: u32 = 1;

pub const HASH_ALGO_SHA256_T16: u8 = 1;
pub const INIT_ID_LEN: usize = 16;

pub const CODEC_PCA_UNIFORM_QUANT: u16 = 1;

pub const Dtype = enum(u8) {
    bf16 = 0,
    fp16 = 1,
    fp32 = 2,
};

/// Fixed-field header of the krvsh init payload (krvsh-codec-spec.md §2.1), 48 bytes.
pub const InitHeader = struct {
    pub const SIZE: usize = 48;

    model_fingerprint: [32]u8,
    codec_id: u16 = CODEC_PCA_UNIFORM_QUANT,
    dtype: Dtype,
    exponent_q8: u8 = 0, // 0 => flat quant; beta*32 for jpeg-style per-component
    n_layers: u16,
    tensors_per_layer: u8 = 2,
    // reserved u8 = 0
    d: u32, // model dim D
    r: u32, // retained rank R

    pub fn write(self: InitHeader, out: []u8) void {
        std.debug.assert(out.len >= SIZE);
        @memcpy(out[0..32], &self.model_fingerprint);
        std.mem.writeInt(u16, out[32..34], self.codec_id, .little);
        out[34] = @intFromEnum(self.dtype);
        out[35] = self.exponent_q8;
        std.mem.writeInt(u16, out[36..38], self.n_layers, .little);
        out[38] = self.tensors_per_layer;
        out[39] = 0; // reserved
        std.mem.writeInt(u32, out[40..44], self.d, .little);
        std.mem.writeInt(u32, out[44..48], self.r, .little);
    }

    pub fn parse(buf: []const u8) error{ShortBuffer}!InitHeader {
        if (buf.len < SIZE) return error.ShortBuffer;
        var fp: [32]u8 = undefined;
        @memcpy(&fp, buf[0..32]);
        return .{
            .model_fingerprint = fp,
            .codec_id = std.mem.readInt(u16, buf[32..34], .little),
            .dtype = @enumFromInt(buf[34]),
            .exponent_q8 = buf[35],
            .n_layers = std.mem.readInt(u16, buf[36..38], .little),
            .tensors_per_layer = buf[38],
            .d = std.mem.readInt(u32, buf[40..44], .little),
            .r = std.mem.readInt(u32, buf[44..48], .little),
        };
    }

    pub fn tensorCount(self: InitHeader) usize {
        return @as(usize, self.n_layers) * self.tensors_per_layer;
    }

    pub fn hasSigma(self: InitHeader) bool {
        return self.exponent_q8 != 0;
    }
};

/// One per-tensor basis block, borrowing the init payload's byte ranges (no copy).
/// mean: fp16[D]; basis: fp16[R*D] (row-major Vt[:R]); sigma: fp16[R] iff hasSigma.
pub const TensorBlock = struct {
    mean: []const u8, // D * 2 bytes
    basis: []const u8, // R * D * 2 bytes
    sigma: ?[]const u8, // R * 2 bytes or null
};

/// fp16 helpers (little-endian) used by the calibration side and tests.
pub fn f32ToFp16LE(x: f32, out: *[2]u8) void {
    const h: f16 = @floatCast(x);
    std.mem.writeInt(u16, out, @bitCast(h), .little);
}
pub fn fp16LEToF32(b: *const [2]u8) f32 {
    const bits = std.mem.readInt(u16, b, .little);
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

/// Compute init_id = first 16 bytes of sha256 over the entire krvsh init payload
/// (header + metadata section + all tensor blocks), i.e. exactly the `cpay` bytes.
/// (krvsh-codec-spec.md §4.)
pub fn initId(cpay: []const u8) [INIT_ID_LEN]u8 {
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(cpay, &digest, .{});
    var id: [INIT_ID_LEN]u8 = undefined;
    @memcpy(&id, digest[0..INIT_ID_LEN]);
    return id;
}

/// Build the krvsh init payload (`cpay` bytes) from already-fp16-encoded tensor blocks.
/// `metadata_json` is the UTF-8 metadata blob (caller builds it; may be empty).
/// Blocks must be in layer/{k,v} order, length == header.tensorCount(), each sized to
/// match D/R and sigma-presence. Returns an owned slice (caller frees).
pub fn buildPayload(
    alloc: std.mem.Allocator,
    header: InitHeader,
    metadata_json: []const u8,
    blocks: []const TensorBlock,
) ![]u8 {
    std.debug.assert(blocks.len == header.tensorCount());
    const mean_bytes = @as(usize, header.d) * 2;
    const basis_bytes = @as(usize, header.r) * header.d * 2;
    const sigma_bytes: usize = if (header.hasSigma()) @as(usize, header.r) * 2 else 0;

    var total: usize = InitHeader.SIZE + 4 + metadata_json.len;
    total += blocks.len * (mean_bytes + basis_bytes + sigma_bytes);

    const out = try alloc.alloc(u8, total);
    errdefer alloc.free(out);

    var pos: usize = 0;
    header.write(out[pos .. pos + InitHeader.SIZE]);
    pos += InitHeader.SIZE;
    std.mem.writeInt(u32, out[pos .. pos + 4][0..4], @intCast(metadata_json.len), .little);
    pos += 4;
    @memcpy(out[pos .. pos + metadata_json.len], metadata_json);
    pos += metadata_json.len;

    for (blocks) |b| {
        std.debug.assert(b.mean.len == mean_bytes);
        std.debug.assert(b.basis.len == basis_bytes);
        @memcpy(out[pos .. pos + mean_bytes], b.mean);
        pos += mean_bytes;
        @memcpy(out[pos .. pos + basis_bytes], b.basis);
        pos += basis_bytes;
        if (header.hasSigma()) {
            const s = b.sigma orelse unreachable;
            std.debug.assert(s.len == sigma_bytes);
            @memcpy(out[pos .. pos + sigma_bytes], s);
            pos += sigma_bytes;
        }
    }
    std.debug.assert(pos == total);
    return out;
}

/// Parsed view over a krvsh init payload. Slices borrow `cpay`.
pub const InitView = struct {
    header: InitHeader,
    metadata_json: []const u8,
    cpay: []const u8, // full payload, for init_id recompute

    pub fn metadata(self: InitView) []const u8 {
        return self.metadata_json;
    }

    /// Return the i-th tensor block (0 <= i < tensorCount), slicing `cpay`.
    pub fn block(self: InitView, i: usize) TensorBlock {
        const h = self.header;
        const mean_bytes = @as(usize, h.d) * 2;
        const basis_bytes = @as(usize, h.r) * h.d * 2;
        const sigma_bytes: usize = if (h.hasSigma()) @as(usize, h.r) * 2 else 0;
        const block_bytes = mean_bytes + basis_bytes + sigma_bytes;

        const base = InitHeader.SIZE + 4 + self.metadata_json.len + i * block_bytes;
        var p = base;
        const mean = self.cpay[p .. p + mean_bytes];
        p += mean_bytes;
        const basis = self.cpay[p .. p + basis_bytes];
        p += basis_bytes;
        const sigma: ?[]const u8 = if (h.hasSigma()) self.cpay[p .. p + sigma_bytes] else null;
        return .{ .mean = mean, .basis = basis, .sigma = sigma };
    }

    /// Recompute init_id over the payload and compare to `expected`.
    pub fn verifyInitId(self: InitView, expected: *const [INIT_ID_LEN]u8) bool {
        const got = initId(self.cpay);
        return std.mem.eql(u8, &got, expected);
    }
};

pub const ParseError = error{ ShortBuffer, BadMetadataLen };

pub fn parsePayload(cpay: []const u8) ParseError!InitView {
    const header = InitHeader.parse(cpay) catch return error.ShortBuffer;
    if (cpay.len < InitHeader.SIZE + 4) return error.ShortBuffer;
    const meta_len = std.mem.readInt(u32, cpay[InitHeader.SIZE .. InitHeader.SIZE + 4][0..4], .little);
    const meta_start = InitHeader.SIZE + 4;
    if (meta_start + meta_len > cpay.len) return error.BadMetadataLen;
    const metadata_json = cpay[meta_start .. meta_start + meta_len];

    // Validate the declared tensor data fits.
    const mean_bytes = @as(usize, header.d) * 2;
    const basis_bytes = @as(usize, header.r) * header.d * 2;
    const sigma_bytes: usize = if (header.hasSigma()) @as(usize, header.r) * 2 else 0;
    const need = meta_start + meta_len + header.tensorCount() * (mean_bytes + basis_bytes + sigma_bytes);
    if (need > cpay.len) return error.ShortBuffer;

    return .{ .header = header, .metadata_json = metadata_json, .cpay = cpay };
}

// ---------------------------------------------------------------------------
// model fingerprint (krvsh-codec-spec.md §3.1)
// ---------------------------------------------------------------------------

/// Compute model_fingerprint = sha256( for each name in identity_keys, in order:
///   utf8(name) || 0x00 || utf8(value) || 0x00 ).
/// `pairs` is the ordered (name, value) list; a missing value is passed as "".
pub fn modelFingerprint(pairs: []const [2][]const u8) [32]u8 {
    var h = Sha256.init(.{});
    for (pairs) |kv| {
        h.update(kv[0]);
        h.update(&[_]u8{0});
        h.update(kv[1]);
        h.update(&[_]u8{0});
    }
    return h.finalResult();
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const t = std.testing;

fn encodeFp16Slice(alloc: std.mem.Allocator, vals: []const f32) ![]u8 {
    const out = try alloc.alloc(u8, vals.len * 2);
    for (vals, 0..) |v, i| f32ToFp16LE(v, out[i * 2 .. i * 2 + 2][0..2]);
    return out;
}

test "init header write/parse roundtrip" {
    const h = InitHeader{
        .model_fingerprint = @splat(0xAB),
        .dtype = .bf16,
        .n_layers = 30,
        .d = 3072,
        .r = 1024,
    };
    var buf: [InitHeader.SIZE]u8 = undefined;
    h.write(&buf);
    const p = try InitHeader.parse(&buf);
    try t.expectEqual(h.codec_id, p.codec_id);
    try t.expectEqual(Dtype.bf16, p.dtype);
    try t.expectEqual(@as(u16, 30), p.n_layers);
    try t.expectEqual(@as(u8, 2), p.tensors_per_layer);
    try t.expectEqual(@as(u32, 3072), p.d);
    try t.expectEqual(@as(u32, 1024), p.r);
    try t.expectEqualSlices(u8, &h.model_fingerprint, &p.model_fingerprint);
    try t.expect(!p.hasSigma());
    try t.expectEqual(@as(usize, 60), p.tensorCount());
}

test "build and parse a small init payload (flat, no sigma)" {
    const alloc = t.allocator;
    // tiny model: 1 layer, 2 tensors, D=4, R=2
    const D = 4;
    const R = 2;
    const h = InitHeader{
        .model_fingerprint = @splat(0),
        .dtype = .fp32,
        .n_layers = 1,
        .d = D,
        .r = R,
    };
    const mean = try encodeFp16Slice(alloc, &[_]f32{ 1, 2, 3, 4 });
    defer alloc.free(mean);
    const basis = try encodeFp16Slice(alloc, &[_]f32{ 0.5, -0.5, 0.25, 0.75, 1, 0, 0, 1 }); // R*D=8
    defer alloc.free(basis);

    const blocks = [_]TensorBlock{
        .{ .mean = mean, .basis = basis, .sigma = null },
        .{ .mean = mean, .basis = basis, .sigma = null },
    };
    const meta = "{\"model_id\":\"test/model\"}";
    const cpay = try buildPayload(alloc, h, meta, &blocks);
    defer alloc.free(cpay);

    const view = try parsePayload(cpay);
    try t.expectEqual(@as(u32, D), view.header.d);
    try t.expectEqualStrings(meta, view.metadata());

    const b0 = view.block(0);
    try t.expectEqualSlices(u8, mean, b0.mean);
    try t.expectEqualSlices(u8, basis, b0.basis);
    try t.expect(b0.sigma == null);

    // fp16 values survive
    try t.expectApproxEqAbs(@as(f32, 3), fp16LEToF32(b0.mean[4..6][0..2]), 0.01);
}

test "init payload with sigma (per-component)" {
    const alloc = t.allocator;
    const D = 3;
    const R = 2;
    const h = InitHeader{
        .model_fingerprint = @splat(0),
        .dtype = .fp16,
        .exponent_q8 = 16, // nonzero => sigma present
        .n_layers = 1,
        .tensors_per_layer = 1,
        .d = D,
        .r = R,
    };
    const mean = try encodeFp16Slice(alloc, &[_]f32{ 0, 0, 0 });
    defer alloc.free(mean);
    const basis = try encodeFp16Slice(alloc, &[_]f32{ 1, 0, 0, 0, 1, 0 }); // R*D=6
    defer alloc.free(basis);
    const sigma = try encodeFp16Slice(alloc, &[_]f32{ 10, 2 });
    defer alloc.free(sigma);

    const blocks = [_]TensorBlock{.{ .mean = mean, .basis = basis, .sigma = sigma }};
    const cpay = try buildPayload(alloc, h, "", &blocks);
    defer alloc.free(cpay);

    const view = try parsePayload(cpay);
    try t.expect(view.header.hasSigma());
    const b = view.block(0);
    try t.expect(b.sigma != null);
    try t.expectApproxEqAbs(@as(f32, 10), fp16LEToF32(b.sigma.?[0..2][0..2]), 0.01);
}

test "init_id is deterministic and content-addressed" {
    const alloc = t.allocator;
    const h = InitHeader{ .model_fingerprint = @splat(0), .dtype = .bf16, .n_layers = 1, .d = 2, .r = 1 };
    const mean = try encodeFp16Slice(alloc, &[_]f32{ 1, 1 });
    defer alloc.free(mean);
    const basis = try encodeFp16Slice(alloc, &[_]f32{ 1, 0 });
    defer alloc.free(basis);
    const blocks = [_]TensorBlock{ .{ .mean = mean, .basis = basis, .sigma = null }, .{ .mean = mean, .basis = basis, .sigma = null } };

    const cpay1 = try buildPayload(alloc, h, "{}", &blocks);
    defer alloc.free(cpay1);
    const cpay2 = try buildPayload(alloc, h, "{}", &blocks);
    defer alloc.free(cpay2);

    const id1 = initId(cpay1);
    const id2 = initId(cpay2);
    try t.expectEqualSlices(u8, &id1, &id2); // same content -> same id

    // different metadata -> different id
    const cpay3 = try buildPayload(alloc, h, "{\"x\":1}", &blocks);
    defer alloc.free(cpay3);
    const id3 = initId(cpay3);
    try t.expect(!std.mem.eql(u8, &id1, &id3));

    const view = try parsePayload(cpay1);
    try t.expect(view.verifyInitId(&id1));
    try t.expect(!view.verifyInitId(&id3));
}

test "model fingerprint matches a hand-computed digest and is order-sensitive" {
    const fp_a = modelFingerprint(&[_][2][]const u8{
        .{ "model_id", "lerobot/fastwam" },
        .{ "revision", "abc123" },
    });
    // recompute the same way
    var h = Sha256.init(.{});
    h.update("model_id");
    h.update(&[_]u8{0});
    h.update("lerobot/fastwam");
    h.update(&[_]u8{0});
    h.update("revision");
    h.update(&[_]u8{0});
    h.update("abc123");
    h.update(&[_]u8{0});
    const want = h.finalResult();
    try t.expectEqualSlices(u8, &want, &fp_a);

    // order matters
    const fp_b = modelFingerprint(&[_][2][]const u8{
        .{ "revision", "abc123" },
        .{ "model_id", "lerobot/fastwam" },
    });
    try t.expect(!std.mem.eql(u8, &fp_a, &fp_b));

    // missing value (empty) differs from present
    const fp_c = modelFingerprint(&[_][2][]const u8{
        .{ "model_id", "lerobot/fastwam" },
        .{ "revision", "" },
    });
    try t.expect(!std.mem.eql(u8, &fp_a, &fp_c));
}

test "parse rejects bad metadata length" {
    const alloc = t.allocator;
    const h = InitHeader{ .model_fingerprint = @splat(0), .dtype = .bf16, .n_layers = 1, .d = 2, .r = 1 };
    const mean = try encodeFp16Slice(alloc, &[_]f32{ 1, 1 });
    defer alloc.free(mean);
    const basis = try encodeFp16Slice(alloc, &[_]f32{ 1, 0 });
    defer alloc.free(basis);
    const blocks = [_]TensorBlock{ .{ .mean = mean, .basis = basis, .sigma = null }, .{ .mean = mean, .basis = basis, .sigma = null } };
    const cpay = try buildPayload(alloc, h, "{}", &blocks);
    defer alloc.free(cpay);

    // corrupt the metadata length to exceed the buffer
    const bad = try alloc.dupe(u8, cpay);
    defer alloc.free(bad);
    std.mem.writeInt(u32, bad[InitHeader.SIZE .. InitHeader.SIZE + 4][0..4], 0xFFFF, .little);
    try t.expectError(error.BadMetadataLen, parsePayload(bad));
}
