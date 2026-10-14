//! krvsh <-> container glue. The krvsh codec is one codec that rides in the krvs
//! container (imported as the `krvs` module). This module supplies the krvsh codec
//! identity and the init-id hash, delegating all box framing to the container.
//!
//! Split rationale: the container (krvs) is codec-agnostic and lives in its own repo.
//! Everything codec-specific -- CODEC_FOURCC, init_id = sha256(cpay)[:16], and the
//! §9.4 verify-on-ingest -- lives here with the rest of the krvsh codec.

const std = @import("std");
const krvs = @import("krvs");
const init_mod = @import("krvsh_init.zig");

const List = std.array_list.Managed(u8);

pub const BuildError = krvs.container.BuildError;
pub const ParseError = krvs.container.ParseError || error{ UnknownCodec, InitIdMismatch };

/// Build an init-id for a krvsh cpay: sha256(cpay)[:16].
fn krvshInitId(cpay: []const u8, scratch: *[init_mod.INIT_ID_LEN]u8) krvs.container.InitId {
    scratch.* = init_mod.initId(cpay);
    return .{ .hash_algo = krvs.container.HASH_ALGO_SHA256_T16, .bytes = scratch };
}

/// Wrap a krvsh init payload (`cpay`) in a container `cnit` box with the krvsh codec id.
pub fn writeCnit(out: *List, cpay: []const u8) BuildError!void {
    var id_bytes: [init_mod.INIT_ID_LEN]u8 = undefined;
    const id = krvshInitId(cpay, &id_bytes);
    try krvs.container.writeCnit(out, &init_mod.CODEC_FOURCC, init_mod.CODEC_VERSION, id, cpay);
}

/// krvsh view over a parsed container cnit: fixed-size init_id + checked codec.
pub const CnitView = struct {
    init_id: [init_mod.INIT_ID_LEN]u8,
    codec_version: u32,
    cpay: []const u8,
    total_size: usize,
};

pub fn parseCnit(buf: []const u8) ParseError!CnitView {
    const v = try krvs.container.parseCnit(buf);
    if (!std.mem.eql(u8, &v.codec_fourcc, &init_mod.CODEC_FOURCC)) return error.UnknownCodec;
    if (v.init_id.bytes.len != init_mod.INIT_ID_LEN) return error.InitIdMismatch;
    var id: [init_mod.INIT_ID_LEN]u8 = undefined;
    @memcpy(&id, v.init_id.bytes);
    return .{ .init_id = id, .codec_version = v.codec_version, .cpay = v.cpay, .total_size = v.total_size };
}

/// Wrap a krvsh frame payload (`fpay`) in a container `kfrm` box.
pub fn writeKfrm(out: *List, init_id: *const [init_mod.INIT_ID_LEN]u8, frame_id: u64, prev_frame_id: u64, fpay: []const u8) BuildError!void {
    const id: krvs.container.InitId = .{ .hash_algo = krvs.container.HASH_ALGO_SHA256_T16, .bytes = init_id };
    try krvs.container.writeKfrm(out, id, frame_id, prev_frame_id, fpay);
}

pub const KfrmView = struct {
    init_id: [init_mod.INIT_ID_LEN]u8,
    frame_id: u64,
    prev_frame_id: u64,
    fpay: []const u8,
    total_size: usize,
};

pub fn parseKfrm(buf: []const u8) ParseError!KfrmView {
    const v = try krvs.container.parseKfrm(buf);
    if (v.init_id.bytes.len != init_mod.INIT_ID_LEN) return error.InitIdMismatch;
    var id: [init_mod.INIT_ID_LEN]u8 = undefined;
    @memcpy(&id, v.init_id.bytes);
    return .{ .init_id = id, .frame_id = v.frame_id, .prev_frame_id = v.prev_frame_id, .fpay = v.fpay, .total_size = v.total_size };
}

pub const writeMarker = krvs.container.writeMarker;
pub const parseMarker = krvs.container.parseMarker;

/// krvsh init resolver: wraps the container resolver and enforces the §9.4 hash check
/// (recompute init_id over cpay, reject on mismatch) at ingest, which the codec owns.
/// krvsh init resolver: fixed 16-byte-id keyed, enforces the §9.4 hash check (recompute
/// init_id over cpay, reject on mismatch) at ingest. krvsh's id is a fixed [16]u8 so we
/// key by value (no borrowed-key lifetime concerns); the generic container resolver is
/// still available for codecs with variable-width ids.
pub const InitResolver = struct {
    map: std.AutoHashMap([init_mod.INIT_ID_LEN]u8, []const u8),

    pub fn init(alloc: std.mem.Allocator) InitResolver {
        return .{ .map = std.AutoHashMap([init_mod.INIT_ID_LEN]u8, []const u8).init(alloc) };
    }
    pub fn deinit(self: *InitResolver) void {
        self.map.deinit();
    }

    /// Register a parsed cnit after verifying cpay hashes to its declared init_id.
    pub fn addCnit(self: *InitResolver, cnit: CnitView) ParseError!void {
        const recomputed = init_mod.initId(cnit.cpay);
        if (!std.mem.eql(u8, &recomputed, &cnit.init_id)) return error.InitIdMismatch;
        try self.map.put(cnit.init_id, cnit.cpay);
    }

    pub fn resolve(self: *const InitResolver, id: *const [init_mod.INIT_ID_LEN]u8) ParseError![]const u8 {
        return self.map.get(id.*) orelse error.InitNotResolved;
    }
};

// re-export the box 4CC constants some tooling references
pub const T_CNIT = krvs.container.T_CNIT;
pub const T_KFRM = krvs.container.T_KFRM;

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const ztest = std.testing;

fn tinyCpay(alloc: std.mem.Allocator) ![]u8 {
    const h = init_mod.InitHeader{ .model_fingerprint = @splat(0), .dtype = .bf16, .n_layers = 1, .d = 2, .r = 1 };
    const mean = try alloc.alloc(u8, 4);
    defer alloc.free(mean);
    init_mod.f32ToFp16LE(0.5, mean[0..2]);
    init_mod.f32ToFp16LE(-0.5, mean[2..4]);
    const basis = try alloc.alloc(u8, 4);
    defer alloc.free(basis);
    init_mod.f32ToFp16LE(1.0, basis[0..2]);
    init_mod.f32ToFp16LE(0.0, basis[2..4]);
    const blocks = [_]init_mod.TensorBlock{
        .{ .mean = mean, .basis = basis, .sigma = null },
        .{ .mean = mean, .basis = basis, .sigma = null },
    };
    return init_mod.buildPayload(alloc, h, "{\"model_id\":\"t/m\"}", &blocks);
}

test "glue cnit roundtrip + init_id" {
    const alloc = ztest.allocator;
    const cpay = try tinyCpay(alloc);
    defer alloc.free(cpay);
    const expect = init_mod.initId(cpay);
    var out = List.init(alloc);
    defer out.deinit();
    try writeCnit(&out, cpay);
    const v = try parseCnit(out.items);
    try ztest.expectEqualSlices(u8, &expect, &v.init_id);
    try ztest.expectEqualSlices(u8, cpay, v.cpay);
}

test "glue resolver verify + resolve" {
    const alloc = ztest.allocator;
    const cpay = try tinyCpay(alloc);
    defer alloc.free(cpay);
    var buf = List.init(alloc);
    defer buf.deinit();
    try writeCnit(&buf, cpay);
    const cnit = try parseCnit(buf.items);
    var r = InitResolver.init(alloc);
    defer r.deinit();
    try r.addCnit(cnit);
    try ztest.expectEqualSlices(u8, cpay, try r.resolve(&cnit.init_id));
    var tampered = cnit;
    tampered.init_id[0] ^= 0xFF;
    try ztest.expectError(error.InitIdMismatch, r.addCnit(tampered));
}

test "glue marker/kfrm stream" {
    const alloc = ztest.allocator;
    const cpay = try tinyCpay(alloc);
    defer alloc.free(cpay);
    var s = List.init(alloc);
    defer s.deinit();
    try writeMarker(&s);
    try writeCnit(&s, cpay);
    const id = init_mod.initId(cpay);
    try writeKfrm(&s, &id, 1, 0, "fpay");
    var pos: usize = try parseMarker(s.items);
    const cnit = try parseCnit(s.items[pos..]);
    pos += cnit.total_size;
    const kfrm = try parseKfrm(s.items[pos..]);
    pos += kfrm.total_size;
    try ztest.expectEqual(s.items.len, pos);
    try ztest.expectEqualSlices(u8, &cnit.init_id, &kfrm.init_id);
}
