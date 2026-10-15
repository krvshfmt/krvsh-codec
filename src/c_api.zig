//! C ABI for the krvsh codec (see include/krvsh.h).
//!
//! Flat C surface over the codec. Same conventions as the container C ABI:
//!   - Caller-owned buffers; nothing allocated here crosses the boundary.
//!   - Write/encode use the two-call size/fill protocol (out=null -> required size in
//!     *written + KRVSH_BUFFER_TOO_SMALL).
//!   - Status codes mirror the Zig error sets.
//!
//! This exposes the per-tensor codec primitives (encode/decode against an init's fp16
//! basis + mean, which the caller obtains from the container cnit's cpay). Container
//! framing (cnit/kfrm boxes) is the krvs container C ABI's job; a C consumer uses both
//! headers together.

const std = @import("std");
const kframe = @import("krvsh_frame.zig");
const kinit = @import("krvsh_init.zig");

const List = std.array_list.Managed(u8);

pub const Status = enum(c_int) {
    ok = 0,
    buffer_too_small = 1,
    short_buffer = 2,
    unknown_coder = 3,
    symbol_count_mismatch = 4,
    bad_metadata = 5,
    size_mismatch = 6,
    internal = 255,
};

fn mapErr(e: anyerror) Status {
    return switch (e) {
        error.ShortBuffer => .short_buffer,
        error.UnknownCoder => .unknown_coder,
        error.SymbolCountMismatch => .symbol_count_mismatch,
        error.BadMetadataLen => .bad_metadata,
        error.Truncated, error.Overflow => .short_buffer,
        else => .internal,
    };
}

/// Compute init_id = sha256(cpay)[:16] into out16 (must point to >= 16 bytes).
export fn krvsh_init_id(cpay: [*]const u8, cpay_len: usize, out16: [*]u8) Status {
    const id = kinit.initId(cpay[0..cpay_len]);
    @memcpy(out16[0..kinit.INIT_ID_LEN], &id);
    return .ok;
}

export fn krvsh_init_id_len() usize {
    return kinit.INIT_ID_LEN;
}

/// Encode one tensor's fpay (prefix + range-coded stream).
/// x_f32: row-major (T, D) f32 input. mean_fp16/basis_fp16: the init's fp16 blocks for
/// this tensor (mean = D fp16, basis = R*D fp16, little-endian). Two-call size protocol.
export fn krvsh_encode_tensor(
    x_f32: [*]const f32,
    t: usize,
    d: usize,
    r: usize,
    mean_fp16: [*]const u8,
    basis_fp16: [*]const u8,
    s: f32,
    out: ?[*]u8,
    out_cap: usize,
    written: *usize,
) Status {
    var list = List.init(std.heap.c_allocator);
    defer list.deinit();
    kframe.encodeTensor(
        std.heap.c_allocator,
        &list,
        x_f32[0 .. t * d],
        mean_fp16[0 .. d * 2],
        basis_fp16[0 .. r * d * 2],
        d,
        r,
        s,
    ) catch |e| return mapErr(e);
    written.* = list.items.len;
    if (out == null or out_cap < list.items.len) return .buffer_too_small;
    @memcpy(out.?[0..list.items.len], list.items);
    return .ok;
}

/// Decode one tensor's fpay into x_hat (row-major (T, D) f32, caller-allocated t*d).
export fn krvsh_decode_tensor(
    fpay: [*]const u8,
    fpay_len: usize,
    mean_fp16: [*]const u8,
    basis_fp16: [*]const u8,
    d: usize,
    r: usize,
    t: usize,
    x_hat: [*]f32,
) Status {
    const payload = kframe.parseTensor(fpay[0..fpay_len]) catch |e| return mapErr(e);
    kframe.decodeTensor(
        std.heap.c_allocator,
        payload,
        mean_fp16[0 .. d * 2],
        basis_fp16[0 .. r * d * 2],
        d,
        r,
        t,
        x_hat[0 .. t * d],
    ) catch |e| return mapErr(e);
    return .ok;
}

/// Compute a model fingerprint over ordered identity key/value pairs.
/// keys/vals are arrays of `n` NUL-terminated C strings. Writes 32 bytes to out32.
export fn krvsh_model_fingerprint(
    keys: [*]const [*:0]const u8,
    vals: [*]const [*:0]const u8,
    n: usize,
    out32: [*]u8,
) Status {
    var pairs = std.array_list.Managed([2][]const u8).init(std.heap.c_allocator);
    defer pairs.deinit();
    for (0..n) |i| {
        pairs.append(.{ std.mem.span(keys[i]), std.mem.span(vals[i]) }) catch return .internal;
    }
    const fp = kinit.modelFingerprint(pairs.items);
    @memcpy(out32[0..32], &fp);
    return .ok;
}

test {
    std.testing.refAllDecls(@This());
}
