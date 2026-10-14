//! End-to-end check on real Fast-WAM KV: read a .krvi (calibrate.py output) + a raw
//! f32 tensor dump, encode each tensor as a frame against the cnit basis, decode, and
//! report fidelity (rel error, cosine) + REAL coded bytes vs raw.
//!
//! This is the honest-numbers step: it proves (1) Python-written cnit bytes parse in
//! Zig, (2) the full encode->decode path reconstructs real KV within tolerance, and
//! (3) real coded size, not an H0 estimate.
//!
//! Usage: e2e <fixture.krvi> <fixture.f32> [s]

const std = @import("std");
const krvsh = @import("krvsh");
const kbox = krvsh.krvsh_box;
const kinit = krvsh.krvsh_init;
const kframe = krvsh.krvsh_frame;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.arena.allocator();
    const args = try init.minimal.args.toSlice(alloc);

    var buf: [512]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    const out = &w.interface;

    if (args.len < 3) {
        try out.print("usage: e2e <krvi> <f32> [s]\n", .{});
        try out.flush();
        return;
    }
    const s: f32 = if (args.len >= 4) try std.fmt.parseFloat(f32, args[3]) else 2.0;

    const krvi = try readFile(io, alloc, args[1]);
    const raw = try readFile(io, alloc, args[2]);

    // Parse marker + cnit.
    var pos: usize = try kbox.parseMarker(krvi);
    const cnit = try kbox.parseCnit(krvi[pos..]);
    pos += cnit.total_size;
    const view = try kinit.parsePayload(cnit.cpay);
    const D: usize = view.header.d;
    const R: usize = view.header.r;

    // Parse the raw tensor dump header: n u32, T u32, D u32.
    const n_tensors = std.mem.readInt(u32, raw[0..4], .little);
    const T = std.mem.readInt(u32, raw[4..8], .little);
    const Draw = std.mem.readInt(u32, raw[8..12], .little);
    if (Draw != D) return error.DimMismatch;

    try out.print("cnit: D={d} R={d} n_tensors(view)={d}  dump: n={d} T={d}\n", .{
        D, R, view.header.tensorCount(), n_tensors, T,
    });

    var data_off: usize = 12;
    const tensor_floats = @as(usize, T) * D;

    var total_raw_bytes: usize = 0;
    var total_coded_bytes: usize = 0;
    var worst_rel: f64 = 0;

    var resolver = kbox.InitResolver.init(alloc);
    try resolver.addCnit(cnit);

    for (0..n_tensors) |ti| {
        const x = try alloc.alloc(f32, tensor_floats);
        for (0..tensor_floats) |i| {
            x[i] = @bitCast(std.mem.readInt(u32, raw[data_off + i * 4 .. data_off + i * 4 + 4][0..4], .little));
        }
        data_off += tensor_floats * 4;

        const blk = view.block(ti);

        // encode this tensor's fpay
        var fpay = std.array_list.Managed(u8).init(alloc);
        try kframe.encodeTensor(alloc, &fpay, x, blk.mean, blk.basis, D, R, s);

        // wrap in a kfrm, then parse it back (exercise the box path)
        var frame_box = std.array_list.Managed(u8).init(alloc);
        try kbox.writeKfrm(&frame_box, &cnit.init_id, @intCast(ti), 0, fpay.items);
        const kv = try kbox.parseKfrm(frame_box.items);

        // resolve init (must match) and decode
        const cpay = try resolver.resolve(&kv.init_id);
        _ = cpay;
        const payload = try kframe.parseTensor(kv.fpay);
        const xhat = try alloc.alloc(f32, tensor_floats);
        try kframe.decodeTensor(alloc, payload, blk.mean, blk.basis, D, R, T, xhat);

        // fidelity
        var num: f64 = 0; // ||x - xhat||^2
        var den: f64 = 0; // ||x||^2
        var dot: f64 = 0;
        var nx: f64 = 0;
        var nh: f64 = 0;
        for (x, xhat) |xi, hi| {
            const diff = @as(f64, xi) - hi;
            num += diff * diff;
            den += @as(f64, xi) * xi;
            dot += @as(f64, xi) * hi;
            nx += @as(f64, xi) * xi;
            nh += @as(f64, hi) * hi;
        }
        const rel = if (den > 0) @sqrt(num / den) else 0;
        const cosine = if (nx > 0 and nh > 0) dot / (@sqrt(nx) * @sqrt(nh)) else 1;
        if (rel > worst_rel) worst_rel = rel;

        const raw_bytes = tensor_floats * 2; // bf16 source reference
        total_raw_bytes += raw_bytes;
        total_coded_bytes += fpay.items.len;

        try out.print("  tensor {d}: rel_err={d:.4} cosine={d:.5} coded={d}B (raw bf16 {d}B, {d:.1}x)\n", .{
            ti, rel, cosine, fpay.items.len, raw_bytes,
            @as(f64, @floatFromInt(raw_bytes)) / @as(f64, @floatFromInt(fpay.items.len)),
        });
    }

    const ratio = @as(f64, @floatFromInt(total_raw_bytes)) / @as(f64, @floatFromInt(total_coded_bytes));
    try out.print("TOTAL: raw(bf16)={d}B coded={d}B  real_ratio={d:.1}x  worst_rel_err={d:.4}  (s={d})\n", .{
        total_raw_bytes, total_coded_bytes, ratio, worst_rel, s,
    });
    try out.flush();
}

fn readFile(io: std.Io, alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .unlimited);
}
