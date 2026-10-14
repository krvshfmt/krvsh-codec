//! Profile krvsh encode/decode throughput on real KV tensors.
//!
//! Reads a .krvi (calibration) + .f32 (raw tensor dump), then times:
//!   encode: project + rms + quantize + zigzag/varint + range-code (per tensor)
//!   decode: range-decode + dequant + reconstruct (per tensor)
//! over many iterations, reporting median/min ms per full frame (all tensors) and
//! throughput in MB/s of raw KV (bf16-equivalent) processed.
//!
//! Calibration (SVD) is one-time/offline and is NOT profiled -- only the per-frame
//! hot path, which is what runs cloud-side (encode) and edge-side (decode).
//!
//! Build with -Doptimize=ReleaseFast for meaningful numbers. Usage:
//!   profile <init.krvi> <tensors.f32> [s] [iters]

const std = @import("std");
const krvsh = @import("krvsh");
const kbox = krvsh.krvsh_box;
const kinit = krvsh.krvsh_init;
const kframe = krvsh.krvsh_frame;

fn nowNs(io: std.Io) i128 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.arena.allocator();
    const args = try init.minimal.args.toSlice(alloc);

    var buf: [1024]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    const out = &w.interface;

    if (args.len < 3) {
        try out.print("usage: profile <init.krvi> <tensors.f32> [s] [iters]\n", .{});
        try out.flush();
        return;
    }
    const s: f32 = if (args.len >= 4) try std.fmt.parseFloat(f32, args[3]) else 2.0;
    const iters: usize = if (args.len >= 5) try std.fmt.parseInt(usize, args[4], 10) else 50;

    const krvi = try std.Io.Dir.cwd().readFileAlloc(io, args[1], alloc, .unlimited);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, args[2], alloc, .unlimited);

    const marker_len = try kbox.parseMarker(krvi);
    const cnit = try kbox.parseCnit(krvi[marker_len..]);
    const view = try kinit.parsePayload(cnit.cpay);
    const D: usize = view.header.d;
    const R: usize = view.header.r;

    const n_tensors = std.mem.readInt(u32, raw[0..4], .little);
    const T = std.mem.readInt(u32, raw[4..8], .little);
    const tensor_floats = @as(usize, T) * D;

    // Load tensors into f32 arrays once.
    const tensors = try alloc.alloc([]f32, n_tensors);
    var off: usize = 12;
    for (0..n_tensors) |ti| {
        const x = try alloc.alloc(f32, tensor_floats);
        for (0..tensor_floats) |i| {
            x[i] = @bitCast(std.mem.readInt(u32, raw[off + i * 4 .. off + i * 4 + 4][0..4], .little));
        }
        off += tensor_floats * 4;
        tensors[ti] = x;
    }

    const raw_bytes_per_frame = n_tensors * tensor_floats * 2; // bf16-equivalent

    // ---- encode timing ----
    // Pre-encode once to get the fpay set and coded size, and reuse buffers.
    var fpays = try alloc.alloc([]u8, n_tensors);
    var enc_times = try alloc.alloc(i128, iters);
    var coded_total: usize = 0;

    // warmup
    for (0..3) |_| {
        for (0..n_tensors) |ti| {
            var tmp = std.array_list.Managed(u8).init(alloc);
            try kframe.encodeTensor(alloc, &tmp, tensors[ti], view.block(ti).mean, view.block(ti).basis, D, R, s);
            tmp.deinit();
        }
    }
    for (0..iters) |it| {
        const t0 = nowNs(io);
        var frame_coded: usize = 0;
        for (0..n_tensors) |ti| {
            var tmp = std.array_list.Managed(u8).init(alloc);
            try kframe.encodeTensor(alloc, &tmp, tensors[ti], view.block(ti).mean, view.block(ti).basis, D, R, s);
            frame_coded += tmp.items.len;
            if (it == 0) {
                fpays[ti] = try alloc.dupe(u8, tmp.items);
            }
            tmp.deinit();
        }
        const t1 = nowNs(io);
        enc_times[it] = t1 - t0;
        coded_total = frame_coded;
    }

    // ---- decode timing (uses the fpays from the first encode) ----
    var dec_times = try alloc.alloc(i128, iters);
    const xhat = try alloc.alloc(f32, tensor_floats);
    for (0..3) |_| {
        for (0..n_tensors) |ti| {
            const p = try kframe.parseTensor(fpays[ti]);
            try kframe.decodeTensor(alloc, p, view.block(ti).mean, view.block(ti).basis, D, R, T, xhat);
        }
    }
    for (0..iters) |it| {
        const t0 = nowNs(io);
        for (0..n_tensors) |ti| {
            const p = try kframe.parseTensor(fpays[ti]);
            try kframe.decodeTensor(alloc, p, view.block(ti).mean, view.block(ti).basis, D, R, T, xhat);
        }
        const t1 = nowNs(io);
        dec_times[it] = t1 - t0;
    }

    const enc = stats(enc_times);
    const dec = stats(dec_times);

    try out.print("config: n_tensors={d} T={d} D={d} R={d} s={d} iters={d}\n", .{ n_tensors, T, D, R, s, iters });
    try out.print("raw per frame: {d} B ({d:.2} MB bf16-equiv), coded: {d} B ({d:.1}x)\n", .{
        raw_bytes_per_frame,
        @as(f64, @floatFromInt(raw_bytes_per_frame)) / 1e6,
        coded_total,
        @as(f64, @floatFromInt(raw_bytes_per_frame)) / @as(f64, @floatFromInt(coded_total)),
    });
    try report(out, "ENCODE", enc, raw_bytes_per_frame);
    try report(out, "DECODE", dec, raw_bytes_per_frame);
    try out.flush();
}

const Stat = struct { median_ns: i128, min_ns: i128 };

fn stats(times: []i128) Stat {
    std.mem.sort(i128, times, {}, std.sort.asc(i128));
    return .{ .median_ns = times[times.len / 2], .min_ns = times[0] };
}

fn report(out: *std.Io.Writer, label: []const u8, st: Stat, raw_bytes: usize) !void {
    const med_ms = @as(f64, @floatFromInt(st.median_ns)) / 1e6;
    const min_ms = @as(f64, @floatFromInt(st.min_ns)) / 1e6;
    const mbps = (@as(f64, @floatFromInt(raw_bytes)) / 1e6) / (med_ms / 1e3);
    try out.print("{s}: median {d:.3} ms  min {d:.3} ms  ({d:.0} MB/s raw KV)\n", .{ label, med_ms, min_ms, mbps });
}
