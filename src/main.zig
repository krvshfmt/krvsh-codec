//! krvsh CLI.
//!
//! Subcommands (see docs/container-spec.md, docs/krvsh-codec-spec.md):
//!   version                              print container version
//!   info   <file>                        inspect a .krvi / .krvf (box tree, ids, meta)
//!   encode <init.krvi> <tensors.f32> <out.krvf> [s]
//!                                        encode each tensor as a frame against the init
//!   decode <in.krvf> <init.krvi> <out.f32>
//!                                        resolve the init and decode frames to raw f32
//!
//! Calibration (fitting the PCA basis) is the offline numpy tool tools/calibrate.py;
//! the CLI owns the per-frame hot path, matching the design split.
//!
//! The .f32 tensor interchange format (shared with calibrate.py --dump-tensors):
//!   u32 n_tensors | u32 T | u32 D | then n_tensors * (T*D) f32-LE values.

const std = @import("std");
const Io = std.Io;
const krvsh = @import("krvsh");
const kbox = krvsh.krvsh_box;
const kinit = krvsh.krvsh_init;
const kframe = krvsh.krvsh_frame;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.arena.allocator();
    const args = try init.minimal.args.toSlice(alloc);

    var obuf: [4096]u8 = undefined;
    var ow = Io.File.stdout().writer(io, &obuf);
    const out = &ow.interface;
    var ebuf: [1024]u8 = undefined;
    var ew = Io.File.stderr().writer(io, &ebuf);
    const err = &ew.interface;

    if (args.len < 2) {
        try err.print("usage: krvsh <version|info|encode|decode> ...\n", .{});
        try err.flush();
        return error.Usage;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "version")) {
        try out.print("krvs container v{d}\n", .{krvsh.CONTAINER_VERSION});
        try out.flush();
    } else if (std.mem.eql(u8, cmd, "info")) {
        try cmdInfo(io, alloc, out, err, args);
    } else if (std.mem.eql(u8, cmd, "encode")) {
        try cmdEncode(io, alloc, out, err, args);
    } else if (std.mem.eql(u8, cmd, "decode")) {
        try cmdDecode(io, alloc, out, err, args);
    } else {
        try err.print("unknown command: {s}\n", .{cmd});
        try err.flush();
        return error.UnknownCommand;
    }
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

fn cmdInfo(io: Io, alloc: std.mem.Allocator, out: *Io.Writer, err: *Io.Writer, args: []const [:0]const u8) !void {
    if (args.len < 3) {
        try err.print("usage: krvsh info <file.krvi|file.krvf>\n", .{});
        try err.flush();
        return error.Usage;
    }
    const data = try readFile(io, alloc, args[2]);
    var pos: usize = try kbox.parseMarker(data);
    try out.print("krvs marker ok ({d} bytes consumed)\n", .{pos});

    while (pos < data.len) {
        const b = try krvsh.krvs.box.parse(data[pos..]);
        if (b.typeEql(kbox.T_CNIT)) {
            const cnit = try kbox.parseCnit(data[pos..]);
            const view = try kinit.parsePayload(cnit.cpay);
            try out.print("cnit: init_id=", .{});
            try printHex(out, &cnit.init_id);
            try out.print(" codec_v={d} D={d} R={d} n_layers={d} tpl={d} dtype={d} cpay={d}B\n", .{
                cnit.codec_version, view.header.d, view.header.r, view.header.n_layers,
                view.header.tensors_per_layer, @intFromEnum(view.header.dtype), cnit.cpay.len,
            });
            try out.print("  metadata: {s}\n", .{view.metadata()});
        } else if (b.typeEql(kbox.T_KFRM)) {
            const kv = try kbox.parseKfrm(data[pos..]);
            try out.print("kfrm: frame_id={d} prev={d} init_id=", .{ kv.frame_id, kv.prev_frame_id });
            try printHex(out, &kv.init_id);
            try out.print(" fpay={d}B\n", .{kv.fpay.len});
        } else {
            try out.print("box '{s}' v{d} mandatory={} ({d}B)\n", .{
                b.fourcc, b.version, b.isMandatory(), b.total_size,
            });
        }
        pos += @intCast(b.total_size);
    }
    try out.flush();
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

fn cmdEncode(io: Io, alloc: std.mem.Allocator, out: *Io.Writer, err: *Io.Writer, args: []const [:0]const u8) !void {
    if (args.len < 5) {
        try err.print("usage: krvsh encode <init.krvi> <tensors.f32> <out.krvf> [s]\n", .{});
        try err.flush();
        return error.Usage;
    }
    const s: f32 = if (args.len >= 6) try std.fmt.parseFloat(f32, args[5]) else 2.0;

    const krvi = try readFile(io, alloc, args[2]);
    const raw = try readFile(io, alloc, args[3]);

    const marker_len = try kbox.parseMarker(krvi);
    const cnit = try kbox.parseCnit(krvi[marker_len..]);
    const view = try kinit.parsePayload(cnit.cpay);
    const D: usize = view.header.d;
    const R: usize = view.header.r;

    const n_tensors = std.mem.readInt(u32, raw[0..4], .little);
    const T = std.mem.readInt(u32, raw[4..8], .little);
    const Draw = std.mem.readInt(u32, raw[8..12], .little);
    if (Draw != D) return error.DimMismatch;
    if (n_tensors != view.header.tensorCount()) return error.TensorCountMismatch;

    var outbuf = std.array_list.Managed(u8).init(alloc);
    try kbox.writeMarker(&outbuf);

    const tensor_floats = @as(usize, T) * D;
    var off: usize = 12;
    var total_coded: usize = 0;
    for (0..n_tensors) |ti| {
        const x = try alloc.alloc(f32, tensor_floats);
        for (0..tensor_floats) |i| {
            x[i] = @bitCast(std.mem.readInt(u32, raw[off + i * 4 .. off + i * 4 + 4][0..4], .little));
        }
        off += tensor_floats * 4;
        const blk = view.block(ti);

        var fpay = std.array_list.Managed(u8).init(alloc);
        try kframe.encodeTensor(alloc, &fpay, x, blk.mean, blk.basis, D, R, s);
        total_coded += fpay.items.len;
        try kbox.writeKfrm(&outbuf, &cnit.init_id, @intCast(ti), 0, fpay.items);
    }

    try writeFile(io, args[4], outbuf.items);
    try out.print("encoded {d} tensors (T={d} D={d} R={d}, s={d}) -> {s}: {d}B frames, {d}B total\n", .{
        n_tensors, T, D, R, s, args[4], total_coded, outbuf.items.len,
    });
    try out.flush();
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

fn cmdDecode(io: Io, alloc: std.mem.Allocator, out: *Io.Writer, err: *Io.Writer, args: []const [:0]const u8) !void {
    if (args.len < 5) {
        try err.print("usage: krvsh decode <in.krvf> <init.krvi> <out.f32>\n", .{});
        try err.flush();
        return error.Usage;
    }
    const krvf = try readFile(io, alloc, args[2]);
    const krvi = try readFile(io, alloc, args[3]);

    // Resolve the init from the .krvi.
    const marker_len = try kbox.parseMarker(krvi);
    const cnit = try kbox.parseCnit(krvi[marker_len..]);
    const view = try kinit.parsePayload(cnit.cpay);
    const D: usize = view.header.d;
    const R: usize = view.header.r;

    var resolver = kbox.InitResolver.init(alloc);
    try resolver.addCnit(cnit);

    // Walk frames.
    var pos: usize = try kbox.parseMarker(krvf);
    // First pass: count frames and get T from the first frame's symbol count.
    var frames = std.array_list.Managed(kbox.KfrmView).init(alloc);
    while (pos < krvf.len) {
        const b = try krvsh.krvs.box.parse(krvf[pos..]);
        if (b.typeEql(kbox.T_KFRM)) {
            const kv = try kbox.parseKfrm(krvf[pos..]);
            _ = try resolver.resolve(&kv.init_id); // must resolve, else error
            try frames.append(kv);
        }
        pos += @intCast(b.total_size);
    }
    if (frames.items.len == 0) return error.NoFrames;

    // Derive T from the first tensor payload: n_symbols == T*R.
    const first = try kframe.parseTensor(frames.items[0].fpay);
    if (first.n_symbols % R != 0) return error.BadSymbolCount;
    const T: usize = first.n_symbols / R;
    const tensor_floats = T * D;

    var outbuf = std.array_list.Managed(u8).init(alloc);
    try appendU32(&outbuf, @intCast(frames.items.len));
    try appendU32(&outbuf, @intCast(T));
    try appendU32(&outbuf, @intCast(D));

    for (frames.items, 0..) |kv, ti| {
        const payload = try kframe.parseTensor(kv.fpay);
        const blk = view.block(ti);
        const xhat = try alloc.alloc(f32, tensor_floats);
        try kframe.decodeTensor(alloc, payload, blk.mean, blk.basis, D, R, T, xhat);
        for (xhat) |v| try appendU32(&outbuf, @bitCast(v));
    }

    try writeFile(io, args[4], outbuf.items);
    try out.print("decoded {d} frames (T={d} D={d}) -> {s}: {d}B\n", .{
        frames.items.len, T, D, args[4], outbuf.items.len,
    });
    try out.flush();
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn readFile(io: Io, alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .unlimited);
}

fn writeFile(io: Io, path: []const u8, data: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}

fn appendU32(list: *std.array_list.Managed(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try list.appendSlice(&b);
}

fn printHex(w: *Io.Writer, bytes: []const u8) !void {
    for (bytes) |b| try w.print("{x:0>2}", .{b});
}

test "main links against krvsh lib" {
    try std.testing.expect(krvsh.CONTAINER_VERSION == 1);
}
