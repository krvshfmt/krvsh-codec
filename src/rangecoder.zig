//! Range coder with an adaptive order-0 model over 256 byte symbols.
//!
//! This is the real-bytes entropy backend (krvsh-codec-spec.md §7). The research
//! reported `ratio = 16*N / (H0(q)*N)`, an order-0 entropy ESTIMATE. This coder emits
//! actual bytes, so the codec can report real coded size and the estimate-vs-actual
//! gap.
//!
//! Design: a classic 32-bit byte-wise range coder (Subbotin-style) with carry
//! handling, driven by a cumulative-frequency model. The model is adaptive order-0:
//! all 256 symbols start at frequency 1; after coding a symbol its count is bumped and
//! the total is rescaled when it would exceed a cap. Encoder and decoder update the
//! model identically, so no table is transmitted.
//!
//! Determinism: integer-only, fixed constants. The same input bytes produce the same
//! coded bytes on any machine. Locked by golden vectors (added with the codec).

const std = @import("std");

const TOP: u32 = 1 << 24; // emit/consume a byte when range drops below this
const BOT: u32 = 1 << 16; // underflow floor to keep range large enough to divide

/// Adaptive order-0 frequency model over 256 symbols.
/// Total frequency is kept < MAX_TOTAL so that `range / total` never underflows the
/// coder's precision. MAX_TOTAL <= BOT guarantees range (>= BOT after renorm) divided
/// by total stays >= 1.
pub const Model = struct {
    pub const N_SYM = 256;
    pub const MAX_TOTAL: u32 = BOT; // 65536
    pub const INC: u32 = 24; // frequency bump per observed symbol

    freq: [N_SYM]u16,
    total: u32,

    pub fn init() Model {
        var m: Model = .{ .freq = undefined, .total = N_SYM };
        for (&m.freq) |*f| f.* = 1;
        return m;
    }

    /// Cumulative frequency below `sym`, and the frequency of `sym`.
    fn cum(self: *const Model, sym: u8) struct { lo: u32, f: u32 } {
        var lo: u32 = 0;
        var i: usize = 0;
        while (i < sym) : (i += 1) lo += self.freq[i];
        return .{ .lo = lo, .f = self.freq[sym] };
    }

    /// Given a scaled value in [0, total), find the symbol whose cumulative range
    /// contains it. Returns the symbol plus its (lo, f).
    fn find(self: *const Model, scaled: u32) struct { sym: u8, lo: u32, f: u32 } {
        var lo: u32 = 0;
        var sym: usize = 0;
        while (sym < N_SYM) : (sym += 1) {
            const f = self.freq[sym];
            if (lo + f > scaled) break;
            lo += f;
        }
        return .{ .sym = @intCast(sym), .lo = lo, .f = self.freq[sym] };
    }

    fn update(self: *Model, sym: u8) void {
        self.freq[sym] += INC;
        self.total += INC;
        if (self.total >= MAX_TOTAL) self.rescale();
    }

    fn rescale(self: *Model) void {
        var new_total: u32 = 0;
        for (&self.freq) |*f| {
            // Halve but keep every symbol >= 1 so nothing becomes impossible to code.
            f.* = @intCast((@as(u32, f.*) + 1) >> 1);
            new_total += f.*;
        }
        self.total = new_total;
    }
};

/// Carryless range coder (Subbotin). `low`/`range` are 32-bit; renormalization emits a
/// byte either when the range falls below TOP or when the top bytes of `low` and
/// `low+range` agree (so that byte is settled). Encoder and decoder renorm identically.
pub const Encoder = struct {
    low: u32 = 0,
    range: u32 = 0xFFFF_FFFF,
    out: *std.array_list.Managed(u8),

    pub fn init(out: *std.array_list.Managed(u8)) Encoder {
        return .{ .out = out };
    }

    fn renorm(self: *Encoder) !void {
        while ((self.low ^ (self.low +% self.range)) < TOP or
            (self.range < BOT and blk: {
                self.range = (0 -% self.low) & (BOT - 1);
                break :blk true;
            }))
        {
            try self.out.append(@intCast(self.low >> 24));
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    pub fn encode(self: *Encoder, model: *Model, sym: u8) !void {
        const c = model.cum(sym);
        const r = self.range / model.total;
        self.low +%= r * c.lo;
        self.range = r * c.f;
        try self.renorm();
        model.update(sym);
    }

    /// Flush the remaining state. Must be called once after the last symbol.
    pub fn finish(self: *Encoder) !void {
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            try self.out.append(@intCast(self.low >> 24));
            self.low <<= 8;
        }
    }
};

pub const Decoder = struct {
    low: u32 = 0,
    range: u32 = 0xFFFF_FFFF,
    code: u32 = 0,
    buf: []const u8,
    pos: usize = 0,

    pub fn init(buf: []const u8) Decoder {
        var d: Decoder = .{ .buf = buf };
        var i: usize = 0;
        while (i < 4) : (i += 1) d.code = (d.code << 8) | d.nextByte();
        return d;
    }

    fn nextByte(self: *Decoder) u32 {
        if (self.pos < self.buf.len) {
            const b = self.buf[self.pos];
            self.pos += 1;
            return b;
        }
        return 0;
    }

    fn renorm(self: *Decoder) void {
        while ((self.low ^ (self.low +% self.range)) < TOP or
            (self.range < BOT and blk: {
                self.range = (0 -% self.low) & (BOT - 1);
                break :blk true;
            }))
        {
            self.code = (self.code << 8) | self.nextByte();
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    pub fn decode(self: *Decoder, model: *Model) u8 {
        const r = self.range / model.total;
        var scaled: u32 = (self.code -% self.low) / r;
        if (scaled >= model.total) scaled = model.total - 1;
        const found = model.find(scaled);
        self.low +%= r * found.lo;
        self.range = r * found.f;
        self.renorm();
        model.update(found.sym);
        return found.sym;
    }
};

/// Convenience: encode a whole buffer with a fresh adaptive model.
pub fn encodeBuf(out: *std.array_list.Managed(u8), data: []const u8) !void {
    var model = Model.init();
    var enc = Encoder.init(out);
    for (data) |b| try enc.encode(&model, b);
    try enc.finish();
}

/// Convenience: decode `n` symbols from `coded` into `out` (len must be >= n).
pub fn decodeBuf(coded: []const u8, out: []u8) void {
    var model = Model.init();
    var dec = Decoder.init(coded);
    for (out) |*o| o.* = dec.decode(&model);
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const t = std.testing;

fn roundtrip(data: []const u8) !void {
    var coded = std.array_list.Managed(u8).init(t.allocator);
    defer coded.deinit();
    try encodeBuf(&coded, data);

    const dec = try t.allocator.alloc(u8, data.len);
    defer t.allocator.free(dec);
    decodeBuf(coded.items, dec);

    try t.expectEqualSlices(u8, data, dec);
}

test "roundtrip empty" {
    try roundtrip(&[_]u8{});
}

test "roundtrip single symbol" {
    try roundtrip(&[_]u8{42});
}

test "roundtrip all same byte" {
    const data: [1000]u8 = @splat(7);
    try roundtrip(&data);
}

test "roundtrip small arbitrary" {
    try roundtrip("the quick brown fox jumps over the lazy dog");
}

test "roundtrip full byte range" {
    var data: [256]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @intCast(i);
    try roundtrip(&data);
}

test "roundtrip random uniform" {
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    var data: [4096]u8 = undefined;
    rand.bytes(&data);
    try roundtrip(&data);
}

test "roundtrip peaky distribution" {
    // Mostly zeros with occasional small values -- the shape quantized KV symbols take
    // after zig-zag. This is where the coder should actually save bytes.
    var prng = std.Random.DefaultPrng.init(1234);
    const rand = prng.random();
    var data: [8192]u8 = undefined;
    for (&data) |*b| {
        const r = rand.float(f32);
        b.* = if (r < 0.80) 0 else if (r < 0.95) @as(u8, @intCast(rand.intRangeAtMost(u8, 1, 3))) else rand.int(u8);
    }
    try roundtrip(&data);
}

test "coded shorter than raw for peaky input" {
    var prng = std.Random.DefaultPrng.init(99);
    const rand = prng.random();
    var data: [16384]u8 = undefined;
    for (&data) |*b| {
        b.* = if (rand.float(f32) < 0.9) 0 else rand.intRangeAtMost(u8, 1, 5);
    }
    var coded = std.array_list.Managed(u8).init(t.allocator);
    defer coded.deinit();
    try encodeBuf(&coded, &data);
    // Heavily peaked -> should compress well below the raw size.
    try t.expect(coded.items.len < data.len / 2);
}

test "model rescales without losing symbols" {
    var m = Model.init();
    // Hammer one symbol until rescale triggers repeatedly.
    var i: usize = 0;
    while (i < 100000) : (i += 1) m.update(0);
    // Every symbol must remain codeable (freq >= 1).
    for (m.freq) |f| try t.expect(f >= 1);
    try t.expect(m.total < Model.MAX_TOTAL);
}
