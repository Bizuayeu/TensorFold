//! ERNIE 5.5's output gate (ernie5_5/hca.cu) under the contract: z times bf16(sigmoid(g)), rounded where torch rounds.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_hca_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const round_ = @import("act_round_check.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const aref = core.act_ref;
const hh = ernie.hca;

/// A planted fault (control entries): the sigmoid left in fp32 for the product, or taken of -g.
pub const Fault = enum { none, unrounded, flipped };

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda ernie5_5 hca tf_ernie55_out_gate (z bf16(sigmoid(g)), then bf16: torch's two roundings)", .symbol = "tf_ernie55_out_gate" },
    .{ .name = "control: ernie5_5 out gate with the sigmoid not rounded to bf16", .symbol = "tf_ctl_ernie55_out_gate_unrounded", .fault = .unrounded },
    .{ .name = "control: ernie5_5 out gate of -g", .symbol = "tf_ctl_ernie55_out_gate_flipped", .fault = .flipped },
};

/// rows x the gated width: HCA's 8,192 (a decode row, a verify window, a chunk), DSA's 16,384, a TP=2 rank's 4,096.
pub const served = &[_]c.Shape{ .{ .rows = 1, .n = 8192, .k = 1 }, .{ .rows = 4, .n = 8192, .k = 1 }, .{ .rows = 64, .n = 8192, .k = 1 }, .{ .rows = 64, .n = 16384, .k = 1 }, .{ .rows = 4, .n = 4096, .k = 1 } };

/// No sums: each output is one product of two values.
fn noSums(_: c.Shape) usize {
    return 0;
}

/// ops: the fp32 sigmoid's expf (2 ulps, 4 units), add and division, and 1 for their cross term with mids, its bf16.
pub fn spec(e: Entry) c.Spec {
    // edges: widths about a block's 2,048 values (8 a thread), so a launch's last block runs short
    const tiles = [3]c.Tile{ .{}, .{ .size = 2048, .step = 8, .min = 8 }, .{} };
    return .{ .name = e.name, .accum = .none, .depth = noSums, .ops = 7, .mids = 1, .out = .bf16, .tiles = tiles, .served = if (e.fault == .none) served else served[0..1] };
}

/// z in [-2, 2); logits in [-8, 8), every 7th from 3 five times wider (to 40 either way, inside expf's fp32 range).
fn fill(z: []u16, gl: []u16, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (z, gl, 0..) |*zv, *gv, i| {
        zv.* = c.toBf16(rng.float(f32) * 4 - 2);
        gv.* = c.toBf16((rng.float(f32) * 16 - 8) * (if (i % 7 == 3) @as(f32, 5) else 1));
    }
}

pub const OutGate = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    n: usize = 0,
    z: []u16 = &.{}, // bf16
    gl: []u16 = &.{}, // the gate's logits, bf16
    dev: [2]u64 = @splat(0), // z, gl
    out: u64 = 0,

    const vt = g.hooks(OutGate);

    pub fn kernel(self: *OutGate) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    fn free(self: *OutGate) void {
        const a = self.rig.gpu.gpa;
        a.free(self.z);
        a.free(self.gl);
        self.z = &.{};
        self.gl = &.{};
    }

    pub fn deinit(self: *OutGate) void {
        self.free();
        self.rig.deinit();
    }

    /// fill's values; gate 1 plants NaN and both infinities in z and in g, and z = inf where g = -inf (NaN).
    pub fn prepare(self: *OutGate, shape: c.Shape, fill_: g.Fill, seed: u64) !bool {
        const n = std.math.mul(usize, shape.rows, shape.n) catch return false;
        if (n == 0) return false;
        hh.gateCheck(n) catch return false;
        self.rig.reset();
        self.free();
        const a = self.rig.gpu.gpa;
        self.z = try a.alloc(u16, n);
        self.gl = try a.alloc(u16, n);
        fill(self.z, self.gl, seed);
        if (fill_ == .nonfinite) {
            const w = shape.n;
            const nan = std.math.nan(f32);
            const inf = std.math.inf(f32);
            const zs = [_]struct { usize, f32 }{ .{ w / 3, nan }, .{ w + w / 2, inf }, .{ 3 * w - 1, -inf }, .{ 4 * w + 7, inf } };
            const gs = [_]struct { usize, f32 }{ .{ 0, nan }, .{ 2 * w - 1, inf }, .{ 3 * w + w / 2, -inf }, .{ 4 * w + 7, -inf } };
            for (zs) |p| if (p[0] < n) {
                self.z[p[0]] = c.toBf16(p[1]);
            };
            for (gs) |p| if (p[0] < n) {
                self.gl[p[0]] = c.toBf16(p[1]);
            };
        }
        self.n = n;
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.z)), try self.rig.input(std.mem.sliceAsBytes(self.gl)) };
        self.out = try self.rig.output(n * 2);
        return true;
    }

    pub fn run(self: *OutGate) !void {
        try ernie.outGateWith(self.f, self.rig.s, self.dev[0], self.dev[1], self.out, self.n);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *OutGate) usize {
        return self.n;
    }

    /// z sigmoid(g) in float64.
    pub fn reference(self: *OutGate, want: []f64, mag: []f64) !void {
        for (want, self.z, self.gl) |*w, zb, gb| w.* = c.fromBf16(zb) * aref.sigmoid(c.fromBf16(gb));
        @memset(mag, 0);
    }

    /// torch's roundings (core's headGateRange): bf16(z bf16(sigmoid(g) (1 + t))) at t = -spread, spread, fp32's reach.
    pub fn intervals(self: *OutGate, low: []f64, high: []f64) !void {
        for (low, high, self.z, self.gl) |*l, *h, zb, gb| {
            const r = aref.headGateRange(c.fromBf16(zb), c.fromBf16(gb), round_.spread);
            l.* = r[0];
            h.* = r[1];
        }
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    hca: ernie.Hca,
    items: [entries.len]OutGate = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var m = try ernie.Hca.load(gpu.d);
        errdefer m.unload();
        set.* = .{ .gpa = gpu.gpa, .hca = m };
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.hca.mod.function(e.symbol) };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.hca.unload();
        set.gpa.destroy(set);
    }
};

/// The output gate's gates 1-4 and its controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x67617465, &t);
    std.debug.print("SUMMARY ernie5_5 out_gate: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "ernie5_5-out-gate contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the served shapes fit the launch, at HCA's 8,192, DSA's 16,384 and a TP=2 rank's 4,096" {
    for (served) |x| try hh.gateCheck(x.rows * x.n);
    for ([_]usize{ 8192, 16384, 4096 }) |w| {
        var seen = false;
        for (served) |x| seen = seen or x.n == w;
        try std.testing.expect(seen);
    }
}

test "the sigmoid's rounding shows: left out, over a tenth of served[0]'s outputs leave torch's ends, which hold its form" {
    const n = served[0].rows * served[0].n;
    const a = std.testing.allocator;
    const z = try a.alloc(u16, 2 * n);
    defer a.free(z);
    fill(z[0..n], z[n..], 1);
    var left: usize = 0;
    for (z[0..n], z[n..]) |zb, gb| {
        const x = c.fromBf16(zb);
        const y = c.fromBf16(gb);
        const r = aref.headGateRange(x, y, round_.spread);
        const own = aref.headGate(x, y, 0);
        try std.testing.expect(r[0] <= own and own <= r[1]);
        const plain = core.bf16_round.bf16(x * aref.sigmoid(y));
        if (plain < r[0] or plain > r[1]) left += 1;
    }
    try std.testing.expect(left * 10 > n);
}
