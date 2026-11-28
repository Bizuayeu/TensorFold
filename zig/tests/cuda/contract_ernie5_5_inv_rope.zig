//! ERNIE 5.5's inverse rope (ernie5_5/hca.cu) under the contract: the attention output's rope dims turned back.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_hca_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const hh = ernie.hca;

const hd = hh.head_dim;
const nh = hh.vha_heads;
/// The released config's compress_rope_theta and rope_scaling (YaRN, factor 16 over 65,536 positions).
const theta = 160000;
const factor = 16;
const original = 65536;

/// A planted fault (control entries): the pairs turned forward, or by the next position's angle.
pub const Fault = enum { none, forward, next_position };

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda ernie5_5 hca tf_ernie55_inv_rope (the output's GPT-J pairs turned back at the row's position)", .symbol = "tf_ernie55_inv_rope" },
    .{ .name = "control: ernie5_5 inv rope turned forward, the q's rope again", .symbol = "tf_ctl_ernie55_inv_rope_forward", .fault = .forward },
    .{ .name = "control: ernie5_5 inv rope at the next position", .symbol = "tf_ctl_ernie55_inv_rope_next_position", .fault = .next_position },
};

/// rows, the first row's position, the heads: the window filling, position 0, a chunk, past sincosf's fast range, 1M.
pub const served = &[_]c.Shape{ .{ .rows = 4, .n = 126, .k = nh }, .{ .rows = 1, .n = 0, .k = nh }, .{ .rows = 64, .n = 4000, .k = nh }, .{ .rows = 1, .n = 199_999, .k = nh }, .{ .rows = 2, .n = (1 << 20) - 2, .k = nh } };

/// No sums: a pair's two products and their sum, each output checked against float64 with its own roundings.
fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn spec(e: Entry) c.Spec {
    return .{ .name = e.name, .accum = .none, .depth = noSums, .out = .bf16, .served = if (e.fault == .none) served else served[0..1] };
}

pub const Inv = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    x: []u16 = &.{}, // [rows][64][512], bf16
    freq: [hh.rope / 2]f32 = undefined,
    extra: []f64 = &.{}, // per output: the allowance for its own roundings beyond the spec's bound
    dev: [3]u64 = @splat(0), // x, freq, pos
    out: u64 = 0,

    const vt = g.hooks(Inv);

    pub fn kernel(self: *Inv) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    fn free(self: *Inv) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.extra);
        self.x = &.{};
        self.extra = &.{};
    }

    pub fn deinit(self: *Inv) void {
        self.free();
        self.rig.deinit();
    }

    /// x in [-2, 2); gate 1 plants NaN in a nope dim, inf in a rope dim and -inf in the last.
    pub fn prepare(self: *Inv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != nh) return false;
        hh.invCheck(shape.rows) catch return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        self.x = try a.alloc(u16, shape.rows * nh * hd);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, f32 }{ .{ 5, std.math.nan(f32) }, .{ hd + hh.nope + 3, std.math.inf(f32) }, .{ self.x.len - 1, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < self.x.len) {
                self.x[p[0]] = c.toBf16(p[1]);
            };
        }
        self.freq = core.rope_yarn.yarn(theta, factor, original).freq;
        const pos0 = [1]i32{@intCast(shape.n)};
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.x)), try self.rig.input(std.mem.sliceAsBytes(&self.freq)), try self.rig.input(std.mem.sliceAsBytes(&pos0)) };
        self.out = try self.rig.output(self.x.len * 2);
        return true;
    }

    pub fn run(self: *Inv) !void {
        try ernie.invRopeWith(self.f, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.out, self.shape.rows);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Inv) usize {
        return self.x.len;
    }

    /// hca.turnBack in float64; mag: a pair's |x c| + |x' s| (a nope dim's |x|); extra: 8 units of those terms.
    pub fn reference(self: *Inv, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.extra);
        self.extra = &.{};
        self.extra = try a.alloc(f64, want.len);
        @memset(self.extra, 0);
        for (0..self.shape.rows) |r| {
            const pos: i64 = @intCast(self.shape.n + r);
            for (0..nh) |h| {
                const at = (r * nh + h) * hd;
                const o = want[at..][0..hd];
                for (o, self.x[at..][0..hd], mag[at..][0..hd]) |*w, b, *m| {
                    w.* = c.fromBf16(b);
                    m.* = @abs(w.*);
                }
                hh.turnBack(o, &self.freq, pos);
                for (self.freq, 0..) |f, l| {
                    const ang = hh.angle(f, pos);
                    const co = @abs(@cos(ang));
                    const si = @abs(@sin(ang));
                    const k = at + hh.nope + 2 * l;
                    const x0 = c.fromBf16(self.x[k]);
                    const x1 = c.fromBf16(self.x[k + 1]);
                    mag[k] = @abs(x0) * co + @abs(x1) * si;
                    mag[k + 1] = @abs(x1) * co + @abs(x0) * si;
                    self.extra[k] = 8 * 0x1p-24 * mag[k];
                    self.extra[k + 1] = 8 * 0x1p-24 * mag[k + 1];
                }
            }
        }
    }

    pub fn bounds(self: *Inv, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        _ = mag;
        for (allow, self.extra) |*l, x| l.* += x * (1 + 0x1p-7);
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    hca: ernie.Hca,
    items: [entries.len]Inv = undefined,

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

/// The inverse rope's gates 1-4 and its controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x696e76, &t);
    std.debug.print("SUMMARY ernie5_5 inv_rope: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "ernie5_5-inv-rope contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the served shapes fit the launch, the controls' first one away from position 0, the last ending at the 1M end" {
    for (served) |x| try hh.invCheck(x.rows);
    try std.testing.expect(served[0].n > 0);
    const last = served[served.len - 1];
    try std.testing.expectEqual(@as(usize, (1 << 20) - 1), last.n + last.rows - 1);
}
