//! Opt-in wait before a prompt chunk when the hottest thermal zone is above a band or would rise past a ceiling.
const std = @import("std");

pub const default_root = "/sys/class/thermal";

/// Degrees. `ceiling_c` also waits for a chunk's rise; a family that sets it calls `Gate.startPrompt` at each prompt.
pub const Bands = struct { high_c: f64, low_c: f64, ceiling_c: ?f64 = null };

/// Asked after each pause: true ends the wait with error.Cancelled (the request was cancelled meanwhile).
pub const Stop = struct { ptr: *anyopaque, check: *const fn (*anyopaque) bool };

pub const Gate = struct {
    bands: ?Bands = null,
    waited_s: f64 = 0,
    /// Set after the first unreadable zone, so later chunks skip the read and the log.
    no_zone: bool = false,
    root: []const u8 = default_root,
    /// The gathered maximum the last chunk began at; null: the next chunk is a prompt's first.
    began: ?f64 = null,
    /// The same when that chunk began straight after another; null: after a rest (a wait, or the gap before a prompt).
    after: ?f64 = null,
    /// The last rise of a chunk begun straight after another, in degrees: what the ceiling expects of the next.
    loaded: f64 = 0,

    /// TF_HEAT_HIGH and TF_HEAT_LOW in degrees, both or neither (neither: no read); TF_HEAT_ROOT moves the zones.
    pub fn fromEnv() !Gate {
        const root = envSpan("TF_HEAT_ROOT") orelse default_root;
        return .{ .bands = try bandsFrom(envSpan("TF_HEAT_HIGH"), envSpan("TF_HEAT_LOW")), .root = root };
    }

    /// A prompt begins: its first chunk has no rise, and it begins after the gap before the prompt.
    pub fn startPrompt(g: *Gate) void {
        g.began = null;
        g.after = null;
    }

    /// `read` returns the gathered maximum in celsius. `sleep_fn` stands in for the 2 s pause.
    pub fn waitUntil(g: *Gate, read: *const fn (*anyopaque) anyerror!f64, sleep_fn: *const fn (*anyopaque) void, ctx: *anyopaque, stop: ?Stop) !void {
        const bands = g.bands orelse return;
        if (g.no_zone) return;
        var max_c = (try g.reading(read, ctx)) orelse return;
        const floor = g.floorFor(bands, max_c) orelse return;
        while (max_c > floor) {
            sleep_fn(ctx);
            g.waited_s += 2;
            if (stop) |s| if (s.check(s.ptr)) return error.Cancelled;
            max_c = (try g.reading(read, ctx)) orelse return;
        }
        g.began = max_c; // the next chunk begins after a rest
        g.after = null;
    }

    /// One gathered reading; null once no zone is readable (logged once, and the chunk runs).
    fn reading(g: *Gate, read: *const fn (*anyopaque) anyerror!f64, ctx: *anyopaque) !?f64 {
        return read(ctx) catch |err| {
            if (err != error.NoZone) return err;
            g.no_zone = true;
            noteNoZone(g.root);
            return null;
        };
    }

    /// Before a chunk at `top`: null to run now, else the reading its wait lasts until (low, or lower to fit a rise).
    fn floorFor(g: *Gate, b: Bands, top: f64) ?f64 {
        const inf = std.math.inf(f64);
        const rise: f64 = if (g.began) |at| (if (at == -inf) 0 else @max(0, top - at)) else 0;
        if (g.after) |at| {
            if (at != -inf and top != -inf) g.loaded = rise; // the last chunk began straight after another
        }
        const ahead: f64 = if (g.began == null) 0 else g.loaded;
        const ceiling = b.ceiling_c orelse inf;
        if (top <= b.high_c and top + ahead <= ceiling) {
            g.after = if (g.began == null) null else top;
            g.began = top;
            return null;
        }
        const floor = @min(b.low_c, ceiling - @max(rise, ahead));
        noteWait(b, top, ahead, floor);
        return floor;
    }

    /// The real pause, on this process's own hottest zone (`gatheredMax` takes several readings), until `stop` says so.
    pub fn beforePromptChunk(g: *Gate, io: std.Io, stop: ?Stop) !void {
        if (g.bands == null) return;
        const Ctx = struct {
            gate: *Gate,
            io: std.Io,
            fn read(ptr: *anyopaque) !f64 {
                const c: *@This() = @ptrCast(@alignCast(ptr));
                var dir = std.Io.Dir.cwd().openDir(c.io, c.gate.root, .{ .iterate = true }) catch return error.NoZone;
                defer dir.close(c.io);
                return gatheredMax(&.{try hottest(c.io, dir)});
            }
            fn sleep(ptr: *anyopaque) void {
                const c: *@This() = @ptrCast(@alignCast(ptr));
                std.Io.sleep(c.io, .fromMilliseconds(2000), .awake) catch {};
            }
        };
        var ctx = Ctx{ .gate = g, .io = io };
        try g.waitUntil(Ctx.read, Ctx.sleep, &ctx, stop);
    }
};

/// Env text is degrees. Both missing means unset.
pub fn bandsFrom(high: ?[]const u8, low: ?[]const u8) !?Bands {
    if (high == null and low == null) return null;
    const hi = high orelse return error.HeatBands;
    const lo = low orelse return error.HeatBands;
    const bands = Bands{ .high_c = try degrees(hi), .low_c = try degrees(lo) };
    if (bands.low_c > bands.high_c) return error.HeatBands;
    return bands;
}

/// TF_HEAT_CEILING's text on `bands`: in degrees, above the high band, and only with the bands.
pub fn ceilingFrom(bands: ?Bands, text: ?[]const u8) !?Bands {
    const t = text orelse return bands;
    var b = bands orelse return error.HeatBands;
    const c = try degrees(t);
    if (c <= b.high_c) return error.HeatBands;
    b.ceiling_c = c;
    return b;
}

pub fn degrees(text: []const u8) !f64 {
    return std.fmt.parseFloat(f64, std.mem.trim(u8, text, " \t\r\n"));
}

/// A zone file is millidegrees. 94000 is 94.0 celsius.
pub fn zoneCelsius(text: []const u8) !f64 {
    const milli = try std.fmt.parseInt(i64, std.mem.trim(u8, text, " \t\r\n"), 10);
    return @as(f64, @floatFromInt(milli)) / 1000.0;
}

pub fn gatheredMax(xs: []const f64) f64 {
    var m = xs[0];
    for (xs[1..]) |x| m = @max(m, x);
    return m;
}

/// Hottest `thermal_zone*/temp` under `dir`. Other names are ignored. No readable zone is an error.
pub fn hottest(io: std.Io, dir: std.Io.Dir) !f64 {
    var it = dir.iterate();
    var max: ?f64 = null;
    var path_buf: [96]u8 = undefined;
    var file_buf: [64]u8 = undefined;
    while (try it.next(io)) |e| {
        if (e.kind != .directory and e.kind != .sym_link) continue;
        if (!std.mem.startsWith(u8, e.name, "thermal_zone")) continue;
        const path = std.fmt.bufPrint(&path_buf, "{s}/temp", .{e.name}) catch continue;
        const text = dir.readFile(io, path, &file_buf) catch continue;
        const c = zoneCelsius(text) catch continue;
        max = if (max) |m| @max(m, c) else c;
    }
    return max orelse error.NoZone;
}

pub fn note(seconds_: f64) void {
    if (@import("builtin").is_test) return;
    std.log.info("heat_wait_s {d:.1}", .{seconds_});
}

/// Why a chunk waits, at the wait's start: above the high band, or its expected rise past the ceiling.
fn noteWait(b: Bands, top: f64, ahead: f64, floor: f64) void {
    if (@import("builtin").is_test) return;
    if (top > b.high_c) {
        std.log.info("heat: hottest zone {d:.1} C above {d} C; the prefill waits for {d:.1} C", .{ top, b.high_c, floor });
    } else {
        std.log.info("heat: hottest zone {d:.1} C plus a chunk's {d:.1} C rise, one run straight after another, above {d} C; the prefill waits for {d:.1} C", .{ top, ahead, b.ceiling_c orelse 0, floor });
    }
}

/// Test builds count the warning. A real process logs it.
var no_zone_notes: u32 = 0;

fn noteNoZone(root: []const u8) void {
    no_zone_notes += 1;
    if (@import("builtin").is_test) return;
    std.log.warn("no readable thermal zone under {s}; running without a heat wait", .{root});
}

fn envSpan(name: [*:0]const u8) ?[]const u8 {
    const p = std.c.getenv(name) orelse return null;
    const s = std.mem.span(p);
    if (s.len == 0) return null;
    return s;
}

test "bands parse as degrees and a lone value is refused" {
    try std.testing.expect((try bandsFrom(null, null)) == null);
    try std.testing.expectError(error.HeatBands, bandsFrom("92", null));
    try std.testing.expectError(error.HeatBands, bandsFrom(null, "88"));
    try std.testing.expectError(error.HeatBands, bandsFrom("88", "92"));
    const bands = (try bandsFrom("92", " 88\n")).?;
    try std.testing.expectEqual(@as(f64, 92), bands.high_c);
    try std.testing.expectEqual(@as(f64, 88), bands.low_c);
    try std.testing.expectEqual(@as(f64, 94), try zoneCelsius("94000\n"));
}

test "the chunk runs under the high band, and a hot reading waits until the low band" {
    const bands = Bands{ .high_c = 92, .low_c = 88 };
    var at = Gate{ .bands = bands };
    try std.testing.expectEqual(@as(?f64, null), at.floorFor(bands, 92)); // at the high band the chunk runs
    try std.testing.expectEqual(@as(?f64, 88), at.floorFor(bands, 92.5)); // above it, a wait until the low band
    try std.testing.expectEqual(@as(f64, 94), gatheredMax(&.{ 80, 94, 91 }));
    const Script = struct {
        temps: []const f64,
        i: usize = 0,
        sleeps: u32 = 0,
        fn read(ptr: *anyopaque) !f64 {
            const s: *@This() = @ptrCast(@alignCast(ptr));
            const t = s.temps[s.i];
            if (s.i + 1 < s.temps.len) s.i += 1;
            return t;
        }
        fn sleep(ptr: *anyopaque) void {
            @as(*@This(), @ptrCast(@alignCast(ptr))).sleeps += 1;
        }
    };
    var hot = Script{ .temps = &.{ 95, 93, 88 } };
    var gate = Gate{ .bands = bands };
    try gate.waitUntil(Script.read, Script.sleep, &hot, null);
    try std.testing.expectEqual(@as(u32, 2), hot.sleeps);
    try std.testing.expectEqual(@as(f64, 4), gate.waited_s);
    var cool = Script{ .temps = &.{90} };
    var idle = Gate{ .bands = bands };
    try idle.waitUntil(Script.read, Script.sleep, &cool, null);
    try std.testing.expectEqual(@as(u32, 0), cool.sleeps);
    var off = Gate{};
    try off.waitUntil(struct {
        fn read(_: *anyopaque) !f64 {
            return error.Read;
        }
    }.read, Script.sleep, &cool, null);
    try std.testing.expectEqual(@as(f64, 0), off.waited_s);
    const Cancelled = struct {
        fn yes(_: *anyopaque) bool {
            return true;
        }
    };
    var stuck = Script{ .temps = &.{ 95, 95, 95 } };
    var hot_gate = Gate{ .bands = bands };
    try std.testing.expectError(error.Cancelled, hot_gate.waitUntil(Script.read, Script.sleep, &stuck, .{ .ptr = &stuck, .check = Cancelled.yes }));
    try std.testing.expectEqual(@as(u32, 1), stuck.sleeps); // a cancel ends the wait after the pause it came in
}

/// A gate's gathered maxima, one a read (the last repeats), and the pauses taken.
const Readings = struct {
    temps: []const f64,
    i: usize = 0,
    sleeps: u32 = 0,
    fn read(ptr: *anyopaque) !f64 {
        const r: *Readings = @ptrCast(@alignCast(ptr));
        const t = r.temps[r.i];
        if (r.i + 1 < r.temps.len) r.i += 1;
        return t;
    }
    fn sleep(ptr: *anyopaque) void {
        @as(*Readings, @ptrCast(@alignCast(ptr))).sleeps += 1;
    }
    /// One chunk's gate over `temps`: the pauses it took.
    fn chunk(g: *Gate, temps: []const f64) !u32 {
        var r = Readings{ .temps = temps };
        try g.waitUntil(read, sleep, &r, null);
        return r.sleeps;
    }
};

test "a ceiling comes with both bands and above the high one" {
    const b = Bands{ .high_c = 92, .low_c = 88 };
    try std.testing.expect((try ceilingFrom(null, null)) == null);
    try std.testing.expectEqual(b, (try ceilingFrom(b, null)).?);
    try std.testing.expectEqual(@as(?f64, 93), (try ceilingFrom(b, " 93\n")).?.ceiling_c);
    try std.testing.expectError(error.HeatBands, ceilingFrom(null, "93"));
    try std.testing.expectError(error.HeatBands, ceilingFrom(b, "92"));
}

test "with a ceiling, a chunk begun straight after another waits until the last such rise fits under it" {
    var g = Gate{ .bands = .{ .high_c = 92, .low_c = 88, .ceiling_c = 93 } };
    g.startPrompt();
    try std.testing.expectEqual(@as(u32, 0), try Readings.chunk(&g, &.{70})); // a prompt's first chunk: no rise
    try std.testing.expectEqual(@as(u32, 0), try Readings.chunk(&g, &.{80})); // chunk 1 began after the gap: not kept
    try std.testing.expectEqual(@as(u32, 2), try Readings.chunk(&g, &.{ 90, 86, 83 })); // 90 + 10 > 93: wait to 83
    try std.testing.expectEqual(@as(u32, 2), try Readings.chunk(&g, &.{ 91, 84, 83 })); // its own 8 C, but 10 C counts
    try std.testing.expectEqual(@as(f64, 8), g.waited_s);
    g.startPrompt();
    try std.testing.expectEqual(@as(u32, 0), try Readings.chunk(&g, &.{ 90, 70 })); // a new prompt's first takes none
    var bands = Gate{ .bands = .{ .high_c = 92, .low_c = 88 } }; // the same readings with the bands alone
    bands.startPrompt();
    for ([_]f64{ 70, 80, 90 }) |t| try std.testing.expectEqual(@as(u32, 0), try Readings.chunk(&bands, &.{t}));
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: i32) i32;
extern "c" fn unsetenv(name: [*:0]const u8) i32;

fn putEnv(name: [*:0]const u8, value: ?[*:0]const u8) void {
    if (value) |v| {
        _ = setenv(name, v, 1);
    } else _ = unsetenv(name);
}

test "TF_HEAT_HIGH and TF_HEAT_LOW set the bands together, and TF_HEAT_ROOT the zones' folder" {
    const keys = [_][*:0]const u8{ "TF_HEAT_HIGH", "TF_HEAT_LOW", "TF_HEAT_ROOT" };
    defer for (keys) |k| putEnv(k, null);
    for (keys) |k| putEnv(k, null);
    try std.testing.expect((try Gate.fromEnv()).bands == null);
    putEnv("TF_HEAT_HIGH", "70");
    try std.testing.expectError(error.HeatBands, Gate.fromEnv());
    putEnv("TF_HEAT_LOW", "60");
    putEnv("TF_HEAT_ROOT", "core-root");
    const both = try Gate.fromEnv();
    try std.testing.expectEqual(@as(f64, 70), both.bands.?.high_c);
    try std.testing.expectEqual(@as(f64, 60), both.bands.?.low_c);
    try std.testing.expectEqualStrings("core-root", both.root);
    putEnv("TF_HEAT_ROOT", null);
    try std.testing.expectEqualStrings(default_root, (try Gate.fromEnv()).root);
}

test "no readable zone logs once and the chunk runs" {
    const Probe = struct {
        reads: u32 = 0,
        sleeps: u32 = 0,
        fn read(ptr: *anyopaque) !f64 {
            const p: *@This() = @ptrCast(@alignCast(ptr));
            p.reads += 1;
            return error.NoZone;
        }
        fn sleep(ptr: *anyopaque) void {
            @as(*@This(), @ptrCast(@alignCast(ptr))).sleeps += 1;
        }
    };
    no_zone_notes = 0;
    var probe = Probe{};
    var gate = Gate{ .bands = .{ .high_c = 92, .low_c = 88 }, .root = "missing-thermal" };
    try gate.waitUntil(Probe.read, Probe.sleep, &probe, null);
    try gate.waitUntil(Probe.read, Probe.sleep, &probe, null);
    try std.testing.expect(gate.no_zone);
    try std.testing.expectEqual(@as(u32, 1), probe.reads);
    try std.testing.expectEqual(@as(u32, 0), probe.sleeps);
    try std.testing.expectEqual(@as(f64, 0), gate.waited_s);
    try std.testing.expectEqual(@as(u32, 1), no_zone_notes);

    const io = std.testing.io;
    const notes = no_zone_notes;
    var missing = Gate{ .bands = .{ .high_c = 92, .low_c = 88 }, .root = "no/such/thermal/root" };
    try missing.beforePromptChunk(io, null);
    try std.testing.expectEqual(notes + 1, no_zone_notes);
    try missing.beforePromptChunk(io, null);
    try std.testing.expectEqual(notes + 1, no_zone_notes);
    try std.testing.expect(missing.no_zone);
    try std.testing.expectEqual(@as(f64, 0), missing.waited_s);

    var empty = std.testing.tmpDir(.{ .iterate = true });
    defer empty.cleanup();
    try std.testing.expectError(error.NoZone, hottest(io, empty.dir));
}

test "hottest zone is the maximum millidegree reading" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "thermal_zone0");
    try tmp.dir.createDirPath(io, "thermal_zone1");
    try tmp.dir.createDirPath(io, "other");
    try tmp.dir.writeFile(io, .{ .sub_path = "thermal_zone0/temp", .data = "80000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "thermal_zone1/temp", .data = "94000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "other/temp", .data = "99000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "thermal_zone2", .data = "not-a-zone" });
    try std.testing.expectEqual(@as(f64, 94), try hottest(io, tmp.dir));
}
