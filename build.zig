const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The codec-agnostic container, pulled from the krvs dependency (path dep for now).
    const krvs_dep = b.dependency("krvs", .{ .target = target, .optimize = optimize });
    const krvs_mod = krvs_dep.module("krvs");

    // Library module: the krvsh codec. Imports the container as "krvs".
    const mod = b.addModule("krvsh", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "krvs", .module = krvs_mod },
        },
    });

    // Static library artifact (C ABI surface comes later; for now a Zig lib).
    const lib = b.addLibrary(.{
        .name = "krvsh",
        .root_module = mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // C-ABI static library: exposes the `export fn`s in src/c_api.zig + install header.
    const capi = b.addLibrary(.{
        .name = "krvsh_c",
        .root_module = mod,
        .linkage = .static,
    });
    capi.installHeadersDirectory(b.path("include"), "", .{});
    b.installArtifact(capi);

    // CLI executable: `krvsh encode` / `krvsh decode`.
    const exe = b.addExecutable(.{
        .name = "krvsh",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "krvsh", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run the krvsh CLI");
    run_step.dependOn(&run_cmd.step);

    // End-to-end fidelity tool (tools/e2e.zig), linked against the lib module.
    const e2e = b.addExecutable(.{
        .name = "e2e",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/e2e.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "krvsh", .module = mod }},
        }),
    });
    b.installArtifact(e2e);

    // Throughput profiler (tools/profile.zig).
    const prof = b.addExecutable(.{
        .name = "profile",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/profile.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "krvsh", .module = mod }},
        }),
    });
    b.installArtifact(prof);

    // Tests: run all `test` blocks in the library module.
    const lib_tests = b.addTest(.{ .root_module = mod });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // Golden-vector tests (test/golden.zig), linked against the lib module.
    const golden_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/golden.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "krvsh", .module = mod }},
        }),
    });
    const run_golden_tests = b.addRunArtifact(golden_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_golden_tests.step);

    // C smoke test: compile test/capi_smoke.c, link the C-ABI lib, run it.
    const csmoke_mod = b.createModule(.{ .target = target, .optimize = optimize });
    csmoke_mod.addCSourceFile(.{ .file = b.path("test/capi_smoke.c"), .flags = &.{"-std=c11"} });
    csmoke_mod.addIncludePath(b.path("include"));
    csmoke_mod.linkLibrary(capi);
    csmoke_mod.link_libc = true;
    const csmoke = b.addExecutable(.{ .name = "capi_smoke", .root_module = csmoke_mod });
    const run_csmoke = b.addRunArtifact(csmoke);
    test_step.dependOn(&run_csmoke.step);
}
