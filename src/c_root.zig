//! Root for the C-ABI library artifact only. Keeps the C exports and links libc
//! (c_api uses std.heap.c_allocator); the plain Zig library (root.zig) does NOT, so
//! pure-Zig consumers stay libc-free.

const c_api = @import("c_api.zig");

comptime {
    _ = c_api;
}
