//! krvsh — KV-cache codec for disaggregated model stages.
//!
//! The krvsh codec rides in the codec-agnostic `krvs` container (separate repo,
//! imported as the `krvs` module). This library is the codec: PCA basis + model
//! metadata in the init payload, quantized entropy-coded coefficients per frame, and
//! the glue that wraps those payloads in container records. See docs/krvsh-codec-spec.md
//! and docs/design.md.

const std = @import("std");

/// The codec-agnostic container, re-exported for consumers that want direct access.
pub const krvs = @import("krvs");

/// dtype conversions at the encode/decode boundary (bf16/fp16/fp32/fp8).
pub const dtype = @import("dtype.zig");

/// Range coder (adaptive order-0 byte model) — the krvsh entropy backend.
pub const rangecoder = @import("rangecoder.zig");

/// Quantization + zig-zag/varint symbol serialization.
pub const quant = @import("quant.zig");

/// krvsh codec-init (`cnit`) payload build/parse + init_id + model fingerprint.
pub const krvsh_init = @import("krvsh_init.zig");

/// krvsh frame (`kfrm`) per-tensor encode/decode.
pub const krvsh_frame = @import("krvsh_frame.zig");

/// krvsh<->container glue: wraps codec payloads in container records (cnit/kfrm/marker)
/// with the krvsh codec id + init-id hash + verify-on-ingest resolver.
pub const krvsh_box = @import("krvsh_glue.zig");

/// C ABI exports (see include/krvsh.h).
pub const c_api = @import("c_api.zig");

// Keep the C ABI exports in the library artifact (not dead-stripped).
comptime {
    _ = c_api;
}

/// Container major version (the `krvs` marker box version).
pub const CONTAINER_VERSION: u8 = krvs.CONTAINER_VERSION;

test {
    std.testing.refAllDecls(@This());
    _ = krvs;
    _ = dtype;
    _ = rangecoder;
    _ = quant;
    _ = krvsh_init;
    _ = krvsh_frame;
    _ = krvsh_box;
    _ = c_api;
}
