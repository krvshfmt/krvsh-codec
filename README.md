# krvsh

A KV-cache codec for disaggregated model stages.

Run model stage A in one location (e.g. cloud video-DiT prefill), ship its compressed
transformer KV cache over a network, resume stage B in another (e.g. edge action expert
/ decoder). `krvsh` is the codec: it compresses a KV tensor to a small byte payload and
reconstructs it. The bytes travel in the codec-agnostic **krvs container** (a separate
project, consumed here as a library).

Transport is out of scope: krvsh produces and consumes bytes; moving them is the
caller's job.

## Layering

- **krvs container** (dependency, separate repo): ISO-BMFF-style box format carrying
  `cnit` (codec-init) and `kfrm` (frame) records, codec-agnostic. Imported as the
  `krvs` module.
- **krvsh codec** (this repo): PCA basis + model metadata in the init, quantized
  entropy-coded coefficients per frame. Wraps its payloads in container records via
  `src/krvsh_glue.zig`.

## Status

v0. See [`docs/`](docs/):
- [docs/design.md](docs/design.md) — rationale, decisions, two-layer architecture.
- [docs/krvsh-codec-spec.md](docs/krvsh-codec-spec.md) — the krvsh codec payloads/dtype.

The container format spec lives in the krvs container repo.

## Build

    zig build            # lib + CLI into zig-out/
    zig build test       # unit tests
    ./zig-out/bin/krvsh version

Requires Zig 0.17.0 (pinned in `.zigversion`). The build pulls the `krvs` container via
a path dependency (`../krvs-container`) for local development; this becomes a URL + hash
dependency once the container is published.

Tooling:

    zig build -Doptimize=ReleaseFast     # optimized CLI + e2e + profile
    tools/calibrate.py in.npz out.krvi   # fit PCA basis (numpy), emit a cnit file

## Why it compresses

The codec is the one validated in prior research: per-layer, per-{k,v} PCA basis
projection + uniform scalar quantization + entropy coding. Measured tolerance on real
models: Fast-WAM policy ~800x at ~10% action error with zero gripper flips; LLaVA-7B
~400x at 0.94 semantic similarity. The basis is one-time calibration side-data carried
in a codec-init (`cnit`) record; the per-frame payload (`kfrm`) is small and references
it by content-hash `init_id`.
