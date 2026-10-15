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

## Example: calibrate, then encode against the calibration

The workflow has two phases. **Calibration** is one-time and offline: fit a PCA basis
from representative KV tensors and save it as a `.krvi` file (a `cnit` codec-init
record). **Encoding** is per-observation: compress each KV tensor into a `.krvf` file
whose frames reference the calibration by its content-hash `init_id`. The two are
decoupled — calibrate once, encode thousands of frames against the saved basis, and ship
the small `.krvf` files without re-sending the large basis.

### 0. Capture the KV cache (the only model-side step)

Everything in krvsh operates on KV tensors as *data* — the codec never loads a model and
has no torch/GPU dependency (just numpy for calibration + the Zig library). The one step
that needs the model is capturing representative KV to calibrate on, and capturing live
KV to encode. That happens in your inference pipeline, not in krvsh.

The contract is an `.npz` with arrays named `layer0_k`, `layer0_v`, `layer1_k`, ... each
of shape `(T, D)` (tokens × model dim), float. `tools/capture_kv_example.py` is a
reference showing the common HuggingFace pattern (grab `past_key_values` after a
prefill); hook points are model-specific, but the output layout is fixed:

```
$ pip install torch transformers numpy          # model-side deps, NOT needed by the codec
$ python3 tools/capture_kv_example.py \
      --model Qwen/Qwen3-8B \
      --context "a long document to prefill..." \
      --out captured_kv.npz
wrote captured_kv.npz: 36 layers, k+v each (512,4096). Feed to calibrate.py ...
```

For a validated non-HF example, the Fast-WAM capture hooked the video-DiT prefill to
collect 30 per-layer `{k, v}` tensors of shape `(98, 3072)`. Any capture works as long
as it emits the `layer{N}_k`/`layer{N}_v` npz above.

### 1. Calibrate and save the calibration data

`tools/calibrate.py` (numpy) fits per-layer, per-{k,v} mean + SVD basis and writes a
`.krvi`. Input is an `.npz` with arrays named `layer0_k`, `layer0_v`, `layer1_k`, ...
each of shape `(T, D)`.

```
$ python3 tools/calibrate.py captured_kv.npz fastwam.krvi \
      --rank 512 \
      --model-id "lerobot/fastwam_libero" \
      --revision "a1b2c3d" \
      --capture-point "mot.video.self_attn" \
      --token-layout "T=98,D=3072"
wrote fastwam.krvi: 4 tensors, D=3072 R=512, cpay=12607782 bytes, file=12607876 bytes, init_id=586b2e2fa7be42b5126929a7f560b2fd
```

`fastwam.krvi` is the saved calibration. Its `init_id` (`586b2e2f...`) is a content hash
of the basis + metadata — the stable identifier that frames reference. The
`--model-id`/`--revision` populate the `model_fingerprint` so a decoder can refuse a
calibration fit for a different checkpoint. Inspect it any time:

```
$ ./zig-out/bin/krvsh info fastwam.krvi
krvs marker ok (20 bytes consumed)
cnit: init_id=586b2e2fa7be42b5126929a7f560b2fd codec_v=1 D=3072 R=512 n_layers=2 tpl=2 dtype=0 cpay=12607782B
  metadata: {"capture_point":"mot.video.self_attn","model_id":"lerobot/fastwam_libero","revision":"a1b2c3d",...}
```

### 2. Encode frames, referencing the calibration

`krvsh encode <init.krvi> <tensors> <out.krvf> [s]` compresses tensors against the saved
calibration. `s` is the rate knob (higher = smaller + lossier). The input tensor format
is a raw dump (`u32 n_tensors | u32 T | u32 D | n×T×D f32-LE`); `--dump-tensors` on
`calibrate.py` emits one, or produce it from your capture pipeline.

```
$ ./zig-out/bin/krvsh encode fastwam.krvi obs.f32 obs.krvf 2.0
encoded 4 tensors (T=98 D=3072 R=512, s=2) -> obs.krvf: 10170B frames, 10518B total
```

Every frame in `obs.krvf` carries the calibration's `init_id`, binding it to that exact
basis without embedding it:

```
$ ./zig-out/bin/krvsh info obs.krvf
krvs marker ok (20 bytes consumed)
kfrm: frame_id=0 prev=0 init_id=586b2e2fa7be42b5126929a7f560b2fd fpay=1224B
kfrm: frame_id=1 prev=0 init_id=586b2e2fa7be42b5126929a7f560b2fd fpay=3330B
...
```

### 3. Decode (resolves the calibration by init_id)

`krvsh decode <in.krvf> <init.krvi> <out>` reconstructs the tensors. The decoder resolves
each frame's `init_id` against the supplied calibration; a mismatch is a hard error, not
a silent wrong reconstruction.

```
$ ./zig-out/bin/krvsh decode obs.krvf fastwam.krvi recon.f32
decoded 4 frames (T=98 D=3072) -> recon.f32: 4816908B
```

The 12.6 MB calibration is sent/pinned once; here four frames cost ~10 KB total and
reference it by id. In a disaggregated deployment the `.krvi` is pinned at the edge
decoder (or fetched once by `init_id`), and only the small `.krvf` frames travel per
observation. See the container spec's calibration-cache section for the resolution rules.

## Tooling

    zig build -Doptimize=ReleaseFast     # optimized CLI + e2e + profile
    tools/calibrate.py in.npz out.krvi   # fit PCA basis (numpy), emit a .krvi

## Why it compresses

The codec is the one validated in prior research: per-layer, per-{k,v} PCA basis
projection + uniform scalar quantization + entropy coding. Measured tolerance on real
models: Fast-WAM policy ~800x at ~10% action error with zero gripper flips; LLaVA-7B
~400x at 0.94 semantic similarity. The basis is one-time calibration side-data carried
in a codec-init (`cnit`) record; the per-frame payload (`kfrm`) is small and references
it by content-hash `init_id`.
