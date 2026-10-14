# krvsh design

KV-cache codec + wire format for disaggregated model stages.

Status: v0 in progress. The normative byte layouts live in
the krvs container spec (separate repo) (codec-agnostic box format) and
[krvsh-codec-spec.md](krvsh-codec-spec.md) (the krvsh codec payloads); this document
is the rationale and the decisions.

## 1. Problem

Run model stage A in location X, ship intermediate state over a network, resume
stage B in location Y. The intermediate state is a transformer KV cache (per-layer
k/v tensors). Raw it is large: Fast-WAM is ~36 MB per observation (30 layers x 2 x
(1, 98, 3072) bf16). Prior research showed it compresses ~100-800x depending on the
downstream task, with a measurable fidelity cost.

What was missing: a real wire format. The research pickled tensors and reported
*estimated* ratios from an order-0 entropy model. To make disaggregation real we need:

1. A byte-exact, versioned container that separates one-time calibration side-data
   from per-frame payload.
2. A real entropy coder, so reported bytes are actual bytes on the wire, not an
   entropy estimate.
3. Encode/decode that roundtrips within tolerance and is deterministic across
   machines: a cloud encoder and an edge decoder must agree bit-for-bit on the
   dequantized result given the same payload.

## 2. Scope (v0)

In:
- Container format spec + reference encoder/decoder.
- Codec: PCA-basis projection + uniform scalar quant (the research codec, the part
  that produced the numbers). Optional jpeg-style per-component step table.
- Real entropy coding of quantized symbols (range coder, see §6).
- A loopback demo: stage A -> serialize -> [bytes] -> deserialize -> stage B.
- Local tests on synthetic + saved real tensors (numpy/CPU). No GPU to test the format.

Out (v0, revisit later):
- Learned/trained codecs (decoder fine-tune on the compressed domain).
- Attention-guided per-token allocation. It works (3x on BLIP) but it is a codec
  refinement; the format allows per-token step metadata, but the v0 encoder does not
  compute it.
- Transport (TCP/QUIC/gRPC). We define the byte format; moving bytes is the caller's
  job. v0 ships a file/bytes API and CLI only.
- Temporal delta across frames. The format reserves a `prev_frame_id` field but v0
  codes each frame independently.

## 3. Two-layer architecture: codec-agnostic container + the krvsh codec

The format splits into two layers so the container can outlive any single codec:

- **Container** (codec-agnostic, the krvs container spec (separate repo)): an
  ISO-BMFF-style box tree. Knows only boxes (size/type/largesize/version/flags),
  nesting, mandatory-vs-skip, the `krvs` magic marker, and the init/frame/fragment
  framing with an init-dependency rule. It does not know what a PCA basis is. A future
  codec reuses it unchanged.
- **krvsh codec** (one registered codec, 4CC `krvs`,
  [krvsh-codec-spec.md](krvsh-codec-spec.md)): defines the contents of the
  codec-opaque boxes — the PCA basis + model metadata in the init, the quantized
  entropy-coded coefficients in the frame.

This mirrors ISO-BMFF/MP4: the box container knows nothing about H.264; a codec
registers a type and the container carries it.

### 3.1 Init vs frame (the amortization made explicit)

The research's biggest honesty gap: the PCA basis is ~378 MB and was excluded from the
reported ratios. In a real system that basis is one-time, amortized state. The
container generalizes this to any codec:

- **Codec-init box (`cnit`)**: one-time codec state keyed by a content-hash `init_id`.
  For krvsh this is the former "CAL": the per-layer per-{k,v} basis (mean + Vt[:R]),
  codec params, and model-identity metadata. Large, sent rarely.
- **Frame box (`kfrm`)**: references an `init_id`, carries only the codec-opaque frame
  payload (for krvsh: entropy-coded quantized coefficients + per-tensor scales). Small,
  per observation.
- **Fragment box (`kfrg`)**: groups frames sharing an init. Defined now because
  retrofitting frame grouping later would break how frames are addressed in a
  multiplexed file.

A decoder must resolve the referenced init (earlier on the stream, or pinned out-of-
band by `init_id`) before decoding a frame. The frame carries its `init_id` by
content hash, so a lone frame file is self-describing about its dependency.

### 3.2 Carriage modes

One concept (CMAF's init+fragment idea, without the `moof`/`mdat`/sample-table
machinery) covers three modes:

1. Independent frame file: `[krvs marker][kfrm]`, init resolved out-of-band.
2. Multiplexed file: `[krvs marker][cnit][kfrg[kfrm,...]]`, self-contained.
3. Streamed: the same boxes arriving over time.

### 3.3 Model identity

The init records *which model it was calibrated for*. The basis is checkpoint-specific
(different fine-tunes of one architecture have different KV statistics), and
D/n_layers alone do not identify a model, so a mismatched init could decode with the
right shapes but produce silently wrong tensors. The init carries a free-form,
flexible-length JSON metadata blob (`model_id`, `revision`, `capture_point`,
`token_layout`, provenance, and any model-specific keys) and a fixed 32-byte
`model_fingerprint` the decoder compares in constant time. The keys that define
identity are themselves declared in the metadata (`identity_keys`), so a model family
can define "same model" without a format change. See krvsh-codec-spec.md §3.

### 3.4 Init size and amortization

Full Fast-WAM (30 layers, k+v, D=3072), fp16 basis:

| R    | init size | per-frame payload (measured) | break-even |
|------|-----------|------------------------------|------------|
| 1024 | 378 MB    | ~45 KB                       | ~8,400 frames |
| 512  | 189 MB    | smaller                      | fewer |
| 256  |  95 MB    | smaller                      | fewer |

The init is almost entirely the basis: `30 x 2 x (R x D) x 2 bytes`. The `mean` (D
fp16) and the optional per-component `sigma` table (R fp16) are rounding error
(~123 KB total at R=1024). fp16 basis is a 2x save over fp32 at negligible fidelity
cost, because the quantizer, not basis precision, is the fidelity bottleneck
(measured). R is the big lever if init size matters: 1024 -> 512 halves it.

## 4. Container layout

Normative layout is split across two specs:
- the krvs container spec (separate repo): box framing + `krvs`/`cnit`/`kfrm`/`kfrg`/
  `crc ` boxes, mandatory-vs-skip, init_id dependency, carriage modes.
- [krvsh-codec-spec.md](krvsh-codec-spec.md): the bytes inside `cnit`'s `cpay`
  (basis + metadata) and `kfrm`'s `fpay` (scales + entropy-coded symbols).

The frame payload is the only thing that scales with frame count. Reported "bytes on
the wire per frame" is the sum of frame-box sizes, and it is real because the symbol
stream is actually entropy-coded.

## 5. Codec (matches the research exactly, so the numbers transfer)

Per layer, per {k,v} tensor X of shape (T, D):

Calibration (once):
- `mean = X_cal.mean(axis=0)`                       # (D,)
- `U, S, Vt = svd(X_cal - mean, full_matrices=False)`
- `basis = Vt[:R].T`                                 # (D, R)
- `sigma = S[:R]`                                    # for the optional per-comp table

Encode (per frame):
- `coef = (X - mean) @ basis`                        # (T, R)
- `rms  = sqrt(mean(coef**2))`                       # scalar
- step: v0 default `beta=0` => flat `Delta = s*rms` (the validated path). `beta>0`
  enables the jpeg-style per-component table `Delta_i = s*rms*(sigma_i/mean(sigma))^(-beta)`
  (research found `beta~0.5` gives ~10-20% polish). Exact formula pinned in code +
  golden test.
- `q = round(coef / Delta)`                          # integer symbols, (T, R)
- entropy-code q (§6). The scale `s` is stored in the frame payload.

Decode (per frame):
- `coef_hat = q * Delta`
- `X_hat = coef_hat @ basis.T + mean`

Determinism: round-half-to-even; fp16 basis/mean fixed; dequant math is plain fp32,
identical on any IEEE-754 box. Golden vectors lock this.

Note on the codec being near-optimal: prior research showed transform + scale are the
big levers; the per-component table and the entropy backend are each ~polish. v0 ships
flat quant (what the 800x numbers were measured on); per-component is behind a flag.

## 6. Entropy coder (the real-bytes piece)

The research reported `ratio = 16*N / (H0(q)*N)`, an order-0 entropy *estimate*. v0
emits real bytes.

Decision: a hand-rolled range coder with an adaptive order-0 model over zig-zag
varint bytes. Simple, deterministic, gets within a few percent of H0 in practice.
Rejected: rANS (same ratio, more fiddly to make byte-exact across impls; defer) and
off-the-shelf zstd on raw int16 (worse on this peaky distribution, and we could not
claim we hit the entropy).

The q distribution is zero-peaked and symmetric. Encode q with zig-zag
(`(q<<1) ^ (q>>31)`), then LEB128 varint, then range-code the varint bytes with an
adaptive byte model. We measure real coded bytes against the H0 estimate and report
the gap; that closes the research's honesty hole directly. Expectation: within ~2-8%
of H0.

## 7. Language / stack: Zig 0.17.0 (pinned)

Chosen for the stated constraints (hand-written asm later, CUDA interop, performance):
- asm: Zig inline `asm` and `.S`/intrinsics are on par with Rust for NEON/AVX kernels.
- CUDA: Zig's C interop is first-class (`@cImport` of cuda_runtime.h, `zig cc`, drive
  nvcc from build.zig) — thinner FFI than Rust's bindgen/cudarc stack. CUDA is not in
  v0 scope (it lives on the GPU host) but the language choice keeps that path clean.
- one toolchain for Zig/C/CUDA/asm.

Cost accepted: thinner ecosystem (we hand-roll the range coder either way) and pre-1.0
language churn, mitigated by pinning 0.17.0 in `.zigversion`. (Ported 0.16 -> 0.17:
`b.args` -> `run_cmd.addPassthruArgs()`; the `**` array-repeat on anonymous literals
was replaced with `@splat`. Golden vectors confirmed the wire bytes were unchanged by
the compiler bump.)

### 7.1 Where SVD/calibration lives

Calibration (mean + SVD basis) is one-time, offline, not on the per-frame hot path.
Zig has no free LAPACK. Decision: a small Python/numpy helper (`tools/calibrate.py`)
emits the `cnit` codec-init record; the Zig core owns everything on the per-frame wire
path (quant, entropy code, box framing) plus `cnit` parsing/decode. v1 may bring SVD
into Zig via BLAS if online recalibration is ever needed.

### 7.2 Python binding

Deferred past v0. The Zig lib will export a C ABI; model-side Python glue can call it
via ctypes when we wire the real-model integration test. The v0 deliverable is the Zig
lib + CLI (encode/decode subcommands).

## 8. Testing (TDD)

Red-first, in order:
1. Box reader/writer: parse/emit box header (size/type/largesize/version/flags),
   skip optional unknown boxes, fail on mandatory unknown. (replaces the old flat
   header; header.zig -> box.zig)
2. Range coder roundtrip on random streams; coded-len <= raw for peaky input.
3. Quant roundtrip: `q = round(coef/Delta)`, dequant within Delta/2 elementwise.
4. cnit serialize/parse: basis+mean survive fp16 roundtrip within fp16 eps;
   init_id recomputes; model_fingerprint check.
5. kfrm serialize/parse referencing a cnit by init_id; decode without the init ->
   explicit error. kfrg groups frames; independent frame file resolves init out-of-band.
6. End-to-end on a saved real Fast-WAM KV sample: compressed bytes and fidelity land
   near the research numbers. This is the regression that keeps us honest.
7. Golden vectors: fixed input -> fixed bytes, committed, so cross-version/
   cross-machine drift is caught.
8. Cross-version: a v2-tagged record is rejected by a v1 decoder with a clear error.

Fixture: a trimmed real Fast-WAM KV sample (2 layers, k+v, (98, 3072)) at
`test/fixtures/kv_fixture.npz`, pulled from the GPU host's captured `kv_sample.pt`.

## 9. Decisions log

1. Language: Zig 0.17.0 (asm + CUDA interop later; CUDA not in v0). Pinned.
2. v0 scope: no transport. Deliver encoder+decoder library and CLI.
3. cal_id hash: sha256 truncated to 16 bytes (`std.crypto.hash.sha2.Sha256`).
4. Fixture-first: real Fast-WAM KV sample before codec code.
5. v0 default quant is flat (`Delta = s*rms`); per-component table behind a flag.
