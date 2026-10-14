# krvsh codec specification

Version: 1. Codec 4CC: `'krvs'`. Status: v0.

This defines what the **krvsh codec** places inside the codec-opaque boxes of the
[krvs container](container-spec.md): the `'cpay'` payload of a `'cnit'` (codec-init)
box, and the `'fpay'` payload of a `'kfrm'` (frame) box. The container framing (boxes,
init/frame/fragment, init_id dependency) is defined there and not repeated here.

Rationale and the measured basis for the compression ratios are in [design.md](design.md).

## 1. What krvsh compresses

A transformer KV cache: per layer, per {k, v}, a tensor `X` of shape `(T, D)` (T
tokens, D model dim). The krvsh init carries a per-tensor PCA basis fit from
calibration data; each frame carries the quantized, entropy-coded projection of that
frame's tensors onto the basis.

Internal byte order for krvsh payloads: **little-endian**. `fp16` = IEEE 754 half.

## 2. Init payload (`'cpay'` inside `'cnit'`)

The krvsh init payload is itself a short header + metadata + per-tensor basis blocks.

### 2.1 Init fixed fields

| offset | size | field             | notes |
|--------|------|-------------------|-------|
| 0      | 32   | model_fingerprint | sha256 over the declared identity keys (§3.1) |
| 32     | 2    | codec_id          | u16, 1 = pca-uniform-quant |
| 34     | 1    | dtype             | u8, source/output KV dtype (§2.1.1) |
| 35     | 1    | exponent_q8       | u8, per-component step exponent beta * 32; 0 => flat |
| 36     | 2    | n_layers          | u16 |
| 38     | 1    | tensors_per_layer | u8, = 2 (k, v) |
| 39     | 1    | reserved          | u8, = 0 |
| 40     | 4    | D                 | u32, model dim (e.g. 3072) |
| 44     | 4    | R                 | u32, retained rank, R <= D |

(The `init_id` is NOT here; it lives in the container's `'inid'` box and is computed
per §4.)

#### 2.1.1 dtype — source/output KV representation

`dtype` is the representation the KV tensors have in the model, at both the encoder's
input and the decoder's output. The codec's INTERNAL math (PCA projection, quant,
reconstruction) is always f32 regardless of dtype: f32 preserves precision for the
linear algebra and is what makes the wire bytes cross-architecture deterministic.
The encoder up-converts source→f32 on ingest; the decoder reconstructs f32 then
DOWN-CONVERTS to `dtype` on egress, so the model resumes with exactly the
representation it expects. Reported fidelity is measured in `dtype`, including the
final down-conversion.

| code | dtype      | bytes | notes |
|------|------------|-------|-------|
| 0    | bf16       | 2     | top 16 bits of f32; round-to-nearest-even on encode |
| 1    | fp16       | 2     | IEEE 754 half |
| 2    | fp32       | 4     | IEEE 754 single (identity) |
| 3    | fp8_e4m3   | 1     | 1-4-3 sign/exp/mantissa, bias 7, no inf (NaN=S.1111.111) |
| 4    | fp8_e5m2   | 1     | 1-5-2, bias 15, IEEE-like inf/NaN |
| 5    | int8       | 1     | RESERVED, not implemented in v0 (see note) |

All multi-byte dtypes are little-endian. Conversions are defined bit-exactly (manual
for bf16/fp8 since Zig lacks native bf16/fp8) and locked by golden vectors, so the
down-conversion is identical on any machine.

int8 (code 5) is RESERVED and NOT implemented: an int8 KV cache is a quantized cache
whose real values are `int8 * scale` with a dequant scale that lives OUTSIDE the raw
int8 bytes (per-tensor or per-channel, model-defined). Supporting it is a different
ingest contract (the scale must be supplied), not a pure dtype conversion, so it is
deferred rather than silently treated as a signed-integer-to-float cast.

A decoder MUST reject a `dtype` code it does not implement (do not guess a width).

### 2.2 Metadata section

Immediately after the fixed fields:

| size     | field    | notes |
|----------|----------|-------|
| 4        | meta_len | u32, length of the metadata blob |
| meta_len | metadata | UTF-8 JSON object (§3) |

### 2.3 Per-tensor basis blocks

Then `n_layers * tensors_per_layer` blocks, in order layer0-k, layer0-v, layer1-k, ...

| size      | field | notes |
|-----------|-------|-------|
| D * 2     | mean  | fp16[D], column means of the (T, D) calibration tensor |
| R * D * 2 | basis | fp16[R][D], rows are `Vt[:R]` (right singular vectors) |
| R * 2     | sigma | fp16[R], singular values. PRESENT ONLY IF `exponent_q8 != 0` |

`basis` is row-major `Vt[:R]` of shape (R, D); the encoder projects with `basis.T`
(D, R). `sigma` is present only when the per-component quant table is enabled.

## 3. Model metadata (JSON)

Free-form, flexible-length, extensible without a codec-version bump. Decoders treat it
as informational except for the fingerprint check (§3.1). Recommended keys (optional in
the blob; encoders SHOULD populate the first four):

| key              | type   | meaning |
|------------------|--------|---------|
| `model_id`       | string | model / checkpoint name, e.g. `lerobot/fastwam_libero_uncond_2cam224` |
| `revision`       | string | checkpoint revision / commit / weights hash fit against |
| `capture_point`  | string | where KV was hooked, e.g. `mot.video.self_attn` |
| `token_layout`   | string | how T is produced, e.g. `image_size=(224,448),cams=2,tokens=98` |
| `n_calib_frames` | number | observations the basis was fit on |
| `source_dtype`   | string | human form of `dtype`, e.g. `bf16` |
| `krvsh_writer`   | string | tool + version that wrote the init |
| `notes`          | string | free-form provenance |
| `identity_keys`  | array  | ordered key names that define model identity (§3.1) |

### 3.1 model_fingerprint and flexible identity

`model_fingerprint` (init fixed field) is the binary, enforced model identity. The set
of keys that constitutes identity is declared by `identity_keys`:

```
model_fingerprint = sha256( for each name in identity_keys, in order:
                              utf8(name) || 0x00 || utf8(str(metadata[name])) || 0x00 )
```

A missing key contributes its name and an empty value. If `identity_keys` is absent, it
defaults to `["model_id", "revision"]`. A model family can thus define "same model"
(add a quantization tag, embodiment id, capture point, etc.) without a format change.

Before decoding frames, a stage-B caller SHOULD compute the fingerprint of the model it
is about to run (same `identity_keys` composition) and compare it to the init's
`model_fingerprint`. A mismatch means the basis was calibrated for a different
model/checkpoint and would decode to silently wrong tensors; the caller MUST refuse.
D and n_layers alone do not identify a model, which is why identity is a fixed 32-byte
field checkable in constant time, not buried only in JSON.

## 4. init_id content hash

Per container §5.1, `init_id` is codec-defined. For krvsh:

```
init_id = first 16 bytes of sha256( init fixed fields (§2.1)
                                 || metadata section (meta_len + metadata)
                                 || all per-tensor basis blocks in order )
```

i.e. the entire init payload as laid out in §2. It excludes any optional container
boxes (`'crc '`, signatures), so adding those later does not change the id.

## 5. Frame payload (`'fpay'` inside `'kfrm'`)

For each of the init's `n_layers * tensors_per_layer` tensors, in the same order:

| size       | field      | notes |
|------------|------------|-------|
| 2          | scale_q    | fp16, the quant step Delta = s * rms for this tensor |
| 4          | n_symbols  | u32, = T * R (T from the frame, R from the init) |
| 1          | coder_id   | u8, 1 = range coder (§7) |
| 4          | stream_len | u32, coded stream length in bytes |
| stream_len | stream     | entropy-coded zig-zag varint of q = round(coef / Delta) |

T comes from the container frame header; R, n_layers, tensors_per_layer come from the
referenced init. The frame payload does not repeat them.

## 6. Codec math

Per layer, per {k, v} tensor `X` of shape (T, D):

Calibration (once, offline; produces the init):
- `mean = X_cal.mean(axis=0)`                       # (D,)
- `U, S, Vt = svd(X_cal - mean, full_matrices=False)`
- `basis = Vt[:R].T`  (stored as `Vt[:R]`)          # (D, R)
- `sigma = S[:R]`                                    # for the optional per-comp table

Encode (per frame):
- `coef = (X - mean) @ basis`                        # (T, R)
- `rms  = sqrt(mean(coef**2))`                       # scalar
- step: v0 default `beta = 0` => flat `Delta = s*rms`. `beta > 0` enables the
  jpeg-style per-component table `Delta_i = s*rms*(sigma_i/mean(sigma))^(-beta)`
  (research: `beta ~ 0.5` gives ~10-20% improvement). Exact formula pinned by code +
  golden vectors.
- `q = round(coef / Delta)`                          # (T, R) signed integers
- entropy-code q (§7); store `scale_q = Delta` per tensor.

Decode (per frame):
- `coef_hat = q * Delta`
- `X_hat = coef_hat @ basis.T + mean`

Determinism: round-half-to-even; fp16 basis/mean fixed; dequant math is plain fp32,
identical on any IEEE-754 machine. Golden vectors lock this.

## 7. Entropy coding (coder_id = 1)

Quantized coefficients `q` (signed) are coded as:

1. Zig-zag to unsigned: `u = (q << 1) ^ (q >> 31)`.
2. LEB128 varint per `u` (7 bits/byte, high bit = continuation).
3. Range-code the varint byte stream with an adaptive order-0 model over the 256 byte
   values.

Decode reverses: range-decode -> varint bytes -> `u` -> `q = (u >> 1) ^ -(u & 1)`.

Range-coder precision, renormalization, and model-update rule are fixed by the
reference implementation and locked by golden vectors; conforming encoders/decoders
MUST produce byte-identical output for the same input. krvsh reports **real coded
bytes** (sum of `stream_len`), and measures them against the order-0 entropy estimate
`H0(q)` used in prior research to quantify the estimate-vs-actual gap.
