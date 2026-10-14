#!/usr/bin/env python3
"""krvsh calibration tool.

Fit a per-tensor PCA basis from calibration KV tensors and write a `.krvi` file: a
`krvs` marker box + a `cnit` codec-init box whose `cpay` is the krvsh init payload
(krvsh-codec-spec.md section 2). This is the one-time, offline step; the Zig codec owns
the per-frame hot path and reads this file back.

Byte layout MUST match the Zig side exactly:
  - box framing (box.zig): big-endian size/type, then version u8 + flags u24 big-endian.
  - init payload (krvsh_init.zig): little-endian scalars, fp16 little-endian tensors.

Usage:
  calibrate.py INPUT.npz OUTPUT.krvi [--rank R] [--model-id ID] [--revision REV]

INPUT.npz: arrays named "<layerN>_k" / "<layerN>_v" of shape (T, D), as produced by the
KV capture. Tensors are sorted layer0_k, layer0_v, layer1_k, layer1_v, ... to match the
init's layer/{k,v} ordering.
"""

import argparse
import hashlib
import json
import re
import struct
import sys

import numpy as np

# ---- constants mirrored from the Zig side ----
MAGIC_MARKER = b"krvs"
T_CNIT = b"cnit"
T_CDCI = b"cdci"
T_INID = b"inid"
T_CPAY = b"cpay"
FLAG_MANDATORY = 0x000001
CONTAINER_MAJOR = 1
CONTAINER_MINOR = 0
CODEC_FOURCC = b"krvs"
CODEC_VERSION = 1
HASH_ALGO_SHA256_T16 = 1
INIT_ID_LEN = 16
CODEC_PCA_UNIFORM_QUANT = 1
DTYPE_BF16, DTYPE_FP16, DTYPE_FP32 = 0, 1, 2


def box(fourcc: bytes, version: int, flags: int, payload: bytes) -> bytes:
    """Compact box header (12 bytes) + payload. Big-endian framing, matching box.zig."""
    assert len(fourcc) == 4
    total = 12 + len(payload)
    assert total <= 0xFFFFFFFF, "largesize not needed for calibration output"
    hdr = struct.pack(">I", total) + fourcc + struct.pack(">I", (version << 24) | (flags & 0xFFFFFF))
    return hdr + payload


def fp16_le(arr: np.ndarray) -> bytes:
    """Encode a float array as little-endian fp16 bytes."""
    return arr.astype("<f2").tobytes()


def model_fingerprint(identity_pairs) -> bytes:
    h = hashlib.sha256()
    for name, value in identity_pairs:
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(value).encode("utf-8"))
        h.update(b"\x00")
    return h.digest()  # 32 bytes


def sorted_tensor_keys(keys):
    """layer0_k, layer0_v, layer1_k, ... -> ordered by (layer index, k before v)."""
    def rank(k):
        m = re.match(r"layer(\d+)_([kv])", k)
        if not m:
            raise ValueError(f"unexpected array name {k!r}; want layerN_k / layerN_v")
        return (int(m.group(1)), 0 if m.group(2) == "k" else 1)
    return sorted(keys, key=rank)


def build_cpay(tensors, rank, metadata: dict):
    """Build the krvsh init payload (cpay). tensors: list of (T,D) float32 arrays in
    layer/{k,v} order. Returns (cpay_bytes, D, R, n_tensors)."""
    D = tensors[0].shape[1]
    for t in tensors:
        if t.shape[1] != D:
            raise ValueError("all tensors must share D")
    R = min(rank, D)
    n = len(tensors)
    if n % 2 != 0:
        raise ValueError("expected an even tensor count (k,v per layer)")
    n_layers = n // 2

    identity_keys = metadata.get("identity_keys", ["model_id", "revision"])
    pairs = [(k, metadata.get(k, "")) for k in identity_keys]
    fp = model_fingerprint(pairs)

    meta_json = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")

    # fixed header (48 bytes), little-endian
    hdr = bytearray()
    hdr += fp                                   # model_fingerprint[32]
    hdr += struct.pack("<H", CODEC_PCA_UNIFORM_QUANT)  # codec_id
    hdr += struct.pack("<B", DTYPE_BF16)        # dtype (source KV is bf16)
    hdr += struct.pack("<B", 0)                 # exponent_q8 = 0 (flat)
    hdr += struct.pack("<H", n_layers)          # n_layers
    hdr += struct.pack("<B", 2)                 # tensors_per_layer
    hdr += struct.pack("<B", 0)                 # reserved
    hdr += struct.pack("<I", D)                 # D
    hdr += struct.pack("<I", R)                 # R
    assert len(hdr) == 48, len(hdr)

    out = bytearray(hdr)
    out += struct.pack("<I", len(meta_json))
    out += meta_json

    for t in tensors:
        x = t.astype(np.float64)
        mean = x.mean(axis=0)                   # (D,)
        xc = x - mean
        # economy SVD; Vt is (min(T,D), D). basis rows = Vt[:R].
        _, _, Vt = np.linalg.svd(xc, full_matrices=False)
        if Vt.shape[0] < R:
            # pad with zeros if rank-deficient (T < R)
            pad = np.zeros((R - Vt.shape[0], D))
            Vt = np.vstack([Vt, pad])
        basis = Vt[:R]                          # (R, D), row-major Vt[:R]
        out += fp16_le(mean.astype(np.float32)) # mean fp16[D]
        out += fp16_le(basis.astype(np.float32))# basis fp16[R*D]
        # no sigma (flat quant, exponent_q8 == 0)

    return bytes(out), D, R, n


def init_id(cpay: bytes) -> bytes:
    return hashlib.sha256(cpay).digest()[:INIT_ID_LEN]


def build_krvi(cpay: bytes) -> bytes:
    # marker
    marker_payload = struct.pack("<I", CONTAINER_MINOR) + struct.pack("<I", 0)
    marker = box(MAGIC_MARKER, CONTAINER_MAJOR, FLAG_MANDATORY, marker_payload)
    # cnit children
    cdci = box(T_CDCI, 1, FLAG_MANDATORY, CODEC_FOURCC + struct.pack("<I", CODEC_VERSION))
    iid = init_id(cpay)
    inid = box(T_INID, 1, FLAG_MANDATORY, bytes([HASH_ALGO_SHA256_T16, INIT_ID_LEN]) + iid)
    cpay_box = box(T_CPAY, 1, FLAG_MANDATORY, cpay)
    cnit = box(T_CNIT, 1, FLAG_MANDATORY, cdci + inid + cpay_box)
    return marker + cnit


def main():
    ap = argparse.ArgumentParser(description="krvsh calibration: npz -> .krvi")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--model-id", default="")
    ap.add_argument("--revision", default="")
    ap.add_argument("--capture-point", default="")
    ap.add_argument("--token-layout", default="")
    ap.add_argument("--dump-tensors", default="",
                    help="also write concatenated raw f32-LE tensors (T,D per tensor) for testing")
    args = ap.parse_args()

    data = np.load(args.input)
    keys = sorted_tensor_keys(list(data.files))
    tensors = [np.asarray(data[k], dtype=np.float32) for k in keys]

    metadata = {
        "model_id": args.model_id,
        "revision": args.revision,
        "capture_point": args.capture_point,
        "token_layout": args.token_layout,
        "n_calib_frames": 1,
        "source_dtype": "bf16",
        "krvsh_writer": "calibrate.py 0.0.0",
        "identity_keys": ["model_id", "revision"],
    }

    cpay, D, R, n = build_cpay(tensors, args.rank, metadata)
    krvi = build_krvi(cpay)
    with open(args.output, "wb") as f:
        f.write(krvi)

    iid = init_id(cpay).hex()
    print(f"wrote {args.output}: {len(keys)} tensors, D={D} R={R}, "
          f"cpay={len(cpay)} bytes, file={len(krvi)} bytes, init_id={iid}")

    if args.dump_tensors:
        T = tensors[0].shape[0]
        with open(args.dump_tensors, "wb") as f:
            # header: n_tensors u32, T u32, D u32 (little-endian), then raw f32-LE data
            f.write(struct.pack("<III", len(tensors), T, D))
            for t in tensors:
                f.write(t.astype("<f4").tobytes())
        print(f"dumped tensors to {args.dump_tensors}: n={len(tensors)} T={T} D={D}")


if __name__ == "__main__":
    main()
