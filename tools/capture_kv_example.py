#!/usr/bin/env python3
"""REFERENCE EXAMPLE: capture a KV cache into the .npz that calibrate.py consumes.

This is the ONE model-side step. Unlike the rest of krvsh (numpy + Zig, no model), this
needs the model and its framework (here: torch + transformers). It is NOT part of the
codec, NOT imported by anything, and NOT run by `zig build test`. It exists to show the
shape of the data the codec expects and how a capture typically produces it.

The hook points are MODEL-SPECIFIC. This script shows the common HuggingFace-transformers
pattern (grab `past_key_values` after a prefill). For other stacks the mechanism differs
-- e.g. the Fast-WAM capture we validated hooked `mot.prefill_video_cache` to collect a
list of 30 per-layer {k, v} tensors of shape (1, 98, 3072). The OUTPUT contract is what
matters and is the same everywhere:

  an .npz with arrays named layer0_k, layer0_v, layer1_k, layer1_v, ...
  each of shape (T, D)  (tokens x model_dim), float.

calibrate.py sorts these layer0_k, layer0_v, layer1_k, ... to match the init's
layer/{k,v} ordering, so the names must follow that scheme.

Usage (example):
  pip install torch transformers numpy
  python3 capture_kv_example.py --model Qwen/Qwen3-8B --context "some long text" --out kv.npz
  # then:  python3 calibrate.py kv.npz model.krvi --model-id Qwen/Qwen3-8B --revision <sha>
"""

import argparse
import numpy as np


def capture_hf(model_id: str, context: str, out_path: str, max_tokens: int | None):
    """Capture KV from a HuggingFace causal-LM prefill via past_key_values."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.eval()

    ids = tok(context, return_tensors="pt").input_ids
    if max_tokens:
        ids = ids[:, :max_tokens]

    with torch.no_grad():
        out = model(ids, use_cache=True)

    # past_key_values: tuple over layers of (key, value); each (batch, heads, T, head_dim).
    # krvsh treats each layer's k and v as a (T, D) matrix, D = heads*head_dim. We fold the
    # head and head_dim axes into D and drop the batch dim. (Folding heads into one D is a
    # representational choice -- it's what the per-layer basis expects here; a per-head
    # basis is possible but is a codec-config change, not shown.)
    arrays = {}
    for layer_idx, (k, v) in enumerate(out.past_key_values):
        # (B, H, T, Dh) -> (T, H*Dh), B assumed 1
        kk = k[0].permute(1, 0, 2).reshape(k.shape[2], -1).to(torch.float32).cpu().numpy()
        vv = v[0].permute(1, 0, 2).reshape(v.shape[2], -1).to(torch.float32).cpu().numpy()
        arrays[f"layer{layer_idx}_k"] = kk
        arrays[f"layer{layer_idx}_v"] = vv

    np.savez(out_path, **arrays)
    n = len(out.past_key_values)
    T, D = arrays["layer0_k"].shape
    print(f"wrote {out_path}: {n} layers, k+v each ({T},{D}). "
          f"Feed to calibrate.py (it sorts layer0_k, layer0_v, ...).")


def main():
    ap = argparse.ArgumentParser(description="Reference KV-capture example (model-side).")
    ap.add_argument("--model", required=True, help="HuggingFace model id")
    ap.add_argument("--context", default="The quick brown fox jumps over the lazy dog. " * 32)
    ap.add_argument("--out", default="kv.npz")
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()
    capture_hf(args.model, args.context, args.out, args.max_tokens)


if __name__ == "__main__":
    main()
