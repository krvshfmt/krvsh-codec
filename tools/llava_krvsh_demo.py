"""LLaVA disaggregation demo (all on mosk): prefill -> capture KV -> krvsh roundtrip ->
inject reconstructed KV -> resume generation. Compare to the uncompressed baseline.

Proves the thesis: the KV cache between prefill and decode can be compressed ~100s x and
generation still produces the right answer. The resume mechanism is the probe-v2 path
(manual decode from the DynamicCache with cache_position); krvsh is inserted on the
cache between prefill and that loop.

Uses the krvsh CLI (calibrate.py + krvsh.exe encode/decode) via files. One process, so
bf16<->f32 happens here in Python; the codec works in f32.
"""
import os, sys, subprocess, struct, time
import numpy as np
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

MODEL = "llava-hf/llava-1.5-7b-hf"
DEV = "cuda"
# This script lives in <repo>/krvsh/tools/. Resolve the codec root from its own path so
# it works wherever the repo is checked out.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEC_ROOT = os.path.dirname(_HERE)  # <repo>/krvsh
KRVSH = os.path.join(_CODEC_ROOT, "zig-out", "bin", "krvsh.exe")
CALIB = os.path.join(_HERE, "calibrate.py")
PY = sys.executable
WORK = os.path.expanduser(r"~\krvsh_work")
os.makedirs(WORK, exist_ok=True)
RANK = 512
MAX_NEW = 40

def log(*a):
    # Windows console is cp1252; degraded model output can contain chars it can't encode.
    s = " ".join(str(x) for x in a)
    sys.stdout.buffer.write(s.encode("utf-8", "replace") + b"\n")
    sys.stdout.flush()

# ---- image: real image by default (random noise is not a fair test -- it has no
# spatial structure so its KV is artificially easy to compress, and the answer is a
# hallucination so "matches baseline" proves little). Pass a path as argv[1]. ----
img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(r"~\img1.jpg")
if os.path.exists(img_path):
    img = Image.open(img_path).convert("RGB")
    log(f"image: {img_path} ({img.size[0]}x{img.size[1]})")
else:
    log(f"WARNING: {img_path} not found, falling back to random noise (not a fair test)")
    np.random.seed(0)
    img = Image.fromarray((np.random.rand(336, 336, 3) * 255).astype("uint8"))

# ---- load ----
log("loading LLaVA...")
proc = AutoProcessor.from_pretrained(MODEL)
model = LlavaForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
model.eval()

prompt = "USER: <image>\nDescribe this image briefly. ASSISTANT:"
inputs = proc(images=img, text=prompt, return_tensors="pt").to(DEV, torch.bfloat16)
inputs["input_ids"] = inputs["input_ids"].long()

def greedy_from_cache(cache, prefill_logits, past_len, n=MAX_NEW):
    """probe-v2 resume: decode greedily from a populated cache."""
    next_tok = prefill_logits[:, -1, :].argmax(-1, keepdim=True)
    toks = [next_tok]
    for step in range(n - 1):
        pos = torch.tensor([past_len + step], device=DEV)
        with torch.no_grad():
            out = model(input_ids=next_tok, past_key_values=cache, use_cache=True, cache_position=pos)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks.append(next_tok)
    return torch.cat(toks, dim=1)

# ---- baseline: prefill then resume with the UNCOMPRESSED cache ----
with torch.no_grad():
    pre = model(**inputs, use_cache=True)
cache = pre.past_key_values
n_layers = len(cache.layers)
B, H, T, Dh = cache.layers[0].keys.shape
D = H * Dh
log(f"prefill: {n_layers} layers, k/v shape (B={B},H={H},T={T},Dh={Dh}), D={D}")
raw_kv_bytes = n_layers * 2 * T * D * 2  # bf16

# Fold the PREFILL cache to (T,D) tensors BEFORE any decode (decode grows the cache
# in place, which would change T). Capture now.
def fold(x):  # (1,H,T,Dh) -> (T, H*Dh) f32, dims from the tensor
    _, h, t, dh = x.shape
    return x[0].permute(1, 0, 2).reshape(t, h * dh).to(torch.float32).cpu().numpy()

tensors = []
for i in range(n_layers):
    tensors.append(fold(cache.layers[i].keys))
    tensors.append(fold(cache.layers[i].values))

# now the baseline decode (mutates `cache`, but we've already folded it)
base_ids = greedy_from_cache(cache, pre.logits, T)
ans_base = proc.decode(base_ids[0], skip_special_tokens=True)
log("=== BASELINE (uncompressed KV) ===")
log(ans_base)

# write npz (for calibrate) + raw f32 dump (for encode)
npz = {}
for i in range(n_layers):
    npz[f"layer{i}_k"] = tensors[2*i]
    npz[f"layer{i}_v"] = tensors[2*i+1]
np.savez(os.path.join(WORK, "kv.npz"), **npz)

f32_path = os.path.join(WORK, "kv.f32")
with open(f32_path, "wb") as f:
    f.write(struct.pack("<III", len(tensors), T, D))
    for t in tensors:
        f.write(t.astype("<f4").tobytes())

# ---- calibrate once (fit basis on this prefill; n=1 demo, overfit caveat below) ----
krvi = os.path.join(WORK, "llava.krvi")
subprocess.run([PY, CALIB, os.path.join(WORK, "kv.npz"), krvi,
                "--rank", str(RANK), "--model-id", MODEL, "--revision", "demo"],
               check=True)

def unfold(mat):  # (T, H*Dh) f32 -> (1,H,T,Dh) bf16
    tt = mat.shape[0]
    t = torch.from_numpy(np.array(mat, dtype=np.float32).reshape(tt, H, Dh)).permute(1, 0, 2).unsqueeze(0)
    return t.to(torch.bfloat16).to(DEV).contiguous()

def load_f32(path):
    with open(path, "rb") as f:
        n, tt, dd = struct.unpack("<III", f.read(12))
        arr = np.frombuffer(f.read(), dtype="<f4").reshape(n, tt, dd)
    return arr

# ---- sweep s: encode -> decode -> rebuild cache -> resume -> compare ----
log("\n s      coded_KB   ratio     answer")
for s in [0.5, 1.0, 2.0, 4.0]:
    krvf = os.path.join(WORK, f"kv_s{s}.krvf")
    out_f32 = os.path.join(WORK, f"recon_s{s}.f32")
    subprocess.run([KRVSH, "encode", krvi, f32_path, krvf, str(s)], check=True,
                   stdout=subprocess.DEVNULL)
    coded = os.path.getsize(krvf)
    subprocess.run([KRVSH, "decode", krvf, krvi, out_f32], check=True,
                   stdout=subprocess.DEVNULL)
    recon = load_f32(out_f32)

    # rebuild a fresh cache via a prefill, overwrite with reconstructed tensors
    with torch.no_grad():
        pre_s = model(**inputs, use_cache=True)
    cache_s = pre_s.past_key_values
    for i in range(n_layers):
        cache_s.layers[i].keys = unfold(recon[2*i])
        cache_s.layers[i].values = unfold(recon[2*i+1])
    ids = greedy_from_cache(cache_s, pre_s.logits, T)
    ans = proc.decode(ids[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
    ratio = raw_kv_bytes / coded
    log(f" {s:<5} {coded/1024:8.1f}  {ratio:7.1f}x   {ans[:70]}")

log(f"\nraw KV = {raw_kv_bytes/1e6:.1f} MB (bf16), {n_layers} layers x 2 x ({T},{D})")
log("baseline answer for reference:")
log("  " + ans_base.split("ASSISTANT:")[-1].strip()[:70])
log("\nCAVEAT: n=1 demo, basis fit on the SAME prefill it compresses (overfit-optimistic).")
log("Research multi-image LLaVA number was ~400x @ 0.94 semantic similarity.")
