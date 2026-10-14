import numpy as np, struct, hashlib, sys

def load(p):
    with open(p, "rb") as f:
        n, T, D = struct.unpack("<III", f.read(12))
        return n, T, D, np.frombuffer(f.read(), dtype="<f4").reshape(n, T, D)

orig, rec, krvf = sys.argv[1], sys.argv[2], sys.argv[3]
n, T, D, o = load(orig)
_, _, _, r = load(rec)
print(f"n={n} T={T} D={D}")
for i in range(n):
    a, b = o[i].astype(np.float64), r[i].astype(np.float64)
    rel = np.sqrt(((a - b) ** 2).sum() / (a ** 2).sum())
    cos = (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"  tensor {i}: rel_err={rel:.4f} cosine={cos:.5f}")
print("krvf_sha256:", hashlib.sha256(open(krvf, "rb").read()).hexdigest()[:16])
