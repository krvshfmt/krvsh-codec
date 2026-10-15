/* krvsh — KV-cache codec C ABI.
 *
 * Hand-written to match src/c_api.zig. A C smoke test (test/capi_smoke.c) includes this
 * header and links the static library; it runs under `zig build test`.
 *
 * These are the per-tensor codec primitives. Container framing (cnit/kfrm records) is
 * the krvs container C ABI's job (krvs.h); a C consumer uses both headers together:
 * parse a cnit with krvs.h to get the init cpay, slice its mean/basis blocks, then call
 * krvsh_encode_tensor / krvsh_decode_tensor here.
 *
 * Conventions:
 *   - Caller-owned buffers; nothing the library allocates crosses the boundary.
 *   - Encode uses a two-call size/fill protocol: out=NULL (or small cap) -> required
 *     size in *written and KRVSH_BUFFER_TOO_SMALL, then call again.
 *   - fp16 blocks are little-endian: mean is D halfs, basis is R*D halfs (row-major
 *     Vt[:R]).
 */
#ifndef KRVSH_H
#define KRVSH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KRVSH_OK = 0,
    KRVSH_BUFFER_TOO_SMALL = 1,
    KRVSH_SHORT_BUFFER = 2,
    KRVSH_UNKNOWN_CODER = 3,
    KRVSH_SYMBOL_COUNT_MISMATCH = 4,
    KRVSH_BAD_METADATA = 5,
    KRVSH_SIZE_MISMATCH = 6,
    KRVSH_INTERNAL = 255
} krvsh_status_t;

/* init_id = sha256(cpay)[:16]. out16 must point to >= krvsh_init_id_len() bytes. */
krvsh_status_t krvsh_init_id(const uint8_t *cpay, size_t cpay_len, uint8_t *out16);
size_t krvsh_init_id_len(void);

/* Encode one tensor's fpay. x_f32 is row-major (t,d). mean_fp16 = d halfs,
 * basis_fp16 = r*d halfs (little-endian). Two-call size protocol via out/out_cap/written. */
krvsh_status_t krvsh_encode_tensor(const float *x_f32, size_t t, size_t d, size_t r,
                                   const uint8_t *mean_fp16, const uint8_t *basis_fp16,
                                   float s,
                                   uint8_t *out, size_t out_cap, size_t *written);

/* Decode one tensor's fpay into x_hat (row-major (t,d) f32, caller-allocated t*d). */
krvsh_status_t krvsh_decode_tensor(const uint8_t *fpay, size_t fpay_len,
                                   const uint8_t *mean_fp16, const uint8_t *basis_fp16,
                                   size_t d, size_t r, size_t t,
                                   float *x_hat);

/* Model fingerprint over n ordered (key,val) NUL-terminated string pairs -> out32 (32 bytes). */
krvsh_status_t krvsh_model_fingerprint(const char *const *keys, const char *const *vals,
                                       size_t n, uint8_t *out32);

#ifdef __cplusplus
}
#endif

#endif /* KRVSH_H */
