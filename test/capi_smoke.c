/* C smoke test for the krvsh codec C ABI. Encode a tensor against an identity-ish fp16
 * basis, decode it, check round-trip within quant tolerance. Run by `zig build test`. */
#include "krvsh.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* minimal f32->fp16 (round-to-nearest-even) for building the test basis/mean */
static uint16_t f32_to_fp16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign; /* flush subnormals to 0 (fine for test) */
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    /* round-to-nearest-even on dropped bits */
    if ((man & 0x1000) && ((man & 0x0FFF) || (h & 1))) h++;
    return h;
}
static void put_fp16(uint8_t *p, float f) { uint16_t h = f32_to_fp16(f); p[0] = h & 0xFF; p[1] = h >> 8; }

static int fail(const char *m) { fprintf(stderr, "krvsh capi_smoke FAIL: %s\n", m); return 1; }

int main(void) {
    if (krvsh_init_id_len() != 16) return fail("init_id_len");
    uint8_t id[16];
    const char *cpay = "pretend-cpay";
    if (krvsh_init_id((const uint8_t *)cpay, strlen(cpay), id) != KRVSH_OK) return fail("init_id");

    /* tiny identity basis: D=R=4, mean=0, Vt[:R] = I -> coef == x. */
    const size_t T = 2, D = 4, R = 4;
    uint8_t mean[4 * 2];
    uint8_t basis[4 * 4 * 2];
    for (size_t i = 0; i < D; i++) put_fp16(&mean[i * 2], 0.0f);
    for (size_t i = 0; i < R * D; i++) put_fp16(&basis[i * 2], 0.0f);
    for (size_t i = 0; i < R; i++) put_fp16(&basis[(i * D + i) * 2], 1.0f);

    float x[2 * 4] = { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f };
    float s = 0.5f;

    size_t need = 0;
    krvsh_encode_tensor(x, T, D, R, mean, basis, s, NULL, 0, &need);
    if (need == 0) return fail("encode size query");
    uint8_t *fpay = (uint8_t *)malloc(need);
    size_t wrote = 0;
    if (krvsh_encode_tensor(x, T, D, R, mean, basis, s, fpay, need, &wrote) != KRVSH_OK) return fail("encode");

    float xhat[2 * 4];
    if (krvsh_decode_tensor(fpay, wrote, mean, basis, D, R, T, xhat) != KRVSH_OK) return fail("decode");

    /* identity basis -> error is pure quantization, <= delta/2; delta = s*rms, rms ~ few.
     * Use a loose bound; this is a smoke test, not a fidelity test. */
    for (size_t i = 0; i < T * D; i++) {
        float e = fabsf(x[i] - xhat[i]);
        if (e > 2.0f) { fprintf(stderr, "x[%zu]=%f xhat=%f\n", i, x[i], xhat[i]); return fail("roundtrip tolerance"); }
    }

    /* fingerprint smoke */
    const char *keys[2] = { "model_id", "revision" };
    const char *vals[2] = { "test/model", "abc" };
    uint8_t fp[32];
    if (krvsh_model_fingerprint(keys, vals, 2, fp) != KRVSH_OK) return fail("fingerprint");

    free(fpay);
    printf("krvsh capi_smoke OK\n");
    return 0;
}
