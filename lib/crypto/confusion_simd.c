#include "../../include/crypto/confusion_simd.h"
#include "../../include/crypto/confusion.h"
#include "../../include/common/optimization.h"
#include "../../include/common/transformation_config.h"
#include "../../include/common/config.h"
#include <string.h>
#include <stdint.h>

static void (*optimal_sub_bytes)(char** state) = NULL;
static void (*optimal_inv_sub_bytes)(char** state) = NULL;
static void (*optimal_add_round_key)(char** state, const uint8_t *round_key) = NULL;

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <immintrin.h>

void add_round_key_sse2(char** state, const uint8_t *round_key) {
    // Load 16 bytes from state into SIMD register
    __m128i state_vec = _mm_set_epi8(
        state[3][3], state[3][2], state[3][1], state[3][0],
        state[2][3], state[2][2], state[2][1], state[2][0],
        state[1][3], state[1][2], state[1][1], state[1][0],
        state[0][3], state[0][2], state[0][1], state[0][0]
    );

    // Load 16 bytes from round_key
    __m128i key_vec = _mm_loadu_si128((__m128i*)round_key);

    // XOR state with key
    __m128i result = _mm_xor_si128(state_vec, key_vec);

    // Store result back to state
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            state[i][j] = (char)_mm_extract_epi8(result, i * STATE_SIZE + j);
        }
    }
}

void sub_bytes_sse2(char** state) {
    // Load 16 bytes from state into SIMD register
    __m128i state_vec = _mm_set_epi8(
        state[3][3], state[3][2], state[3][1], state[3][0],
        state[2][3], state[2][2], state[2][1], state[2][0],
        state[1][3], state[1][2], state[1][1], state[1][0],
        state[0][3], state[0][2], state[0][1], state[0][0]
    );

    // Extract each byte, perform S-box lookup, store in array
    uint8_t substituted[16];
    for (int i = 0; i < 16; i++) {
        uint8_t byte_val = (uint8_t)_mm_extract_epi8(state_vec, i);
        substituted[i] = sbox[byte_val];
    }

    // Pack substituted bytes back into SIMD register
    __m128i result = _mm_set_epi8(
        substituted[15], substituted[14], substituted[13], substituted[12],
        substituted[11], substituted[10], substituted[9], substituted[8],
        substituted[7], substituted[6], substituted[5], substituted[4],
        substituted[3], substituted[2], substituted[1], substituted[0]
    );

    // Store result back to state
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            state[i][j] = (char)_mm_extract_epi8(result, i * STATE_SIZE + j);
        }
    }
}

void inv_sub_bytes_sse2(char** state) {
    // Load 16 bytes from state into SIMD register
    __m128i state_vec = _mm_set_epi8(
        state[3][3], state[3][2], state[3][1], state[3][0],
        state[2][3], state[2][2], state[2][1], state[2][0],
        state[1][3], state[1][2], state[1][1], state[1][0],
        state[0][3], state[0][2], state[0][1], state[0][0]
    );

    // Extract each byte, perform inverse S-box lookup, store in array
    uint8_t substituted[16];
    for (int i = 0; i < 16; i++) {
        uint8_t byte_val = (uint8_t)_mm_extract_epi8(state_vec, i);
        substituted[i] = inv_sbox[byte_val];
    }

    // Pack substituted bytes back into SIMD register
    __m128i result = _mm_set_epi8(
        substituted[15], substituted[14], substituted[13], substituted[12],
        substituted[11], substituted[10], substituted[9], substituted[8],
        substituted[7], substituted[6], substituted[5], substituted[4],
        substituted[3], substituted[2], substituted[1], substituted[0]
    );

    // Store result back to state
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            state[i][j] = (char)_mm_extract_epi8(result, i * STATE_SIZE + j);
        }
    }
}

#if defined(__AVX__) && !defined(__AVX2__)
void add_round_key_avx(char** state, const uint8_t *round_key) {
    add_round_key_sse2(state, round_key);
}

void sub_bytes_avx(char** state) {
    sub_bytes_sse2(state);
}

void inv_sub_bytes_avx(char** state) {
    inv_sub_bytes_sse2(state);
}
#else
void add_round_key_avx(char** state, const uint8_t *round_key) {
    add_round_key_sse2(state, round_key);
}

void sub_bytes_avx(char** state) {
    sub_bytes_sse2(state);
}

void inv_sub_bytes_avx(char** state) {
    inv_sub_bytes_sse2(state);
}
#endif

#if defined(__AVX2__)
void add_round_key_avx2(char** state, const uint8_t *round_key) {
    add_round_key_sse2(state, round_key);
}

void sub_bytes_avx2(char** state) {
    sub_bytes_sse2(state);
}

void inv_sub_bytes_avx2(char** state) {
    inv_sub_bytes_sse2(state);
}
#else
void add_round_key_avx2(char** state, const uint8_t *round_key) {
    add_round_key_sse2(state, round_key);
}

void sub_bytes_avx2(char** state) {
    sub_bytes_sse2(state);
}

void inv_sub_bytes_avx2(char** state) {
    inv_sub_bytes_sse2(state);
}
#endif // AVX2

#else
void add_round_key_sse2(char** state, const uint8_t *round_key) {
    add_round_key_original(state, round_key);
}

void add_round_key_avx(char** state, const uint8_t *round_key) {
    add_round_key_original(state, round_key);
}

void add_round_key_avx2(char** state, const uint8_t *round_key) {
    add_round_key_original(state, round_key);
}

void sub_bytes_sse2(char** state) {
    sub_bytes_original(state);
}

void sub_bytes_avx(char** state) {
    sub_bytes_original(state);
}

void sub_bytes_avx2(char** state) {
    sub_bytes_original(state);
}

void inv_sub_bytes_sse2(char** state) {
    inv_sub_bytes_original(state);
}

void inv_sub_bytes_avx(char** state) {
    inv_sub_bytes_original(state);
}

void inv_sub_bytes_avx2(char** state) {
    inv_sub_bytes_original(state);
}
#endif // SSE2

void init_confusion_simd(void) {
    if (optimal_add_round_key != NULL) {
        return;
    }

    optimal_add_round_key = get_optimal_implementation(
        (void*)add_round_key_original,
        (void*)add_round_key_sse2,
        (void*)add_round_key_avx,
        (void*)add_round_key_avx2,
        &g_opt_settings
    );

    optimal_sub_bytes = get_optimal_implementation(
        (void*)sub_bytes_original,
        (void*)sub_bytes_sse2,
        (void*)sub_bytes_avx,
        (void*)sub_bytes_avx2,
        &g_opt_settings
    );

    optimal_inv_sub_bytes = get_optimal_implementation(
        (void*)inv_sub_bytes_original,
        (void*)inv_sub_bytes_sse2,
        (void*)inv_sub_bytes_avx,
        (void*)inv_sub_bytes_avx2,
        &g_opt_settings
    );
}

void add_round_key_simd(char** state, const uint8_t *round_key) {
    if (optimal_add_round_key == NULL) {
        init_confusion_simd();
    }
    optimal_add_round_key(state, round_key);
}

void sub_bytes_simd(char** state) {
    if (optimal_sub_bytes == NULL) {
        init_confusion_simd();
    }
    optimal_sub_bytes(state);
}

void inv_sub_bytes_simd(char** state) {
    if (optimal_inv_sub_bytes == NULL) {
        init_confusion_simd();
    }
    optimal_inv_sub_bytes(state);
}
