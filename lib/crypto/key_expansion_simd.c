#include "../../include/crypto/key_expansion_simd.h"
#include "../../include/crypto/key_expansion.h"
#include "../../include/common/optimization.h"
#include "../../include/common/transformation_config.h"
#include "../../include/common/config.h"
#include "../../include/common/failures.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

static void (*optimal_expand_key)(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) = NULL;

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <immintrin.h>

void expand_key_sse2(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) {
    if (key_size != STATE_SIZE * STATE_SIZE) {
        fprintf(stderr, "Invalid key size\n");
        return;
    }

    if (expanded_key_size != EXPANDED_KEY_SIZE) {
        fprintf(stderr, "Invalid expanded key size\n");
        return;
    }

    memcpy(expanded_key, key, key_size);
    uint8_t temp[4];

    for (size_t i = 4; i < 44; i++) {
        int prev_word_index = (i-1) * 4;
        for (int j = 0; j < 4; j++) {
            temp[j] = expanded_key[prev_word_index + j];
        }

        if (i % 4 == 0) {
            // Rotate using SSE2 shuffle
            __m128i temp_vec = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            temp[3], temp[2], temp[1], temp[0]);
            __m128i rotated = _mm_shuffle_epi8(temp_vec,
                                               _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8,
                                                           7, 6, 5, 4, 0, 3, 2, 1));

            // Extract rotated bytes
            temp[0] = (uint8_t)_mm_extract_epi8(rotated, 0);
            temp[1] = (uint8_t)_mm_extract_epi8(rotated, 1);
            temp[2] = (uint8_t)_mm_extract_epi8(rotated, 2);
            temp[3] = (uint8_t)_mm_extract_epi8(rotated, 3);

            // S-box lookups
            for (int j = 0; j < 4; j++) {
                temp[j] = sbox[temp[j]];
            }

            // Rcon
            temp[0] ^= rcon[i / 4 - 1];
        }
    }
}

#if defined(__AVX__) && !defined(__AVX2__)
void expand_key_avx(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) {
    expand_key_sse2(key, key_size, expanded_key, expanded_key_size);
}
#else
void expand_key_avx(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) {
    expand_key_sse2(key, key_size, expanded_key, expanded_key_size);
}
#endif

#if defined(__AVX2__)
void expand_key_avx2(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) {
    expand_key_sse2(key, key_size, expanded_key, expanded_key_size);
}
#else
void expand_key_avx2(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) {
    expand_key_sse2(key, key_size, expanded_key, expanded_key_size);
}
#endif // AVX2

#else
void expand_key_sse2(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) {
    expand_key_original(key, key_size, expanded_key, expanded_key_size);
}

void expand_key_avx(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) {
    expand_key_original(key, key_size, expanded_key, expanded_key_size);
}

void expand_key_avx2(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) {
    expand_key_original(key, key_size, expanded_key, expanded_key_size);
}
#endif // SSE2

void init_key_expansion_simd(void) {
    if (optimal_expand_key != NULL) {
        return;
    }

    optimal_expand_key = get_optimal_implementation(
        (void*)expand_key_original,
        (void*)expand_key_sse2,
        (void*)expand_key_avx,
        (void*)expand_key_avx2,
        &g_opt_settings
    );
}

void expand_key_simd(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size) {
    if (optimal_expand_key == NULL) {
        init_key_expansion_simd();
    }
    optimal_expand_key(key, key_size, expanded_key, expanded_key_size);
}
