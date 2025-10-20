#include "../../include/crypto/diffusion_simd.h"
#include "../../include/crypto/diffusion.h"
#include "../../include/common/optimization.h"
#include <string.h>

static void (*optimal_mix_columns)(char** state) = NULL;
static void (*optimal_shift_rows)(char** state) = NULL;
static void (*optimal_inv_shift_rows)(char** state) = NULL;
static void (*optimal_inv_mix_columns)(char** state) = NULL;

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <immintrin.h>

void mix_columns_sse2(char** state) {
    char temp[STATE_SIZE][STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++){
            temp[i][j] = state[i][j];
        }
    }

    for (size_t i = 0; i < STATE_SIZE; i++) {
        uint8_t byte_0j = (uint8_t)temp[0][i];
        uint8_t byte_1j = (uint8_t)temp[1][i];
        uint8_t byte_2j = (uint8_t)temp[2][i];
        uint8_t byte_3j = (uint8_t)temp[3][i];

        __m128i column_bytes = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                           byte_3j, byte_2j, byte_1j, byte_0j);
        __m128i high_bit_mask = _mm_set1_epi8(0x80);
        __m128i has_high_mask = _mm_and_si128(column_bytes, high_bit_mask);
        has_high_mask = _mm_cmpeq_epi8(has_high_mask, high_bit_mask);

        __m128i shifted_bytes = _mm_slli_epi16(column_bytes, 1);
        shifted_bytes = _mm_and_si128(shifted_bytes, _mm_set1_epi8(0xFF));

        __m128i reduction_mask = _mm_and_si128(has_high_mask, _mm_set1_epi8(0x1B));
        __m128i multiplied_by_2 = _mm_xor_si128(shifted_bytes, reduction_mask);

        __m128i multiplied_by_3 = _mm_xor_si128(multiplied_by_2, column_bytes);

        uint8_t byte_0j_times_2 = _mm_extract_epi8(multiplied_by_2, 0);
        uint8_t byte_1j_times_2 = _mm_extract_epi8(multiplied_by_2, 1);
        uint8_t byte_2j_times_2 = _mm_extract_epi8(multiplied_by_2, 2);
        uint8_t byte_3j_times_2 = _mm_extract_epi8(multiplied_by_2, 3);
        
        uint8_t byte_0j_times_3 = _mm_extract_epi8(multiplied_by_3, 0);
        uint8_t byte_1j_times_3 = _mm_extract_epi8(multiplied_by_3, 1);
        uint8_t byte_2j_times_3 = _mm_extract_epi8(multiplied_by_3, 2);
        uint8_t byte_3j_times_3 = _mm_extract_epi8(multiplied_by_3, 3);

        state[0][i] = (char)(byte_0j_times_2 ^ byte_1j_times_3 ^ byte_2j ^ byte_3j);
        state[1][i] = (char)(byte_0j ^ byte_1j_times_2 ^ byte_2j_times_3 ^ byte_3j);
        state[2][i] = (char)(byte_0j ^ byte_1j ^ byte_2j_times_2 ^ byte_3j_times_3);
        state[3][i] = (char)(byte_0j_times_3 ^ byte_1j ^ byte_2j ^ byte_3j_times_2);
    }
}

void shift_rows_sse2(char** state) {
    // Load entire 16-byte state into SSE2 register
    __m128i state_vec = _mm_set_epi8(
        state[3][3], state[3][2], state[3][1], state[3][0],
        state[2][3], state[2][2], state[2][1], state[2][0],
        state[1][3], state[1][2], state[1][1], state[1][0],
        state[0][3], state[0][2], state[0][1], state[0][0]
    );

    // Shuffle mask for shift_rows: row 0 unchanged, row 1 left by 1, row 2 left by 2, row 3 left by 3
    // Original layout: [0][0-3], [1][0-3], [2][0-3], [3][0-3]
    // After shift: [0][0-3], [1][1-3,0], [2][2-3,0-1], [3][3,0-2]
    __m128i shuffle_mask = _mm_set_epi8(
        12, 15, 14, 13,  // row 3: [3][3], [3][0], [3][1], [3][2] -> left by 3
        10, 9, 8, 11,    // row 2: [2][2], [2][3], [2][0], [2][1] -> left by 2
        5, 4, 7, 6,      // row 1: [1][1], [1][2], [1][3], [1][0] -> left by 1
        3, 2, 1, 0       // row 0: [0][0], [0][1], [0][2], [0][3] -> unchanged
    );

    __m128i result = _mm_shuffle_epi8(state_vec, shuffle_mask);

    // Store result back to state
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            state[i][j] = (char)_mm_extract_epi8(result, i * STATE_SIZE + j);
        }
    }
}

void inv_shift_rows_sse2(char** state) {
    // Load entire 16-byte state into SSE2 register
    __m128i state_vec = _mm_set_epi8(
        state[3][3], state[3][2], state[3][1], state[3][0],
        state[2][3], state[2][2], state[2][1], state[2][0],
        state[1][3], state[1][2], state[1][1], state[1][0],
        state[0][3], state[0][2], state[0][1], state[0][0]
    );

    // Shuffle mask for inv_shift_rows: row 0 unchanged, row 1 right by 1, row 2 right by 2, row 3 right by 3
    // After inv shift: [0][0-3], [1][3,0-2], [2][2-3,0-1], [3][1-3,0]
    __m128i shuffle_mask = _mm_set_epi8(
        13, 14, 15, 12,  // row 3: [3][1], [3][2], [3][3], [3][0] -> right by 3
        10, 11, 8, 9,    // row 2: [2][2], [2][3], [2][0], [2][1] -> right by 2
        7, 4, 5, 6,      // row 1: [1][3], [1][0], [1][1], [1][2] -> right by 1
        3, 2, 1, 0       // row 0: [0][0], [0][1], [0][2], [0][3] -> unchanged
    );

    __m128i result = _mm_shuffle_epi8(state_vec, shuffle_mask);

    // Store result back to state
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            state[i][j] = (char)_mm_extract_epi8(result, i * STATE_SIZE + j);
        }
    }
}

void inv_mix_columns_sse2(char** state) {
    char temp[STATE_SIZE][STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++) {
            temp[i][j] = state[i][j];
        }
    }

    for (size_t i = 0; i < STATE_SIZE; i++) {
        uint8_t byte_0j = (uint8_t)temp[0][i];
        uint8_t byte_1j = (uint8_t)temp[1][i];
        uint8_t byte_2j = (uint8_t)temp[2][i];
        uint8_t byte_3j = (uint8_t)temp[3][i];

        __m128i column_bytes = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           byte_3j, byte_2j, byte_1j, byte_0j);

        // Compute multiply by 2
        __m128i high_bit_mask = _mm_set1_epi8(0x80);
        __m128i has_high_mask = _mm_and_si128(column_bytes, high_bit_mask);
        has_high_mask = _mm_cmpeq_epi8(has_high_mask, high_bit_mask);
        __m128i shifted_bytes = _mm_slli_epi16(column_bytes, 1);
        shifted_bytes = _mm_and_si128(shifted_bytes, _mm_set1_epi8(0xFF));
        __m128i reduction_mask = _mm_and_si128(has_high_mask, _mm_set1_epi8(0x1B));
        __m128i multiplied_by_2 = _mm_xor_si128(shifted_bytes, reduction_mask);

        // Compute multiply by 4 (2 * 2)
        has_high_mask = _mm_and_si128(multiplied_by_2, high_bit_mask);
        has_high_mask = _mm_cmpeq_epi8(has_high_mask, high_bit_mask);
        shifted_bytes = _mm_slli_epi16(multiplied_by_2, 1);
        shifted_bytes = _mm_and_si128(shifted_bytes, _mm_set1_epi8(0xFF));
        reduction_mask = _mm_and_si128(has_high_mask, _mm_set1_epi8(0x1B));
        __m128i multiplied_by_4 = _mm_xor_si128(shifted_bytes, reduction_mask);

        // Compute multiply by 8 (4 * 2)
        has_high_mask = _mm_and_si128(multiplied_by_4, high_bit_mask);
        has_high_mask = _mm_cmpeq_epi8(has_high_mask, high_bit_mask);
        shifted_bytes = _mm_slli_epi16(multiplied_by_4, 1);
        shifted_bytes = _mm_and_si128(shifted_bytes, _mm_set1_epi8(0xFF));
        reduction_mask = _mm_and_si128(has_high_mask, _mm_set1_epi8(0x1B));
        __m128i multiplied_by_8 = _mm_xor_si128(shifted_bytes, reduction_mask);

        // Compute required multipliers: 9 = 8^1, 11 = 8^2^1, 13 = 8^4^1, 14 = 8^4^2
        __m128i multiplied_by_9 = _mm_xor_si128(multiplied_by_8, column_bytes);
        __m128i multiplied_by_11 = _mm_xor_si128(multiplied_by_8, _mm_xor_si128(multiplied_by_2, column_bytes));
        __m128i multiplied_by_13 = _mm_xor_si128(multiplied_by_8, _mm_xor_si128(multiplied_by_4, column_bytes));
        __m128i multiplied_by_14 = _mm_xor_si128(multiplied_by_8, _mm_xor_si128(multiplied_by_4, multiplied_by_2));

        // Extract all needed values
        uint8_t byte_0j_times_9 = _mm_extract_epi8(multiplied_by_9, 0);
        uint8_t byte_1j_times_9 = _mm_extract_epi8(multiplied_by_9, 1);
        uint8_t byte_2j_times_9 = _mm_extract_epi8(multiplied_by_9, 2);
        uint8_t byte_3j_times_9 = _mm_extract_epi8(multiplied_by_9, 3);

        uint8_t byte_0j_times_11 = _mm_extract_epi8(multiplied_by_11, 0);
        uint8_t byte_1j_times_11 = _mm_extract_epi8(multiplied_by_11, 1);
        uint8_t byte_2j_times_11 = _mm_extract_epi8(multiplied_by_11, 2);
        uint8_t byte_3j_times_11 = _mm_extract_epi8(multiplied_by_11, 3);

        uint8_t byte_0j_times_13 = _mm_extract_epi8(multiplied_by_13, 0);
        uint8_t byte_1j_times_13 = _mm_extract_epi8(multiplied_by_13, 1);
        uint8_t byte_2j_times_13 = _mm_extract_epi8(multiplied_by_13, 2);
        uint8_t byte_3j_times_13 = _mm_extract_epi8(multiplied_by_13, 3);

        uint8_t byte_0j_times_14 = _mm_extract_epi8(multiplied_by_14, 0);
        uint8_t byte_1j_times_14 = _mm_extract_epi8(multiplied_by_14, 1);
        uint8_t byte_2j_times_14 = _mm_extract_epi8(multiplied_by_14, 2);
        uint8_t byte_3j_times_14 = _mm_extract_epi8(multiplied_by_14, 3);

        // Apply inverse mix columns matrix
        state[0][i] = (char)(byte_0j_times_14 ^ byte_1j_times_11 ^ byte_2j_times_13 ^ byte_3j_times_9);
        state[1][i] = (char)(byte_0j_times_9 ^ byte_1j_times_14 ^ byte_2j_times_11 ^ byte_3j_times_13);
        state[2][i] = (char)(byte_0j_times_13 ^ byte_1j_times_9 ^ byte_2j_times_14 ^ byte_3j_times_11);
        state[3][i] = (char)(byte_0j_times_11 ^ byte_1j_times_13 ^ byte_2j_times_9 ^ byte_3j_times_14);
    }
}

#if defined(__AVX__) && !defined(__AVX2__)
#include <immintrin.h>

void mix_columns_avx(char** state) {
    char temp[STATE_SIZE][STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++){
            temp[i][j] = state[i][j];
        }
    }

    for (size_t i = 0; i < STATE_SIZE; i++) {
        uint8_t byte_0j = (uint8_t)temp[0][i];
        uint8_t byte_1j = (uint8_t)temp[1][i];
        uint8_t byte_2j = (uint8_t)temp[2][i];
        uint8_t byte_3j = (uint8_t)temp[3][i];

        __m128i column_bytes = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                           byte_3j, byte_2j, byte_1j, byte_0j);
        __m128i high_bit_mask = _mm_set1_epi8(0x80);
        __m128i has_high_mask = _mm_and_si128(column_bytes, high_bit_mask);
        has_high_mask = _mm_cmpeq_epi8(has_high_mask, high_bit_mask);

        __m128i shifted_bytes = _mm_slli_epi16(column_bytes, 1);
        shifted_bytes = _mm_and_si128(shifted_bytes, _mm_set1_epi8(0xFF));

        __m128i reduction_mask = _mm_and_si128(has_high_mask, _mm_set1_epi8(0x1B));
        __m128i multiplied_by_2 = _mm_xor_si128(shifted_bytes, reduction_mask);

        __m128i multiplied_by_3 = _mm_xor_si128(multiplied_by_2, column_bytes);

        uint8_t byte_0j_times_2 = _mm_extract_epi8(multiplied_by_2, 0);
        uint8_t byte_1j_times_2 = _mm_extract_epi8(multiplied_by_2, 1);
        uint8_t byte_2j_times_2 = _mm_extract_epi8(multiplied_by_2, 2);
        uint8_t byte_3j_times_2 = _mm_extract_epi8(multiplied_by_2, 3);
        
        uint8_t byte_0j_times_3 = _mm_extract_epi8(multiplied_by_3, 0);
        uint8_t byte_1j_times_3 = _mm_extract_epi8(multiplied_by_3, 1);
        uint8_t byte_2j_times_3 = _mm_extract_epi8(multiplied_by_3, 2);
        uint8_t byte_3j_times_3 = _mm_extract_epi8(multiplied_by_3, 3);

        // Pre-compute parts of the MixColumns formula using AVX instruction throughput
        __m128i result_parts = _mm_set_epi32(
            byte_0j_times_3 ^ byte_1j ^ byte_2j ^ byte_3j_times_2,
            byte_0j ^ byte_1j ^ byte_2j_times_2 ^ byte_3j_times_3,
            byte_0j ^ byte_1j_times_2 ^ byte_2j_times_3 ^ byte_3j,
            byte_0j_times_2 ^ byte_1j_times_3 ^ byte_2j ^ byte_3j
        );
        
        state[0][i] = (char)_mm_extract_epi8(result_parts, 0);
        state[1][i] = (char)_mm_extract_epi8(result_parts, 4);
        state[2][i] = (char)_mm_extract_epi8(result_parts, 8);
        state[3][i] = (char)_mm_extract_epi8(result_parts, 12);
    }
}

void shift_rows_avx(char** state) {
    shift_rows_sse2(state);
}

void inv_shift_rows_avx(char** state) {
    inv_shift_rows_sse2(state);
}

void inv_mix_columns_avx(char** state) {
    inv_mix_columns_sse2(state);
}
#else
void mix_columns_avx(char** state) {
    mix_columns_sse2(state);
}

void shift_rows_avx(char** state) {
    shift_rows_sse2(state);
}

void inv_shift_rows_avx(char** state) {
    inv_shift_rows_sse2(state);
}

void inv_mix_columns_avx(char** state) {
    inv_mix_columns_sse2(state);
}
#endif


#if defined(__AVX2__)
void mix_columns_avx2(char** state) {
    char temp[STATE_SIZE][STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++) {
            temp[i][j] = state[i][j];
        }
    }

    for (int pair = 0; pair < 2; pair++) {
        int col1 = pair * 2;
        int col2 = col1 + 1;
        
        uint8_t c1_byte_0 = (uint8_t)temp[0][col1];
        uint8_t c1_byte_1 = (uint8_t)temp[1][col1];
        uint8_t c1_byte_2 = (uint8_t)temp[2][col1];
        uint8_t c1_byte_3 = (uint8_t)temp[3][col1];
        
        uint8_t c2_byte_0 = (uint8_t)temp[0][col2];
        uint8_t c2_byte_1 = (uint8_t)temp[1][col2];
        uint8_t c2_byte_2 = (uint8_t)temp[2][col2];
        uint8_t c2_byte_3 = (uint8_t)temp[3][col2];

        __m256i columns = _mm256_set_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c2_byte_3, c2_byte_2, c2_byte_1, c2_byte_0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c1_byte_3, c1_byte_2, c1_byte_1, c1_byte_0
        );
        
        __m256i high_bit_mask = _mm256_set1_epi8(0x80);
        __m256i has_high_bit = _mm256_and_si256(columns, high_bit_mask);
        has_high_bit = _mm256_cmpeq_epi8(has_high_bit, high_bit_mask);
        
        __m256i shifted = _mm256_slli_epi16(columns, 1);
        shifted = _mm256_and_si256(shifted, _mm256_set1_epi8(0xFF));
        
        __m256i reduction = _mm256_and_si256(has_high_bit, _mm256_set1_epi8(0x1B));
        __m256i multiplied_by_2 = _mm256_xor_si256(shifted, reduction);
        
        __m256i multiplied_by_3 = _mm256_xor_si256(multiplied_by_2, columns);
        
        uint8_t c1_byte_0_times_2 = _mm256_extract_epi8(multiplied_by_2, 0);
        uint8_t c1_byte_1_times_2 = _mm256_extract_epi8(multiplied_by_2, 1);
        uint8_t c1_byte_2_times_2 = _mm256_extract_epi8(multiplied_by_2, 2);
        uint8_t c1_byte_3_times_2 = _mm256_extract_epi8(multiplied_by_2, 3);
        
        uint8_t c1_byte_0_times_3 = _mm256_extract_epi8(multiplied_by_3, 0);
        uint8_t c1_byte_1_times_3 = _mm256_extract_epi8(multiplied_by_3, 1);
        uint8_t c1_byte_2_times_3 = _mm256_extract_epi8(multiplied_by_3, 2);
        uint8_t c1_byte_3_times_3 = _mm256_extract_epi8(multiplied_by_3, 3);
        
        uint8_t c2_byte_0_times_2 = _mm256_extract_epi8(multiplied_by_2, 16);
        uint8_t c2_byte_1_times_2 = _mm256_extract_epi8(multiplied_by_2, 17);
        uint8_t c2_byte_2_times_2 = _mm256_extract_epi8(multiplied_by_2, 18);
        uint8_t c2_byte_3_times_2 = _mm256_extract_epi8(multiplied_by_2, 19);
        
        uint8_t c2_byte_0_times_3 = _mm256_extract_epi8(multiplied_by_3, 16);
        uint8_t c2_byte_1_times_3 = _mm256_extract_epi8(multiplied_by_3, 17);
        uint8_t c2_byte_2_times_3 = _mm256_extract_epi8(multiplied_by_3, 18);
        uint8_t c2_byte_3_times_3 = _mm256_extract_epi8(multiplied_by_3, 19);
        
        state[0][col1] = (char)(c1_byte_0_times_2 ^ c1_byte_1_times_3 ^ c1_byte_2 ^ c1_byte_3);
        state[1][col1] = (char)(c1_byte_0 ^ c1_byte_1_times_2 ^ c1_byte_2_times_3 ^ c1_byte_3);
        state[2][col1] = (char)(c1_byte_0 ^ c1_byte_1 ^ c1_byte_2_times_2 ^ c1_byte_3_times_3);
        state[3][col1] = (char)(c1_byte_0_times_3 ^ c1_byte_1 ^ c1_byte_2 ^ c1_byte_3_times_2);
        
        state[0][col2] = (char)(c2_byte_0_times_2 ^ c2_byte_1_times_3 ^ c2_byte_2 ^ c2_byte_3);
        state[1][col2] = (char)(c2_byte_0 ^ c2_byte_1_times_2 ^ c2_byte_2_times_3 ^ c2_byte_3);
        state[2][col2] = (char)(c2_byte_0 ^ c2_byte_1 ^ c2_byte_2_times_2 ^ c2_byte_3_times_3);
        state[3][col2] = (char)(c2_byte_0_times_3 ^ c2_byte_1 ^ c2_byte_2 ^ c2_byte_3_times_2);
    }
}

void shift_rows_avx2(char** state) {
    shift_rows_sse2(state);
}

void inv_shift_rows_avx2(char** state) {
    inv_shift_rows_sse2(state);
}

void inv_mix_columns_avx2(char** state) {
    inv_mix_columns_sse2(state);
}
#else
void mix_columns_avx2(char** state) {
    mix_columns_sse2(state);
}

void shift_rows_avx2(char** state) {
    shift_rows_sse2(state);
}

void inv_shift_rows_avx2(char** state) {
    inv_shift_rows_sse2(state);
}

void inv_mix_columns_avx2(char** state) {
    inv_mix_columns_sse2(state);
}
#endif // AVX2

#else
void mix_columns_sse2(char** state) {
    mix_columns_original(state);
}

void mix_columns_avx2(char** state) {
    mix_columns_original(state);
}

void shift_rows_sse2(char** state) {
    shift_rows_original(state);
}

void shift_rows_avx(char** state) {
    shift_rows_original(state);
}

void shift_rows_avx2(char** state) {
    shift_rows_original(state);
}

void inv_shift_rows_sse2(char** state) {
    inv_shift_rows_original(state);
}

void inv_shift_rows_avx(char** state) {
    inv_shift_rows_original(state);
}

void inv_shift_rows_avx2(char** state) {
    inv_shift_rows_original(state);
}

void inv_mix_columns_sse2(char** state) {
    inv_mix_columns_original(state);
}

void inv_mix_columns_avx(char** state) {
    inv_mix_columns_original(state);
}

void inv_mix_columns_avx2(char** state) {
    inv_mix_columns_original(state);
}
#endif // SSE2


void init_diffusion_simd(void) {
    if (optimal_mix_columns != NULL) {
        return;
    }
    
    optimal_mix_columns = get_optimal_implementation(
        (void*)mix_columns_original,      
        (void*)mix_columns_sse2,  
        (void*)mix_columns_avx,                     
        (void*)mix_columns_avx2,  
        &g_opt_settings
    );

    optimal_shift_rows = get_optimal_implementation(
        (void*)shift_rows_original,
        (void*)shift_rows_sse2,
        (void*)shift_rows_avx,
        (void*)shift_rows_avx2,
        &g_opt_settings
    );

    optimal_inv_shift_rows = get_optimal_implementation(
        (void*)inv_shift_rows_original,
        (void*)inv_shift_rows_sse2,
        (void*)inv_shift_rows_avx,
        (void*)inv_shift_rows_avx2,
        &g_opt_settings
    );

    optimal_inv_mix_columns = get_optimal_implementation(
        (void*)inv_mix_columns_original,
        (void*)inv_mix_columns_sse2,
        (void*)inv_mix_columns_avx,
        (void*)inv_mix_columns_avx2,
        &g_opt_settings
    );
}

void mix_columns_simd(char** state) {
    if (optimal_mix_columns == NULL) {
        init_diffusion_simd();
    }
    optimal_mix_columns(state);
}

void shift_rows_simd(char** state) {
    if (optimal_shift_rows == NULL) {
        init_diffusion_simd();
    }
    optimal_shift_rows(state);
}

void inv_shift_rows_simd(char** state) {
    if (optimal_inv_shift_rows == NULL) {
        init_diffusion_simd();
    }
    optimal_inv_shift_rows(state);
}

void inv_mix_columns_simd(char** state) {
    if (optimal_inv_mix_columns == NULL) {
        init_diffusion_simd();
    }
    optimal_inv_mix_columns(state);
}