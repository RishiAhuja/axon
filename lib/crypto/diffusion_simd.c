#include "../../include/crypto/diffusion_simd.h"
#include "../../include/crypto/diffusion.h"
#include "../../include/common/optimization.h"
#include <string.h>

// Forward declare the original scalar implementation
extern void mix_columns_original(char** state);

static void (*optimal_mix_columns)(char** state) = NULL;

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
#else
void mix_columns_avx(char** state) {
    mix_columns_sse2(state);
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
#else
void mix_columns_avx2(char** state) {
    mix_columns_sse2(state);
}
#endif // AVX2

#else
// Forward declare the original implementation
extern void mix_columns_original(char** state);

void mix_columns_sse2(char** state) {
    mix_columns_original(state);
}

void mix_columns_avx(char** state) {
    mix_columns_original(state);
}

void mix_columns_avx2(char** state) {
    mix_columns_original(state);
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
}

void mix_columns_simd(char** state) {
    if (optimal_mix_columns == NULL) {
        init_diffusion_simd();
    }
    optimal_mix_columns(state);
}