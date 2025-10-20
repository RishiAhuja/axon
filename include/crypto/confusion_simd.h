#ifndef CONFUSION_SIMD_H
#define CONFUSION_SIMD_H

#include <stdint.h>

// Public wrapper functions (with runtime dispatch)
void sub_bytes_simd(char** state);
void inv_sub_bytes_simd(char** state);
void add_round_key_simd(char** state, const uint8_t *round_key);

// Init function
void init_confusion_simd(void);

// Individual variant functions (for testing/direct use)
void sub_bytes_sse2(char** state);
void sub_bytes_avx(char** state);
void sub_bytes_avx2(char** state);

void inv_sub_bytes_sse2(char** state);
void inv_sub_bytes_avx(char** state);
void inv_sub_bytes_avx2(char** state);

void add_round_key_sse2(char** state, const uint8_t *round_key);
void add_round_key_avx(char** state, const uint8_t *round_key);
void add_round_key_avx2(char** state, const uint8_t *round_key);

#endif // CONFUSION_SIMD_H
