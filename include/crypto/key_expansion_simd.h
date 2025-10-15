#ifndef KEY_EXPANSION_SIMD_H
#define KEY_EXPANSION_SIMD_H

#include <stdio.h>
#include <stdint.h>

// Public wrapper function (with runtime dispatch)
void expand_key_simd(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size);

// Init function
void init_key_expansion_simd(void);

// Individual variant functions (for testing/direct use)
void expand_key_sse2(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size);
void expand_key_avx(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size);
void expand_key_avx2(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size);

#endif // KEY_EXPANSION_SIMD_H
