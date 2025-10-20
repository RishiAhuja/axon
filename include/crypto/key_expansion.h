#ifndef CRYPTO_KEY_EXPANSION_H
#define CRYPTO_KEY_EXPANSION_H

#include <stdio.h>
#include <stdint.h>

void expand_key(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size);
void print_expanded_key(const uint8_t *expanded_key);

// Internal scalar implementation (used by SIMD fallback)
void expand_key_original(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size);

#endif // CRYPTO_KEY_EXPANSION_H