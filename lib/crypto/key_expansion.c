#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/common/failures.h"
#include "../../include/common/config.h"
#include "../../include/crypto/key_expansion.h"
#include "../../include/crypto/key_expansion_simd.h"
#include <stdint.h>
#include "../../include/common/transformation_config.h"

void expand_key_original(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size)
{
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
    
    for (size_t i = 4; i < 44; i++)
    {
        int prev_word_index = (i-1) * 4;
        for (int j = 0; j < 4; j++) {
            temp[j] = expanded_key[prev_word_index + j];
        }
        
        if(i % 4 == 0) {
            // Rotate
            uint8_t t = temp[0];
            for (int j = 0; j < 3; j++) {
                temp[j] = temp[j + 1];
            }
            temp[3] = t;

            // S-box
            for (int j = 0; j < 4; j++) {
                temp[j] = sbox[temp[j]];
            }

            // Rcon
            temp[0] ^= rcon[i / 4 - 1];
        }
    }

}

void expand_key(const char* key, size_t key_size, char* expanded_key, size_t expanded_key_size)
{
    expand_key_simd(key, key_size, expanded_key, expanded_key_size);
}



void print_expanded_key(const uint8_t *expanded_key) {
    for (int round = 0; round < 11; round++) {
        printf("Round key %2d: ", round);
        for (int byte = 0; byte < 16; byte++) {
            printf("%02x ", expanded_key[round * 16 + byte]);
        }
        printf("\n");
    }
}