#include "../../include/common/transformation_config.h"
#include "../../include/common/config.h"
#include "../../include/crypto/confusion_simd.h"
#include <stdint.h>
#include <stdio.h>

void sub_bytes_original(char** state){
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            state[i][j] = sbox[(unsigned char)state[i][j]];
        }
    }
}

void sub_bytes(char** state){
    sub_bytes_simd(state);
}

void inv_sub_bytes_original(char** state){
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            state[i][j] = inv_sbox[(unsigned char)state[i][j]];
        }
    }
}

void inv_sub_bytes(char** state){
    inv_sub_bytes_simd(state);
}


void add_round_key_original(char** state, const uint8_t *round_key){
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            state[i][j] ^= round_key[i * STATE_SIZE + j];
        }
    }
}

void add_round_key(char** state, const uint8_t *round_key){
    add_round_key_simd(state, round_key);
}

void apply_rounds_keys(char** state, const uint8_t *expanded_key) {
    add_round_key(state, expanded_key);
    
    for (size_t round = 0; round < 10; round++) {
        sub_bytes(state);
        add_round_key(state, expanded_key + ((round + 1) * STATE_SIZE * STATE_SIZE));
    }
}

void print_confused_state(char** state){
    for (size_t i = 0; i < STATE_SIZE; i++){
        for (size_t j = 0; j < STATE_SIZE; j++){
            printf("%02x ", (unsigned char)state[i][j]);
        }
        printf("\n");
    }
}