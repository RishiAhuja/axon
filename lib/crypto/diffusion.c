// mixin columns

#include "../../include/common/config.h"
#include <stdio.h>
#include <stdint.h>
#include "../../include/crypto/diffusion_simd.h"

char multiply_by_2(uint8_t byte) {
    return (byte << 1) ^ ((byte & 0x80) ? 0x1b : 0);
}

char multiply_by_3(uint8_t byte) {
    return multiply_by_2(byte) ^ byte;
}

char multiply_by_9(char byte) {
    return multiply_by_2(multiply_by_2(multiply_by_2(byte))) ^ byte;
}

char multiply_by_11(char byte) {
    return multiply_by_2(multiply_by_2(multiply_by_2(byte)) ^ byte) ^ byte;
}

char multiply_by_13(char byte) {
    return multiply_by_2(multiply_by_2(multiply_by_2(byte) ^ byte)) ^ byte;
}

char multiply_by_14(char byte) {
    return multiply_by_2(multiply_by_2(multiply_by_2(byte) ^ byte) ^ byte);
}

// | 2 3 1 1 |
// | 1 2 3 1 |
// | 1 1 2 3 |
// | 3 1 1 2 |

void mix_columns_original(char** state){
    char temp[STATE_SIZE][STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++){
            temp[i][j] = state[i][j];
        }
    }
    for (int j = 0; j < STATE_SIZE; j++) {
        state[0][j] = (multiply_by_2(temp[0][j]) ^ multiply_by_3(temp[1][j]) ^ temp[2][j] ^ temp[3][j]);
        
        state[1][j] = (temp[0][j] ^ multiply_by_2(temp[1][j]) ^ multiply_by_3(temp[2][j]) ^ temp[3][j]);
            
        state[2][j] = (temp[0][j] ^ temp[1][j] ^ multiply_by_2(temp[2][j]) ^ multiply_by_3(temp[3][j]));
            
        state[3][j] = (multiply_by_3(temp[0][j]) ^ temp[1][j] ^ temp[2][j] ^ multiply_by_2(temp[3][j]));
    }
}

void mix_columns(char** state){
    mix_columns_simd(state);
}

void shift_rows_original(char** state){
    char temp[STATE_SIZE][STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++){
            temp[i][j] = state[i][j];
        }
    }

    char t1_1 = temp[1][0];
    for (size_t j = 0; j < STATE_SIZE - 1; j++){
        temp[1][j] = temp[1][j + 1];
    }
    temp[1][STATE_SIZE - 1] = t1_1;

    char t2_1 = temp[2][0];
    char t2_2 = temp[2][1];
    for (size_t j = 0; j < STATE_SIZE - 2; j++){
        temp[2][j] = temp[2][j + 2];
    }
    temp[2][STATE_SIZE - 2] = t2_1;
    temp[2][STATE_SIZE - 1] = t2_2;

    char t3_1 = temp[3][0];
    char t3_2 = temp[3][1];
    char t3_3 = temp[3][2];
    for (size_t j = 0; j < STATE_SIZE - 3; j++){
        temp[3][j] = temp[3][j + 3];
    }
    temp[3][STATE_SIZE - 1] = t3_3;
    temp[3][STATE_SIZE - 2] = t3_2;
    temp[3][STATE_SIZE - 3] = t3_1;
    

    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++){
            state[i][j] = temp[i][j];
        }
    }
}

void shift_rows(char** state){
    shift_rows_simd(state);
}


void inv_mix_columns_original(char** state) {
    char temp[STATE_SIZE][STATE_SIZE];
    
    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++) {
            temp[i][j] = state[i][j];
        }
    }

    for (int j = 0; j < STATE_SIZE; j++) {
        state[0][j] = multiply_by_14(temp[0][j]) ^ multiply_by_11(temp[1][j]) ^ 
                      multiply_by_13(temp[2][j]) ^ multiply_by_9(temp[3][j]);
                        
        state[1][j] = multiply_by_9(temp[0][j]) ^ multiply_by_14(temp[1][j]) ^ 
                      multiply_by_11(temp[2][j]) ^ multiply_by_13(temp[3][j]);
                       
        state[2][j] = multiply_by_13(temp[0][j]) ^ multiply_by_9(temp[1][j]) ^ 
                      multiply_by_14(temp[2][j]) ^ multiply_by_11(temp[3][j]);
                       
        state[3][j] = multiply_by_11(temp[0][j]) ^ multiply_by_13(temp[1][j]) ^ 
                      multiply_by_9(temp[2][j]) ^ multiply_by_14(temp[3][j]);
    }
}

void inv_mix_columns(char** state) {
    inv_mix_columns_simd(state);
}


void inv_shift_rows_original(char** state) {
    char temp[STATE_SIZE][STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++) {
            temp[i][j] = state[i][j];
        }
    }
 
    char t1 = temp[1][STATE_SIZE - 1];
    for (int j = STATE_SIZE - 1; j > 0; j--) {
        temp[1][j] = temp[1][j - 1];
    }
    temp[1][0] = t1;

    char t2_0 = temp[2][STATE_SIZE - 1];
    char t2_1 = temp[2][STATE_SIZE - 2];
    for (int j = STATE_SIZE - 1; j > 1; j--) {
        temp[2][j] = temp[2][j - 2];
    }
    temp[2][1] = t2_0;
    temp[2][0] = t2_1;

    char t3_0 = temp[3][STATE_SIZE - 1];
    char t3_1 = temp[3][STATE_SIZE - 2];
    char t3_2 = temp[3][STATE_SIZE - 3];
    for (int j = STATE_SIZE - 1; j > 2; j--) {
        temp[3][j] = temp[3][j - 3];
    }
    temp[3][2] = t3_0;
    temp[3][1] = t3_1;
    temp[3][0] = t3_2;

    for (int i = 0; i < STATE_SIZE; i++) {
        for (size_t j = 0; j < STATE_SIZE; j++) {
            state[i][j] = temp[i][j];
        }
    }
}

void inv_shift_rows(char** state) {
    inv_shift_rows_simd(state);
}