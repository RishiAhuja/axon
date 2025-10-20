#ifndef CRYTPO_CONSUSION_H
#define CRYTPO_CONSUSION_H

#include <stdint.h>

void sub_bytes(char** state);
void add_round_key(char** state, const uint8_t *round_key);
void apply_rounds_keys(char** state, const uint8_t *expanded_key);
void print_confused_state(char** state);
void inv_sub_bytes(char** state);

// Internal scalar implementations (used by SIMD fallback)
void sub_bytes_original(char** state);
void inv_sub_bytes_original(char** state);
void add_round_key_original(char** state, const uint8_t *round_key);

#endif