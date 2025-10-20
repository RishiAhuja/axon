#ifndef CRYPTO_DIFFUSION_H
#define CRYPTO_DIFFUSION_H

#include <stdint.h>

#include "../../include/common/config.h"

void mix_columns(char** state);
void shift_rows(char** state);
void inv_mix_columns(char** state);
void inv_shift_rows(char** state);

// Internal scalar implementations (used by SIMD fallback)
void mix_columns_original(char** state);
void shift_rows_original(char** state);
void inv_shift_rows_original(char** state);
void inv_mix_columns_original(char** state);

#endif // CRYPTO_DIFFUSION_H