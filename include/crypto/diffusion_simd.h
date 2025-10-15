#ifndef DIFFUSION_SIMD_H
#define DIFFUSION_SIMD_H

// Public wrapper functions (with runtime dispatch)
void mix_columns_simd(char** state);
void shift_rows_simd(char** state);
void inv_shift_rows_simd(char** state);
void inv_mix_columns_simd(char** state);

// Init function
void init_diffusion_simd(void);

// Individual variant functions (for testing/direct use)
void mix_columns_sse2(char** state);
void mix_columns_avx(char** state); 
void mix_columns_avx2(char** state);

void shift_rows_sse2(char** state);
void shift_rows_avx(char** state);
void shift_rows_avx2(char** state);

void inv_shift_rows_sse2(char** state);
void inv_shift_rows_avx(char** state);
void inv_shift_rows_avx2(char** state);

void inv_mix_columns_sse2(char** state);
void inv_mix_columns_avx(char** state);
void inv_mix_columns_avx2(char** state);

#endif // DIFFUSION_SIMD_H