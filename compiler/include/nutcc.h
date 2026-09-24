#pragma once
#define NT_KERNEL extern "C"
static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(void*) == 8,
              "NutCC requires 32-bit int/float and 64-bit frontend pointers");
struct nutcc_dim3 { const unsigned int x, y, z; };
extern const nutcc_dim3 threadIdx, blockIdx, blockDim, gridDim;
