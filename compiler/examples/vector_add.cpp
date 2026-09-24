#include "nutcc.h"

NT_KERNEL void vector_add(const float* a, const float* b,
                          float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
