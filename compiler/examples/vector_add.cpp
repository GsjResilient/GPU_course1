// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Author: Di Zhao, Trae CN
// Contact: zhaodi@ucas.ac.cn
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers,
//             General-Purpose Graphics Processor Architecture,
//             Morgan & Claypool, 2018
// September 2026
// Improved by: 明同辉（26秋），王一锴（26秋），崔文君（26秋）
// =============================================================================
#include "nutcc.h"

NT_KERNEL void vector_add(const float* a, const float* b,
                          float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
