// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.4
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
// September 2026
// =============================================================================
#pragma once
#define NT_KERNEL extern "C"
static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(void*) == 8,
              "NutCC requires 32-bit int/float and 64-bit frontend pointers");
struct nutcc_dim3 { const unsigned int x, y, z; };
extern const nutcc_dim3 threadIdx, blockIdx, blockDim, gridDim;
