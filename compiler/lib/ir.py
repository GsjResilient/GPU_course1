# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .diagnostics import SourceLoc


@dataclass(frozen=True)
class VReg:
    id: int
    kind: str


@dataclass
class IRInst:
    op: str
    dst: Optional[VReg]
    args: Tuple[VReg, ...]
    imm: Optional[int]
    guard: Optional[VReg]
    loc: SourceLoc


@dataclass
class Program:
    name: str
    params: list
    param_size: int
    instructions: List[IRInst]
