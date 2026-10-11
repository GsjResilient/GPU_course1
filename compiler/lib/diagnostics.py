# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    column: int


class CompileError(Exception):
    def __init__(self, code, message, loc):
        self.code, self.message, self.loc = code, message, loc
        super().__init__(f'{loc.file}:{loc.line}:{loc.column}: {code}: {message}')
