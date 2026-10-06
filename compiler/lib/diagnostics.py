# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Author: Di Zhao, Trae CN
# Contact: zhaodi@ucas.ac.cn
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers,
#             General-Purpose Graphics Processor Architecture,
#             Morgan & Claypool, 2018
# September 2026
# Improved by: 明同辉（26秋），王一锴（26秋），崔文君（26秋）
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
