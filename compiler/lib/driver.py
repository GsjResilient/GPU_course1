# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
from .frontend import parse_source
from .validate import validate_kernel
from .lower import lower_kernel
from .backend import generate
from .artifact import write_artifact


def compile_file(source,kernel,prefix,clang=None,opt_level=1):
    unit=parse_source(source,clang)
    ast=validate_kernel(unit,kernel)
    program=lower_kernel(ast)
    machine=generate(program,opt_level=opt_level)
    return write_artifact(machine,unit,prefix)
