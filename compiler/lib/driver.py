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
