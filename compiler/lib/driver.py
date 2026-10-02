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
