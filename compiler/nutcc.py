#!/usr/bin/env python3
"""Compile a restricted C++ kernel to NutShellGPU machine instructions."""
import argparse
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compiler.lib.driver import compile_file
from compiler.lib.artifact import VERSION
from compiler.lib.diagnostics import CompileError


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',action='version',version='nutcc '+VERSION)
    parser.add_argument('source',type=Path)
    parser.add_argument('--kernel',required=True)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--clang',default='clang++')
    optimization=parser.add_mutually_exclusive_group()
    optimization.add_argument('-O0',dest='opt_level',action='store_const',const=0,help='emit baseline code without predicate reuse')
    optimization.add_argument('-O1',dest='opt_level',action='store_const',const=1,help='reuse unchanged adjacent predicates (default)')
    parser.set_defaults(opt_level=1)
    args=parser.parse_args(argv)
    try:
        manifest=compile_file(args.source,args.kernel,args.out,args.clang,opt_level=args.opt_level)
        print('Compiled:',manifest)
        return 0
    except CompileError as e:
        print(str(e)+'\n编译失败，未更新现有产物；请勿将旧文件视为本次成功结果。',file=sys.stderr)
        return 2
    except Exception as e:
        print(f'internal compiler error: {type(e).__name__}: {e}',file=sys.stderr)
        return 3


if __name__=='__main__':
    sys.exit(main())
