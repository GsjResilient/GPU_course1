#!/usr/bin/env python3
# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
"""Load a supplied artifact and check exact, representable FP32 additions."""
import argparse,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from compiler.lib.artifact import load_artifact
from compiler.lib.runtime import BufferRef,run_artifact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--backend',choices=('func','cycle'),required=True)
    p.add_argument('--n',type=int,default=65)
    a=p.parse_args()
    try:
        if not 1<=a.n<=1000000:raise ValueError('n must be 1..1000000')
        meta,_=load_artifact(a.artifact)
        if meta['kernel']!='vector_add' or [x['kind'] for x in meta['params']]!=['ptr_f32']*3+['u32']:
            raise ValueError('expected vector_add artifact with (float*, float*, float*, u32) parameters')
        left=[float(i%1000)/4 for i in range(a.n)];right=[float(13-i%97)/4 for i in range(a.n)]
        def pack(values):return struct.pack('<'+'f'*a.n,*values)
        result=run_artifact(a.artifact,a.backend,dict(a=pack(left),b=pack(right),out=bytes(4*a.n)),
                            [BufferRef('a'),BufferRef('b'),BufferRef('out'),a.n],((a.n+31)//32,1,1),(32,1,1),
                            a.artifact.parent/('run-'+a.backend))
        if result.buffers['out']!=pack([x+y for x,y in zip(left,right)]):raise ValueError('CPU reference mismatch')
        print(f'PASS backend={a.backend} n={a.n} completed=true instructions={result.instructions} cycles={result.cycles}')
        return 0
    except Exception as e:print('FAIL: '+str(e),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
