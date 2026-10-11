# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
"""Regressions for predicate reuse, especially writes to the cached guard."""
import hashlib
import struct
import subprocess
import sys
import unittest
from pathlib import Path

from compiler.lib.artifact import load_artifact,write_artifact
from compiler.lib.backend import generate
from compiler.lib.build import REPO
from compiler.lib.diagnostics import CompileError,SourceLoc
from compiler.lib.driver import compile_file
from compiler.lib.frontend import parse_source
from compiler.lib.ir import IRInst,Program,VReg
from compiler.lib.lower import lower_kernel
from compiler.lib.runtime import BufferRef,run_artifact
from compiler.lib.validate import validate_kernel
from compiler.tests.support import source_file


def is_p0_setup(word):
    return word>>57==0x40 and (word>>40)&255==0


class OptimizationTests(unittest.TestCase):
    def vector_program(self):
        return lower_kernel(validate_kernel(parse_source(REPO/'compiler/examples/vector_add.cpp'),'vector_add'))

    def test_default_reuses_predicate_in_vector_add(self):
        machine=generate(self.vector_program())
        self.assertEqual(len(machine.words),46)
        self.assertEqual(sum(map(is_p0_setup,machine.words)),1)

    def test_o0_preserves_previous_machine_code(self):
        baseline=generate(self.vector_program(),opt_level=0)
        code=b''.join(struct.pack('<Q',w) for w in baseline.words)
        # Captured from released 88ec027, before adding any optimizer.
        self.assertEqual(hashlib.sha256(code).hexdigest(),'ca936172617669d998af0c24d5384ef8a5ecf79a08d0f1d82f18731cf9b88876')
        optimized=generate(self.vector_program(),opt_level=1)
        self.assertEqual([w for w in optimized.words if not is_p0_setup(w)],
                         [w for w in baseline.words if not is_p0_setup(w)])
        self.assertEqual((optimized.params,optimized.param_size,optimized.regs_u32),
                         (baseline.params,baseline.param_size,baseline.regs_u32))

    def test_cli_modes_default_and_invalid_options(self):
        source=REPO/'compiler/examples/vector_add.cpp'
        with source_file('unused') as s:
            for name,flags,want in [('o0',['-O0'],66),('o1',['-O1'],46),('default',[],46)]:
                prefix=s.parent/name
                r=subprocess.run([sys.executable,str(REPO/'compiler/nutcc.py'),str(source),'--kernel','vector_add','--out',str(prefix),*flags],capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                meta,_=load_artifact(prefix.with_suffix('.json'))
                self.assertEqual(meta['instruction_count'],want)
            previous={p.name:p.read_bytes() for p in s.parent.glob('o0.*')}
            for flags in (['-O2'],['-O0','-O1']):
                r=subprocess.run([sys.executable,str(REPO/'compiler/nutcc.py'),str(source),'--kernel','vector_add','--out',str(s.parent/'o0'),*flags],capture_output=True,text=True)
                self.assertEqual(r.returncode,2,r.stderr)
                self.assertEqual(previous,{p.name:p.read_bytes() for p in s.parent.glob('o0.*')})

    def test_invalid_api_level_is_located_compile_error(self):
        with source_file('NT_KERNEL void k(){}') as s:
            p=lower_kernel(validate_kernel(parse_source(s),'k'))
            for level in (-1,2,True,'1',None):
                for call in (lambda:generate(p,opt_level=level),lambda:compile_file(s,'k',s.parent/'bad',opt_level=level)):
                    with self.subTest(level=level),self.assertRaisesRegex(CompileError,'E_OPT_LEVEL') as e:call()
                    self.assertIn('kernel.cpp:',str(e.exception))

    def synthetic(self,body):
        loc=SourceLoc('predicate-fixture',1,1)
        self.g,self.h,self.a,self.b,self.c=[VReg(i,'u32') for i in range(5)]
        def ins(op,dst,args=(),imm=None,guard=None):return IRInst(op,dst,tuple(args),imm,guard,loc)
        instructions=[ins('const',self.g,imm=1),ins('const',self.h,imm=0),ins('const',self.a,imm=7)]
        return Program('k',[],0,instructions+body(ins)+[ins('exit',None)])

    def assert_setups(self,program,expected):
        machine=generate(program)
        self.assertEqual(sum(map(is_p0_setup,machine.words)),expected)

    def test_same_guard_reused(self):
        p=self.synthetic(lambda i:[i('copy',self.b,[self.a],guard=self.g),i('copy',self.c,[self.b],guard=self.g)])
        self.assert_setups(p,1)

    def test_guard_switch_invalidates_cache(self):
        p=self.synthetic(lambda i:[i('copy',self.b,[self.a],guard=self.g),i('copy',self.c,[self.a],guard=self.h),i('copy',self.b,[self.a],guard=self.g)])
        self.assert_setups(p,3)

    def test_unguarded_gap_invalidates_cache(self):
        p=self.synthetic(lambda i:[i('copy',self.b,[self.a],guard=self.g),i('copy',self.c,[self.a]),i('copy',self.b,[self.a],guard=self.g)])
        self.assert_setups(p,2)

    def test_guard_destination_write_invalidates_cache(self):
        p=self.synthetic(lambda i:[i('copy',self.b,[self.a],guard=self.g),i('copy',self.g,[self.h],guard=self.g),i('copy',self.c,[self.a],guard=self.g)])
        self.assert_setups(p,2)

    def test_p1_comparison_does_not_clobber_p0(self):
        p=self.synthetic(lambda i:[i('copy',self.b,[self.a],guard=self.g),i('cmp_ne',self.c,[self.a,self.h],guard=self.g),i('copy',self.b,[self.a],guard=self.g)])
        self.assert_setups(p,1)

    def test_guard_self_write_blocks_later_store_on_both_simulators(self):
        with source_file('NT_KERNEL void k(unsigned int* out){}') as s:
            unit=parse_source(s);p=lower_kernel(validate_kernel(unit,'k'));loc=p.instructions[0].loc
            pointer=p.instructions[0].dst
            zero,one,tid,g,value=[VReg(i,'u32') for i in range(1,6)]
            def ins(op,dst,args=(),imm=None,guard=None):return IRInst(op,dst,tuple(args),imm,guard,loc)
            p.instructions=p.instructions[:1]+[ins('const',zero,imm=0),ins('const',one,imm=1),
                ins('sr',tid,imm=0),ins('cmp_lt',g,[tid,one]),ins('const',value,imm=7),
                ins('copy',g,[zero],guard=g),ins('store',None,[pointer,value],guard=g),ins('exit',None)]
            for level in (0,1):
                manifest=write_artifact(generate(p,opt_level=level),unit,s.parent/f'o{level}')
                for backend in ('func','cycle'):
                    result=run_artifact(manifest,backend,{'out':struct.pack('<I',11)},[BufferRef('out')],
                                        (1,1,1),(32,1,1),s.parent/f'o{level}-{backend}')
                    self.assertEqual(result.buffers['out'],struct.pack('<I',11))
