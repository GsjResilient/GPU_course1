# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
import unittest
import subprocess,sys
from compiler.tests.support import source_file
from compiler.tests.test_lower import lower_text
from compiler.lib.diagnostics import CompileError


class ControlTests(unittest.TestCase):
    def test_nonemitting_expansion_has_budget(self):
        source='NT_KERNEL void k(){'
        source+=''.join('for(int i'+str(i)+'=0;i'+str(i)+'<256;++i'+str(i)+'){' for i in range(5))
        source+='int unused;'+'}'*6
        with source_file(source) as p:
            result=subprocess.run([sys.executable,'compiler/nutcc.py',str(p),'--kernel','k','--out',str(p.parent/'out')],capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('E_EXPANSION_LIMIT',result.stderr)

    def test_condition_snapshot(self):
        p = lower_text('NT_KERNEL void k(int* out){int x=0;if(x==0){x=1;}else{x=2;}out[0]=x;}')
        self.assertFalse(any(i.op in ('bra','ssy') for i in p.instructions))
        self.assertEqual(sum(i.op == 'exit' for i in p.instructions),1)
        guarded = [i for i in p.instructions if i.op=='copy' and i.guard is not None]
        self.assertEqual(len(guarded),2)
        self.assertNotEqual(guarded[0].guard,guarded[1].guard)

    def test_short_circuit_guards_reads(self):
        p = lower_text('NT_KERNEL void k(int* p,int n){'
                       'if(n>0 && p[0]>1){p[1]=4;} if(n==0 || p[2]>0){p[3]=5;}}')
        for i in p.instructions:
            if i.op in ('load','store'):
                self.assertIsNotNone(i.guard)

    def test_constant_branches(self):
        p = lower_text('NT_KERNEL void k(int* p){if(false){p[0]=1;}if(true){p[0]=2;}}')
        stores=[i for i in p.instructions if i.op=='store']
        self.assertEqual(len(stores),1)
        self.assertIsNone(stores[0].guard)

    def test_definite_assignment(self):
        lower_text('NT_KERNEL void k(int* p,int n){int x;if(n>0){x=1;}else{x=2;}p[0]=x;}')
        with self.assertRaisesRegex(CompileError,'E_UNINITIALIZED'):
            lower_text('NT_KERNEL void k(int* p,int n){int x;if(n>0){x=1;}p[0]=x;}')

    def test_loop_counts(self):
        for count in (0,1,3,256):
            p=lower_text('NT_KERNEL void k(int* p){for(int j=0;j<'+str(count)+';++j){p[j]=j;}}')
            self.assertEqual(sum(i.op=='store' for i in p.instructions),count)

    def test_loop_rejections(self):
        for body in ('for(int j=0;j<n;++j){}','for(int j=0;j<257;++j){}',
                     'for(int j=0;j<3;++j){j=2;}',
                     'for(int j=0;j<256;++j){for(int k=0;k<256;++k){p[0]=k;}}'):
            with self.subTest(body=body),self.assertRaises(CompileError):
                lower_text('NT_KERNEL void k(int* p,int n){'+body+'}')

    def test_nested_conditions_and_loops(self):
        p=lower_text('NT_KERNEL void k(int* p,int n){for(int j=0;j<2;++j){'
                     'for(int k=0;k<3;++k){if(n>0){if(k<n){p[j*3+k]=k;}}}}}')
        self.assertEqual(sum(i.op=='store' for i in p.instructions),6)

    def test_unreachable_unsupported_loop(self):
        with self.assertRaisesRegex(CompileError,'E_LOOP'):
            lower_text('NT_KERNEL void k(int n){if(false){for(int j=0;j<n;++j){}}}')
