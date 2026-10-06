# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
import unittest
from compiler.tests.support import source_file
from compiler.lib.frontend import parse_source
from compiler.lib.validate import validate_kernel
from compiler.lib.diagnostics import CompileError
try:
    from compiler.lib.lower import lower_kernel, f32_literal_bits
except ModuleNotFoundError:
    lower_kernel = None


def lower_text(text, name='k'):
    with source_file(text) as p:
        return lower_kernel(validate_kernel(parse_source(p), name))


class LowerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(lower_kernel, 'IR lowering is not implemented')

    def test_parameter_layout(self):
        p = lower_text('NT_KERNEL void k(unsigned int n,float a,float* x){}')
        self.assertEqual([a['offset'] for a in p.params], [0, 8, 16])
        self.assertEqual(p.param_size, 24)

    def test_thread_index_and_store(self):
        p = lower_text('NT_KERNEL void k(unsigned int* out){'
                       'unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;out[i]=i;}')
        self.assertEqual([x.imm for x in p.instructions if x.op == 'sr'], [8,11,0])
        self.assertEqual(sum(x.op == 'store' for x in p.instructions), 1)

    def test_literal_rounding(self):
        for text, bits in [('0.1',0x3dcccccd),('-0.0',0x80000000),
                           ('1.000000059604644775390625',0x3f800000),
                           ('1.000000059604644775390626',0x3f800001),
                           ('0x1p-149',1)]:
            self.assertEqual(f32_literal_bits(text), bits)

    def test_precedence_and_shadowing(self):
        p = lower_text('NT_KERNEL void k(int* out){int x=1+2*3;{int x=4;out[0]=x;}out[1]=x;}')
        ops = [i.op for i in p.instructions]
        self.assertLess(ops.index('mul'), ops.index('add'))
        self.assertEqual(ops.count('store'),2)

    def test_uninitialized(self):
        with self.assertRaisesRegex(CompileError,'E_UNINITIALIZED'):
            lower_text('NT_KERNEL void k(int* p){int x;p[0]=x;}')

    def test_memory_reads_remain_ordered(self):
        p = lower_text('NT_KERNEL void k(int* p){p[0]=4;p[1]=p[0]+1;}')
        self.assertEqual([i.op for i in p.instructions if i.op in ('load','store')], ['store','load','store'])

    def test_negative_zero_lowering(self):
        p = lower_text('NT_KERNEL void k(float* p){p[0]=-0.0f;}')
        self.assertTrue(any(i.imm == 0x80000000 for i in p.instructions))
