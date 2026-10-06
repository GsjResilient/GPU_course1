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
import unittest
import subprocess,sys
from compiler.tests.support import source_file

try:
    from compiler.lib.frontend import parse_source
    from compiler.lib.validate import validate_kernel
    from compiler.lib.diagnostics import CompileError
except ModuleNotFoundError:
    parse_source = None


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(parse_source, 'typed C++ frontend is not implemented')

    def check(self, text, name='k'):
        with source_file(text) as p:
            return validate_kernel(parse_source(p), name)

    def test_typed_parameters(self):
        k = self.check('NT_KERNEL void k(const float* a, float* b, unsigned int n) {}')
        self.assertEqual([p['kind'] for p in k.parameters], ['ptr_f32', 'ptr_f32', 'u32'])
        self.assertTrue(k.parameters[0]['readonly'])

    def test_float_literal(self):
        self.check('NT_KERNEL void k(float* p) {p[0]=1.0f;}')

    def test_numeric_conversions_rejected(self):
        for expression in ('1', '1.0'):
            with self.subTest(expression=expression), self.assertRaisesRegex(CompileError, 'E_CONVERSION'):
                self.check('NT_KERNEL void k(float* p) {p[0]=' + expression + ';}')
        with self.assertRaisesRegex(CompileError, 'E_CONVERSION'):
            self.check('NT_KERNEL void k(unsigned int* p) {int a=1; p[0]=a;}')

    def test_global_constexpr_initializers_obey_subset(self):
        for declaration,kind in [('constexpr float C=1;','float'),
                                 ('constexpr float C=1.0;','float'),
                                 ('constexpr unsigned int C=-1;','unsigned int'),
                                 ('constexpr int C=4/2;','int'),
                                 ('constexpr int C=(int)1.0f;','int')]:
            with self.subTest(declaration=declaration),self.assertRaises(CompileError):
                self.check(declaration+'NT_KERNEL void k('+kind+'* p){p[0]=C;}')
        with source_file('#include "constants.h"\nNT_KERNEL void k(float* p){p[0]=C;}') as s:
            (s.parent/'constants.h').write_text('constexpr float C=1;')
            with self.assertRaisesRegex(CompileError,'E_CONVERSION'):
                validate_kernel(parse_source(s),'k')

    def test_overflow_literal_is_located_compile_error(self):
        with source_file('NT_KERNEL void k(float* p){p[0]=1e40f;}') as s:
            result=subprocess.run([sys.executable,'compiler/nutcc.py',str(s),'--kernel','k','--out',str(s.parent/'out')],capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('E_LITERAL',result.stderr)
            self.assertIn('kernel.cpp:1:',result.stderr)

    def test_missing_entry(self):
        with self.assertRaisesRegex(CompileError, 'E_KERNEL'):
            self.check('NT_KERNEL void other() {}')

    def test_clang_error_has_location(self):
        with self.assertRaises(CompileError) as e:
            self.check('NT_KERNEL void k() { this is invalid; }')
        self.assertIn('kernel.cpp:', str(e.exception))

    def test_rejection_matrix(self):
        bodies = ['while(true) {}', 'do {} while(false);', 'goto lab; lab:;',
                  'switch(1) {case 1: break;}', 'return; int a=1;',
                  'int* a=new int;', 'throw 1;', 'volatile int a=1;',
                  'static int a=1;', 'long a=1;', 'double a=1.0;',
                  'int a=1/2;', 'int a=2%1;', 'asm("nop");',
                  'for(int k=0;k<3;++k) {break;}', 'for(int k=0;k<3;++k) {continue;}']
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(CompileError):
                self.check('NT_KERNEL void k() {' + body + '}')

    def test_calls_and_records_rejected(self):
        for text in ['int f(); NT_KERNEL void k(){int a=f();}',
                     'struct X {int a;}; NT_KERNEL void k(){X x;}',
                     'template<class T>T f(T a){return a;} NT_KERNEL void k(){f(1);}',
                     'int mutable_global; NT_KERNEL void k(){}']:
            with self.subTest(text=text), self.assertRaises(CompileError):
                self.check(text)

    def test_const_write_rejected(self):
        with self.assertRaises(CompileError):
            self.check('NT_KERNEL void k(const int* p){p[0]=1;}')

    def test_supported_conditions_and_loop(self):
        self.check('NT_KERNEL void k(int* p, int n) {int a=0; '
                   'for(int j=0;j<3;++j) {if(j<n && n>0) a+=j; else a-=j;} p[0]=a;}')


if __name__ == '__main__':
    unittest.main()
