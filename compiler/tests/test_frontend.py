import unittest
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
