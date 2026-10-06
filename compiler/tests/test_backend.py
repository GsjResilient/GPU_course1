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
import subprocess
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from compiler.tests.test_lower import lower_text
from compiler.lib.ir import Program,VReg,IRInst
from compiler.lib.diagnostics import SourceLoc,CompileError
from compiler.lib.toolchain import native_command
try:
    from compiler.lib.ntas import encode_i,encode_r,encode_mi
    from compiler.lib.backend import generate
    from compiler.lib.regalloc import allocate
except ModuleNotFoundError:
    generate=None


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(generate,'NTAS backend is not implemented')

    def test_known_words(self):
        self.assertEqual(encode_i(0x02,2,0x12345678),0x0400021234567800)
        self.assertEqual(encode_r(0x40,1,3,4,x=0x15,y=0x83),0x8000010304158300)
        self.assertEqual(encode_r(0x09,7,8,9,y=0x0a),0x1200070809000A00)
        self.assertEqual(encode_mi(0x61,5,6,0,0,7,guard=(2,True)),0xC3A0050600000007)

    def test_bad_fields(self):
        for call in (lambda:encode_r(256,0),lambda:encode_r(1,256),
                     lambda:encode_i(2,1,1<<32),lambda:encode_r(1,2,guard=(7,True)),
                     lambda:encode_mi(0x60,0,0,0,128,0)):
            with self.assertRaises(ValueError):call()

    def test_generated_program_has_exit_and_guards(self):
        p=generate(lower_text('NT_KERNEL void k(int* p,int n){if(n>0){p[0]=n;}else{p[1]=2;}}'))
        self.assertEqual(p.words[-1],0xae00000000000000)
        self.assertEqual(sum((w>>57)==0x57 for w in p.words),1)
        self.assertTrue(any((w>>56)&1 for w in p.words))

    def test_no_fusion(self):
        p=generate(lower_text('NT_KERNEL void k(float* p,float a,float b){p[0]=a*b+1.0f;}'))
        ops=[w>>57 for w in p.words]
        self.assertIn(0x32,ops);self.assertIn(0x30,ops);self.assertNotIn(0x33,ops)

    def test_live_ranges(self):
        loc=SourceLoc('test',1,1)
        a,b,c=VReg(0,'i32'),VReg(1,'i32'),VReg(2,'i32')
        p=Program('k',[],0,[IRInst('const',a,(),1,None,loc),IRInst('copy',b,(a,),None,None,loc),
                             IRInst('const',c,(),2,None,loc),IRInst('add',b,(b,c),None,None,loc)])
        alloc=allocate(p)
        self.assertEqual(alloc.registers[a.id],alloc.registers[c.id])
        self.assertNotEqual(alloc.registers[b.id],alloc.registers[c.id])
        with self.assertRaisesRegex(CompileError,'E_REGISTERS'):allocate(p,limit=3)

    def test_code_budget(self):
        # IR fits, but guards expand to additional machine instructions.
        with self.assertRaisesRegex(CompileError,'E_CODE_SIZE'):
            generate(lower_text('NT_KERNEL void k(int* p,int n){for(int j=0;j<256;++j){'
                                'if(n>0){p[j]=n*n+n*n+n*n;}}}'))

    def test_original_decoder_accepts_code_rejects_corruption(self):
        source=Path('compiler/tests/decode_check.cpp')
        self.assertTrue(source.is_file(),'original decoder checker is missing')
        with TemporaryDirectory() as td:
            root=Path(td); checker=root/'decode'
            subprocess.run(native_command()+['-std=c++17','-I','NutShellGPU_sim',str(source),'-o',str(checker)],check=True,capture_output=True)
            p=generate(lower_text('NT_KERNEL void k(float* p,float n){if(n>0.0f){p[threadIdx.x]=-n;}}'))
            code=root/'code.bin';code.write_bytes(b''.join(struct.pack('<Q',w) for w in p.words))
            self.assertEqual(subprocess.run([str(checker),str(code),str(p.regs_u32)],capture_output=True).returncode,0)
            code.write_bytes(bytes(8))
            self.assertNotEqual(subprocess.run([str(checker),str(code),'4'],capture_output=True).returncode,0)
