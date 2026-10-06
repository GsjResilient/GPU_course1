# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
import unittest
import hashlib,json,struct,subprocess,sys
from pathlib import Path
from compiler.tests.support import source_file
from compiler.lib.diagnostics import CompileError
try:
    from compiler.lib.driver import compile_file
    from compiler.lib.artifact import load_artifact
except ModuleNotFoundError:
    compile_file=None


class ArtifactTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(compile_file,'artifact compiler is not implemented')

    def test_hash_and_determinism(self):
        with source_file('NT_KERNEL void k(){}') as s:
            m=compile_file(s,'k',s.parent/'输出 directory'/'k')
            meta,code=load_artifact(m)
            self.assertEqual(meta['code_sha256'],hashlib.sha256(code).hexdigest())
            self.assertEqual(len(code),meta['instruction_count']*8)
            before=[p.read_bytes() for p in sorted(m.parent.iterdir())]
            compile_file(s,'k',s.parent/'输出 directory'/'k')
            self.assertEqual(before,[p.read_bytes() for p in sorted(m.parent.iterdir())])

    def test_failure_does_not_publish(self):
        with source_file('NT_KERNEL void k(){}') as s:
            m=compile_file(s,'k',s.parent/'k');before=m.read_bytes()
            s.write_text('NT_KERNEL void k(float* p){p[0]=1;}')
            with self.assertRaises(CompileError):compile_file(s,'k',s.parent/'k')
            self.assertEqual(m.read_bytes(),before)

    def test_corrupt_metadata(self):
        with source_file('NT_KERNEL void k(int* p){}') as s:
            m=compile_file(s,'k',s.parent/'k');original=m.read_text()
            for key,value in [('format','unknown'),('profile','unknown'),('regs_u32',True),
                              ('instruction_count',0),('code_file','../escape'),('entry',8),
                              ('code_sha256','0'*64),('param_size',3)]:
                with self.subTest(key=key):
                    meta=json.loads(original);meta[key]=value;m.write_text(json.dumps(meta))
                    with self.assertRaises(CompileError):load_artifact(m)
            m.write_text(original[:-1]+',"entry":0}')
            with self.assertRaises(CompileError):load_artifact(m)

    def test_truncated_and_missing(self):
        with source_file('NT_KERNEL void k(){}') as s:
            m=compile_file(s,'k',s.parent/'k')
            (s.parent/'k.ntbin').write_bytes(b'123')
            with self.assertRaises(CompileError):load_artifact(m)
            with self.assertRaises(CompileError):load_artifact(s.parent/'missing.json')

    def test_cli(self):
        with source_file('NT_KERNEL void k(){}') as s:
            cmd=[sys.executable,'compiler/nutcc.py',str(s),'--kernel','k','--out',str(s.parent/'out')]
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertTrue((s.parent/'out.json').is_file())
            s.write_text('NT_KERNEL void k(){while(true){}}')
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('未更新',r.stderr)
