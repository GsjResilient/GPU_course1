"""Regression coverage for reusable sessions, modules, writes and launch ABI."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ntcc import compile_file
from ntruntime import Session
from ntdb import Debugger


class SessionReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='reuse with space ')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def module(self,name,amount):
        source=self.root/(name+'.ntcu')
        source.write_text('__global__ void same(uint* output) {\n output[threadIdx.x] += '+str(amount)+'u;\n}\n')
        return compile_file(source,self.root/name)

    def test_repeat_module_selection_in_place_and_dynamic_shared_both_backends(self):
        first=self.module('first',1)
        second=self.module('second',7)
        for backend in ('func','cycle'):
            with self.subTest(backend=backend),Session(first,backend) as session:
                other=session.load_module(second)
                self.assertEqual(session.load_module(second),other)
                pointer=session.upload(struct.pack('<3I',10,20,30))
                for module in (0,other,0):
                    session.launch(1,3,{'output':pointer},dynamic_smem=512,module=module)
                    session.synchronize()
                self.assertEqual(struct.unpack('<3I',session.download(pointer,12)),(19,29,39))
                session.write(pointer,struct.pack('<3I',0,1,2))
                session.launch(1,3,{'output':pointer},module=other)
                session.synchronize()
                self.assertEqual(struct.unpack('<3I',session.download(pointer,12)),(7,8,9))
                with self.assertRaises(RuntimeError):
                    session.write(pointer+12,b'abcd')
                with self.assertRaises(RuntimeError):
                    session.launch(1,3,{'output':pointer},dynamic_smem=1000000)
                session.launch(1,3,{'output':pointer})
                session.synchronize()

    def test_pending_launch_module_load_selection_and_budget_resume(self):
        first=self.module('first',1)
        second=self.module('second',7)
        for backend in ('func','cycle'):
            with self.subTest(backend=backend),Session(first,backend) as session:
                pointer=session.upload(struct.pack('<I',0))
                session.launch(1,1,{'output':pointer})
                with self.assertRaises(RuntimeError):
                    session.launch(1,1,{'output':pointer})
                with self.assertRaises(RuntimeError):
                    session.load_module(second)
                with self.assertRaises(RuntimeError):
                    session.select_module(0)
                self.assertEqual(session.command('run 1')['reason'],'budget')
                session.synchronize()
                self.assertEqual(struct.unpack('<I',session.download(pointer,4)),(1,))
                session.load_module(second)

    def test_debugger_after_module_switch_and_relaunch(self):
        first=self.module('first',1)
        second=self.module('second',7)
        with Session(first) as session:
            other=session.load_module(second)
            pointer=session.upload(struct.pack('<I',0))
            for module,value in ((0,1),(other,8),(0,9)):
                session.launch(1,1,{'output':pointer},module=module)
                debugger=Debugger(session)
                debugger.breakpoint(2)
                self.assertEqual(debugger.run()['reason'],'breakpoint')
                debugger.command('delete 2')
                self.assertEqual(debugger.run()['reason'],'completed')
                self.assertEqual(struct.unpack('<I',session.download(pointer,4)),(value,))


if __name__=='__main__':
    unittest.main()
