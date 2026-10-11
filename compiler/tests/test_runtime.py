# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
import unittest
import json,struct,hashlib,subprocess,sys
from pathlib import Path
from compiler.tests.support import source_file
from compiler.lib.driver import compile_file
try:
    from compiler.lib.runtime import run_artifact,BufferRef,RunError,check_span
    from compiler.lib.build import build_tools
except ModuleNotFoundError:
    run_artifact=None


class RuntimeTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(run_artifact,'simulator bridge is not implemented')

    def run_source(self,text,buffers,args,block=(32,1,1),**kwargs):
        with source_file(text) as s:
            results=[]
            for level in (0,1):
                mode=s.parent/f'O{level}'
                m=compile_file(s,'k',mode/'k',opt_level=level)
                results.extend(run_artifact(m,backend,buffers,args,(1,1,1),block,mode/backend,**kwargs)
                               for backend in ('func','cycle'))
            return results

    def test_run_constant(self):
        results=self.run_source('NT_KERNEL void k(unsigned int* p){p[threadIdx.x]=7U;}',
                                {'p':bytes(128)},[BufferRef('p')])
        for r in results:
            self.assertTrue(r.completed)
            self.assertEqual(struct.unpack('<32I',r.buffers['p']),(7,)*32)

    def test_guard_and_tail(self):
        results=self.run_source('NT_KERNEL void k(unsigned int* p,unsigned int n){'
                                'unsigned int i=threadIdx.x;if(i<n){p[i]=i+2U;}}',
                                {'p':bytes(12)},[BufferRef('p'),3])
        for r in results:self.assertEqual(struct.unpack('<3I',r.buffers['p']),(2,3,4))

    def test_aliasing(self):
        results=self.run_source('NT_KERNEL void k(int* p,int* q){p[0]=4;q[0]+=1;}',
                                {'p':bytes(4)},[BufferRef('p'),BufferRef('p')],(1,1,1))
        for r in results:self.assertEqual(r.buffers['p'],struct.pack('<i',5))

    def test_guard_sentinels(self):
        with self.assertRaisesRegex(RunError,'sentinel'):
            self.run_source('NT_KERNEL void k(int* p){p[1]=2;}',{'p':bytes(4)},[BufferRef('p')],(1,1,1))

    def test_contract_rejections(self):
        with source_file('NT_KERNEL void k(int* p){}') as s:
            m=compile_file(s,'k',s.parent/'k')
            for args,block in [([], (1,1,1)),([BufferRef('missing')],(1,1,1)),
                               ([BufferRef('p',8)],(1,1,1)),([BufferRef('p')],(0,1,1)),
                               ([BufferRef('p')],(1025,1,1))]:
                with self.subTest(args=args,block=block),self.assertRaises(RunError):
                    run_artifact(m,'func',{'p':bytes(4)},args,(1,1,1),block,s.parent/'run')
        with self.assertRaises(RunError):check_span(0xfffffff0,32)
        check_span(0xfffffff0,16)

    def test_timeout_and_step_budget(self):
        with source_file('NT_KERNEL void k(int* p){p[threadIdx.x]=2;}') as s:
            m=compile_file(s,'k',s.parent/'k')
            for backend in ('func','cycle'):
                with self.assertRaises(RunError):
                    run_artifact(m,backend,{'p':bytes(128)},[BufferRef('p')],(1,1,1),(32,1,1),s.parent/backend,max_steps=1)
            with self.assertRaisesRegex(RunError,'timeout'):
                run_artifact(m,'func',{'p':bytes(128)},[BufferRef('p')],(1,1,1),(32,1,1),s.parent/'timeout',timeout_s=0.000001)

    def test_binary_protocol_rejection(self):
        tools=build_tools(Path('build/nutcc-tools'))
        with source_file('bad') as s:
            request=s.parent/'bad.bin';request.write_bytes(b'NTRUN01\0')
            r=subprocess.run([str(tools['bridge']),'func',str(request),str(s.parent/'out.bin')],capture_output=True)
            self.assertNotEqual(r.returncode,0)

    def test_runtime_cli(self):
        with source_file('NT_KERNEL void k(unsigned int* p){p[threadIdx.x]=7U;}') as s:
            m=compile_file(s,'k',s.parent/'k');(s.parent/'input.bin').write_bytes(bytes(4))
            launch=s.parent/'launch.json'
            launch.write_text(json.dumps({'format':'nutcc-launch-v1','grid':[1,1,1],'block':[1,1,1],
                                          'buffers':{'p':'input.bin'},'args':[{'buffer':'p','offset':0}]}))
            r=subprocess.run([sys.executable,'compiler/run.py','--artifact',str(m),'--backend','func',
                              '--launch',str(launch),'--out',str(s.parent/'out')],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual((s.parent/'out'/'p.bin').read_bytes(),struct.pack('<I',7))
