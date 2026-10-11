# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
EXAMPLES=PROJECT/'examples/toolchain'
sys.path.insert(0, str(ROOT))
from ntcc import Compiler, CompileError, compile_file
from ntruntime import Session, prepare, collect
from ntdb import Debugger
from ntasm import render, assemble


class ToolchainTests(unittest.TestCase):
    executed_opcodes=set()
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def module(self, source):
        path=self.dir/'test.ntcu';path.write_text(source,encoding='utf-8')
        module=compile_file(path,self.dir/'test')
        meta=json.loads(module.read_text(encoding='utf-8'))
        self.assertEqual(assemble(render(meta)),module.with_suffix('').with_suffix('.ntas').read_bytes())
        return module

    def execute(self, source, args, grid=1, block=32, backend='func'):
        with Session(self.module(source),backend) as s:
            buffers=prepare(s,{'grid':grid,'block':block,'arguments':args})
            event=s.synchronize()
            self.executed_opcodes.update(ins['op'] for ins in s.meta['instructions'])
            return collect(s,buffers),event

    def test_saxpy_multicta_tail_both_backends(self):
        source=(EXAMPLES/'saxpy.ntcu').read_text()
        n=197; x=[float(i) for i in range(n)]; y=[float(-i) for i in range(n)]
        for backend in ('func','cycle'):
            out,event=self.execute(source,{'x':x,'y':y,'a':2.5,'n':n},grid=7,backend=backend)
            self.assertEqual(out['y'],[1.5*i for i in range(n)])
            self.assertEqual(out['x'],x)
            self.assertEqual(event['reason'],'completed')
            if backend=='cycle':self.assertGreater(event['cycles'],0)

    def test_divergent_if_else_and_reconvergence(self):
        src='''__global__ void choose(int* out) {
 uint i = threadIdx.x;
 int value = 0;
 if (i < 13u) { value = -3; } else { value = 7; }
 out[i] = value + 2;
}'''
        for b in ('func','cycle'):
            out,_=self.execute(src,{'out':[0]*32},backend=b)
            self.assertEqual(out['out'],[-1]*13+[9]*19)

    def test_divergent_while(self):
        src='''__global__ void sums(uint* out) {
 uint i = threadIdx.x;
 uint j = 0u;
 uint sum = 0u;
 while (j < i) { sum = sum + j; j = j + 1u; }
 out[i] = sum;
}'''
        for b in ('func','cycle'):
            out,_=self.execute(src,{'out':[0]*32},backend=b)
            self.assertEqual(out['out'],[i*(i-1)//2 for i in range(32)])

    def test_nested_control_and_early_return(self):
        src='''__global__ void nested(uint* out) {
 uint i = threadIdx.x;
 if (i >= 25u) { return; }
 uint j = 0u;
 uint sum = 0u;
 while (j < 3u) {
   if (i < 10u) { sum = sum + 2u; } else { sum = sum + 1u; }
   j = j + 1u;
 }
 out[i] = sum;
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[99]*32},backend=backend)
            self.assertEqual(out['out'],[6]*10+[3]*15+[99]*7)

    def test_nested_lane_dependent_loops(self):
        src='''__global__ void nested(uint* out) {
 uint i = threadIdx.x;
 uint j = 0u;
 uint sum = 0u;
 while (j < i) {
   uint k = 0u;
   while (k < j) { sum = sum + 1u; k = k + 1u; }
   j = j + 1u;
 }
 out[i] = sum;
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[0]*8},block=8,backend=backend)
            self.assertEqual(out['out'],[i*(i-1)//2 for i in range(8)])

    def test_return_in_loop(self):
        src='''__global__ void returns(uint* out) {
 uint i = threadIdx.x;
 uint j = 0u;
 while (j < 4u) {
   if (j == i) { return; }
   j = j + 1u;
 }
 out[i] = 44u;
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[0]*8},block=8,backend=backend)
            self.assertEqual(out['out'],[0]*4+[44]*4)

    def test_signed_index_and_pointer_copy(self):
        src='''__global__ void pointer(int* out) {
 int* p = out;
 int i = 2;
 p[i - 1] = -17;
}'''
        out,_=self.execute(src,{'out':[0]*3},block=1)
        self.assertEqual(out['out'],[0,-17,0])

    def test_high_address_pointer_abi(self):
        # Synthetic high address deliberately bypasses host allocation API to
        # validate paired pointer parameter loads and 64-bit address arithmetic.
        src='__global__ void high(uint* p) { uint* q = p; q[1u] = 77u; }'
        with Session(self.module(src)) as s:
            s.launch(1,1,{'p':0x100010000})
            d=Debugger(s)
            for _ in range(3):d.run(True)
            self.assertEqual(d.variable('p')['value'],0x100010000)
            self.assertEqual(s.synchronize()['reason'],'completed')

    def test_breakpoint_inspection_and_resume(self):
        source=(EXAMPLES/'saxpy.ntcu').read_text()
        output_line=next(i for i,text in enumerate(source.splitlines(),1) if 'y[i] = value' in text)
        path=self.module(source)
        with Session(path) as s:
            buffers=prepare(s,json.loads((EXAMPLES/'saxpy.launch.json').read_text()))
            d=Debugger(s);d.breakpoint(output_line)
            self.assertEqual(d.run()['reason'],'breakpoint')
            self.assertEqual(d.variable('value')['value'],12.5)
            d.lane=4
            self.assertEqual(d.variable('i')['value'],4)
            self.assertEqual(d.variable('value')['value'],62.5)
            self.assertTrue(d.variable('value')['active'])
            self.assertEqual(collect(s,buffers)['y'],[10,20,30,40,50])
            d.command(f'delete {output_line}')
            self.assertEqual(d.run()['reason'],'completed')
            self.assertEqual(collect(s,buffers)['y'],[12.5,25,37.5,50,62.5])

    def test_breakpoint_resume_across_warps(self):
        src='''__global__ void fill(uint* out) {
 uint i = blockIdx.x * blockDim.x + threadIdx.x;
 out[i] = i + 100u;
}'''
        with Session(self.module(src)) as s:
            buffers=prepare(s,{'grid':3,'block':64,'arguments':{'out':[0]*192}})
            d=Debugger(s);d.breakpoint(3);seen=[]
            for _ in range(7):
                e=d.run()
                if e['reason']=='completed':break
                self.assertEqual(e['reason'],'breakpoint');seen.append((e['sm'],e['warp']))
            self.assertEqual(len(seen),6)
            self.assertEqual(len(set(seen)),6)
            self.assertEqual(collect(s,buffers)['out'],list(range(100,292)))

    def test_instruction_and_source_step(self):
        src='''__global__ void step(uint* out) {
 uint i = threadIdx.x;
 uint value = i + 7u;
 out[i] = value;
}'''
        with Session(self.module(src)) as s:
            buffers=prepare(s,{'grid':1,'block':32,'arguments':{'out':[0]*32}})
            d=Debugger(s)
            with self.assertRaises(ValueError):d.variable('value')
            d.breakpoint(3);d.run();d.command('delete 3')
            e=d.source_step();self.assertEqual(e['line'],4)
            self.assertEqual(d.variable('value')['value'],7)
            before=s.command('step 1')['instructions']
            after=s.command('step 1')['instructions']
            self.assertEqual(after-before,1)
            s.synchronize();self.assertEqual(collect(s,buffers)['out'],list(range(7,39)))

    def test_loop_breakpoint_rehits(self):
        src='''__global__ void loop(uint* out) {
 uint j = 0u;
 while (j < 3u) {
   out[0u] = j;
   j = j + 1u;
 }
}'''
        with Session(self.module(src)) as s:
            prepare(s,{'grid':1,'block':1,'arguments':{'out':[0]}})
            d=Debugger(s);d.breakpoint(4)
            for j in range(3):
                self.assertEqual(d.run()['reason'],'breakpoint')
                self.assertEqual(d.variable('j')['value'],j)
            self.assertEqual(d.run()['reason'],'completed')

    def test_pause_does_not_change_instruction_count_or_output(self):
        src=(EXAMPLES/'saxpy.ntcu').read_text()
        config={'grid':8,'block':64,'arguments':{'x':[1.]*512,'y':[2.]*512,'a':3.,'n':512}}
        module=self.module(src)
        with Session(module) as s:
            buffers=prepare(s,config);normal=s.synchronize();expected=collect(s,buffers)
        with Session(module) as s:
            buffers=prepare(s,config)
            for _ in range(97):self.assertEqual(s.command('step 1')['reason'],'step')
            resumed=s.synchronize()
            self.assertEqual(resumed['instructions'],normal['instructions'])
            self.assertEqual(collect(s,buffers),expected)

    def test_invalid_programs_fail_closed(self):
        programs=[
            '__global__ void k(float* x) { x[0] = 1; }',
            '__global__ void k(uint* x) { uint a = missing; }',
            '__global__ void k(uint* x) { x[0] = 4u / 2u; }',
            '__global__ void k(uint* x) { uint a = 4294967296; }',
            '__global__ void k(uint* x) { uint a = 1; uint a = 2; }',
            '__global__ void k(uint* x) { float a = (float)1; }',
            '__global__ void k(uint* x) { if (threadIdx.x) { __syncthreads(); } }',
            '__global__ void k(uint* x) { uint a = 1; } trailing',
        ]
        for source in programs:
            with self.subTest(source=source),self.assertRaises(CompileError):Compiler().compile(source,'bad.ntcu')

    def test_binary_metadata_mismatch_rejected(self):
        module=self.module('__global__ void k(uint* p) { p[0] = 2u; }')
        module.with_name('test.ntas').write_bytes(b'\x00'*8)
        with self.assertRaises(ValueError):Session(module)

    def test_bad_launch_focus_and_transfer_rejected(self):
        module=self.module('__global__ void k(uint* p) { p[0] = 2u; }')
        with Session(module) as s:
            with self.assertRaises(RuntimeError):s.download(0,4)
            p=s.malloc(4)
            with self.assertRaises(ValueError):s.launch(1,1,{'bad':p})
            with self.assertRaises(ValueError):s.launch(1,0,{'p':p})
            s.launch(1,1,{'p':p})
            with self.assertRaises(RuntimeError):s.command('state 0 0 32')
            with self.assertRaises(RuntimeError):s.command('break 3')

    def test_budget_pause_and_resume(self):
        module=self.module('__global__ void k(uint* p) { p[0] = 2u; }')
        with Session(module) as s:
            p=s.malloc(4);s.launch(1,1,{'p':p})
            self.assertEqual(s.command('run 1')['reason'],'budget')
            self.assertEqual(s.synchronize()['reason'],'completed')
            self.assertEqual(struct.unpack('<I',s.download(p,4))[0],2)

    def test_source_step_honors_breakpoint(self):
        src='''__global__ void k(uint* p) {
 uint i = threadIdx.x;
 p[i] = 2u;
}'''
        with Session(self.module(src)) as s:
            p=s.malloc(128);s.launch(1,32,{'p':p})
            d=Debugger(s);d.breakpoint(2)
            event=d.source_step()
            self.assertEqual(event['reason'],'breakpoint')
            self.assertEqual(event['line'],2)

    def test_inactive_lane_variable_is_unavailable(self):
        source=(EXAMPLES/'saxpy.ntcu').read_text()
        output_line=next(i for i,text in enumerate(source.splitlines(),1) if 'y[i] = value' in text)
        module=self.module(source)
        with Session(module) as s:
            prepare(s,json.loads((EXAMPLES/'saxpy.launch.json').read_text()))
            d=Debugger(s);d.breakpoint(output_line);d.run();d.lane=9
            with self.assertRaises(ValueError):d.variable('value')

    def test_authorship_and_module_metadata(self):
        import ntcc,ntdb
        self.assertEqual(ntcc.__author__,'Tonghui Ming')
        self.assertEqual(ntdb.__author__,'Tonghui Ming')
        _,meta=Compiler().compile('__global__ void k() {}','test.ntcu')
        self.assertEqual(meta['compiler_author'],'Tonghui Ming')

    def test_ntcu_extension_required(self):
        source=self.dir/'kernel.cu';source.write_text('__global__ void k() {}')
        with self.assertRaises(CompileError):compile_file(source,self.dir/'out')
        source=source.with_suffix('.ntcu');source.write_text('__global__ void k() {}')
        self.assertTrue(compile_file(source,self.dir/'out').exists())

    def test_multidimensional_launch_and_assembly(self):
        src='''__global__ void k(uint* out) {
 uint i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
 out[i] = i + 7u;
}'''
        binary,meta=Compiler().compile(src,'grid.ntcu')
        self.assertEqual(assemble(render(meta)),binary)
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[0]*12},grid=[2,3],block=[2],backend=backend)
            self.assertEqual(out['out'],list(range(7,19)))

    def test_assembly_rejects_bad_fields(self):
        for source in ('BRA missing','MOV32I R999, 1, 0','LD R1, [R0+99999], 0, 0','MOV R1, R2, R3, 256, 0','EXIT R1'):
            with self.subTest(source=source),self.assertRaises(ValueError):assemble(source)

    def test_for_break_continue_nested_and_scope(self):
        src='''__global__ void loops(uint* out) {
 uint lane=threadIdx.x; uint total=0;
 for (uint i=0; i<lane+4u; ++i) {
   if (i==2u) continue;
   if (i==5u) break;
   for (uint j=0; j<3u; j++) {
     if (j==1u) continue;
     total += i+j;
   }
 }
 uint i=99; out[lane]=total+i;
}'''
        expected=[99+sum(i+j for i in range(min(lane+4,5)) if i!=2 for j in (0,2)) for lane in range(8)]
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[0]*8},block=8,backend=backend)
            self.assertEqual(out['out'],expected)

    def test_for_empty_parts_and_return(self):
        src='''__global__ void loops(uint* out) {
 uint lane=threadIdx.x; uint i=0; uint total=0;
 for (;;) { if (i>=lane) break; total+=i++; }
 for (;i<8u;) { if (i==6u && lane==3u) return; ++i; }
 out[lane]=total;
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[99]*8},block=8,backend=backend)
            self.assertEqual(out['out'],[0,0,1,99,6,10,15,21])

    def test_short_circuit_and_once_only_lvalue(self):
        src='''__global__ void exprs(int* out) {
 int i=0; int a=0; int b=0;
 if (0 && ++a) out[3]=99;
 if (1 || ++b) out[3]=7;
 out[i++] += 4;
 out[1]=a+b+i;
 out[2]=(i==1) && !(a!=0);
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[2,0,0,0]},block=1,backend=backend)
            self.assertEqual(out['out'],[6,1,1,7])

    def test_integer_operators_and_intrinsics(self):
        src='''__global__ void integer(int* out) {
 int a=-17; int b=5;
 out[0]=a/b; out[1]=a%b;
 int x=5; x<<=2; x|=3; x^=1; x&=31; x>>=1;
 out[2]=x; out[3]=~x; out[4]=min(a,b); out[5]=max(a,b);
 out[6]=abs(a); out[7]=__popc(0xf0fu); out[8]=__clz(16u);
 out[9]=__ctz(16u); out[10]=__find_msb(16u);
 out[11]=__bmsk(5u); out[12]=__imad(3,4,5);
 out[13]=a/0; out[14]=a%0;
}'''
        expected=[-3,-2,11,-12,-17,5,17,8,27,4,4,31,17,0,-17]
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[0]*15},block=1,backend=backend)
            self.assertEqual(out['out'],expected)

    def test_float_intrinsics(self):
        src='''__global__ void math(float* out) {
 out[0]=fabsf(-2.5f); out[1]=rcpf(4.0f); out[2]=rsqrtf(4.0f);
 out[3]=sinf(0.0f); out[4]=cosf(0.0f); out[5]=exp2f(3.0f);
 out[6]=log2f(8.0f); out[7]=floorf(-1.25f); out[8]=ceilf(-1.25f);
 out[9]=truncf(-1.25f); out[10]=nearbyintf(1.75f);
 out[11]=fminf(3.0f,2.0f); out[12]=fmaxf(3.0f,2.0f);
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[0.0]*13},block=1,backend=backend)
            self.assertEqual(out['out'],[2.5,.25,.5,0,1,8,3,-2,-1,-1,2,2,3])

    def test_atomic_and_barrier_multiple_warps(self):
        src='''__global__ void atomic(uint* count,uint* out) {
 uint old=atomicAdd(count,1u);
 __syncthreads();
 out[threadIdx.x]=count[0];
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'count':[0],'out':[0]*64},block=64,backend=backend)
            self.assertEqual(out['count'],[64]);self.assertEqual(out['out'],[64]*64)

    def test_atomic_cas_and_unsigned_min(self):
        src='''__global__ void atomic(uint* count,uint* out) {
 out[0]=atomicCAS(count,5u,9u);
 out[1]=atomicMin(count,0x80000000u);
 out[2]=atomicMax(count,0xffffffffu);
 out[3]=atomicExch(count,7u);
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'count':[5],'out':[0]*4},block=1,backend=backend)
            self.assertEqual(out['out'],[5,9,9,0xffffffff]);self.assertEqual(out['count'],[7])

    def test_inline_alu_and_round_trip(self):
        cases=[
            ('IADD3 %0, %1, %2, 0, 7',3,4,0,14),
            ('IMAD %0, %1, %2, 0, 0',3,4,5,17),
            ('SHF %0, %1, %2, 10, 0',16,0,0,4),
            ('BFE %0, %1, %2, 0, 0',240,4,4,15),
            ('BFI %0, %1, %2, 0, 0',15,4,4,244),
            ('BREV %0, %1, %2, 0, 0',1,0,0,0x80000000),
            ('FSET %0, %1, %2, 2, 3',0x3f800000,0x40000000,0,0xffffffff),
        ]
        for template,a,b,r,expected in cases:
            src=f'''__global__ void k(uint* out) {{
 uint a={a}u;uint b={b}u;uint r={r}u;
 asm("{template}" : "+r"(r) : "r"(a),"r"(b));out[0]=r;
}}'''
            binary,meta=Compiler().compile(src,'inline.ntcu')
            self.assertEqual(assemble(render(meta)),binary)
            for backend in ('func','cycle'):
                with self.subTest(template=template,backend=backend):
                    out,_=self.execute(src,{'out':[0]},block=1,backend=backend)
                    self.assertEqual(out['out'],[expected])

    def test_inline_predicates_and_branch(self):
        src='''__global__ void k(uint* out) {
 uint lane=threadIdx.x;uint limit=4;uint a=11;uint b=22;uint r=0;
 asm("SETP P1, %1, %2, 18, 3; SELP %0, %3, %4, 0, 1;"
     "SSY done; @P1 BRA chosen; MOV %0, %4, RZ, 0, 0; BRA done;"
     "chosen: MOV %0, %3, RZ, 0, 0; done: NOP"
     : "=r"(r) : "r"(lane),"r"(limit),"r"(a),"r"(b));
 out[lane]=r;
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[0]*8},block=8,backend=backend)
            self.assertEqual(out['out'],[11]*4+[22]*4)

    def test_inline_wide_memory_and_cvta(self):
        src='''__global__ void k(uint* inp,uint* out) {
 uint a=0;uint b=0;uint c=0;uint d=0;uint* addr=inp;
 asm("CVTA %0, %1, RZ, 0, 0" : "=l"(addr) : "l"(inp));
 asm("LD128 %0, [%1+0], 80, 0" : "=q"(a,b,c,d) : "l"(addr));
 asm("ST128 %0, [%1+0], 80, 1" : : "q"(a,b,c,d),"l"(out));
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'inp':[3,4,5,6],'out':[0]*4},block=1,backend=backend)
            self.assertEqual(out['out'],[3,4,5,6])

    def test_inline_m_format_ldu_and_red(self):
        src='''__global__ void k(uint* inp,uint* out) {
 uint x=0;uint off=4;uint add=9;
 asm("LDU %0, [%1+%2], 32, 1" : "=r"(x) : "l"(inp),"r"(off));
 out[0]=x;
 asm("RED RZ, %0, %1, 32, 1" : : "l"(out),"r"(add));
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'inp':[3,4],'out':[0]},block=1,backend=backend)
            self.assertEqual(out['out'],[13])

    def test_incomplete_instructions_rejected_by_both_frontends(self):
        from ntasm import UNSUPPORTED
        for name in UNSUPPORTED:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):assemble(name)
                with self.assertRaises(CompileError):Compiler().compile(f'__global__ void k() {{ asm("{name}"); }}','bad.ntcu')

    def test_extended_syntax_errors(self):
        bad=['break;','continue;','for(uint i=0;i<1;++i){} i=3;',
             'uint x=1; asm("MOV R0,R0,R0,0,0");',
             'if(1) asm("NOP");','asm("NOP");return;',
             'float x=(float)1;','uint x=1u/2u;',
             'uint x=1; asm("MOV %1, %0, RZ, 0, 0" : "=r"(x));']
        for body in bad:
            with self.subTest(body=body),self.assertRaises(CompileError):
                Compiler().compile('__global__ void k(){'+body+'}','bad.ntcu')

    def test_grid_dimensions_xyz_values(self):
        src='__global__ void k(uint* out){out[0]=gridDim.x;out[1]=gridDim.y;out[2]=gridDim.z;}'
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[0]*3},grid=[2,3,4],block=1,backend=backend)
            self.assertEqual(out['out'],[2,3,4])

    def test_for_debug_resume_equivalence(self):
        src='''__global__ void k(uint* out) {
 uint n=threadIdx.x;
 uint sum=0;
 for(uint i=0;i<n;++i) {
   if(i==2u) continue;
   sum+=i;
 }
 out[n]=sum;
}'''
        expected,event=self.execute(src,{'out':[0]*8},block=8)
        with Session(self.module(src)) as s:
            buffers=prepare(s,{'grid':1,'block':8,'arguments':{'out':[0]*8}})
            d=Debugger(s);d.breakpoint(6)
            self.assertEqual(d.run()['reason'],'breakpoint')
            d.run(True)
            # Clear breakpoint through the debugger's protocol before finishing.
            [s.command(f'delete {pc}') for pc in d.breakpoints[6]]
            done=s.synchronize()
            self.assertEqual(collect(s,buffers),expected)
            self.assertEqual(done['instructions'],event['instructions'])

    def test_float_arithmetic_unsigned_shift_and_rounding(self):
        src='''__global__ void k(float* out,uint* bits) {
 float a=3.0f;float b=2.0f;
 out[0]=a+b;out[1]=a-b;out[2]=a*b;
 uint x=0x80000000u;bits[0]=x>>31;
 uint r=0;uint p=0x3f800000u;uint q=0x33800000u;
 asm("FADD.m3 %0, %1, %2, 0, 0" : "=r"(r) : "r"(p),"r"(q));
 bits[1]=r;
}'''
        for backend in ('func','cycle'):
            out,_=self.execute(src,{'out':[0.0]*3,'bits':[0]*2},block=1,backend=backend)
            self.assertEqual(out['out'],[5.0,1.0,6.0]);self.assertEqual(out['bits'],[1,0x3f800001])

    def test_assembler_rejects_incomplete_modes(self):
        for text in ('S2R R0,RZ,RZ,40,0','ATOM R0,R2,R4,48,1',
                     'IADD R1,R2,R4,6,0','MOV.m4 R0,R1,RZ,0,0',
                     'LD R0,[R2+0],128,0','BAR R0,R0,R0,48,0'):
            with self.subTest(text=text),self.assertRaises(ValueError):assemble(text)

if __name__=='__main__':
    program=unittest.main(verbosity=2,exit=False)
    if len(sys.argv)==1 and program.result.wasSuccessful():
        from ntasm import OPS
        missing=set(OPS)-ToolchainTests.executed_opcodes
        if missing:raise AssertionError(f'opcodes without execution examples: {missing}')
        print(f'OPCODE_EXAMPLES_PASS {len(OPS)}')
    sys.exit(not program.result.wasSuccessful())
