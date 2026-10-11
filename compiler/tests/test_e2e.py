# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
import itertools,json,os,struct,subprocess,unittest
from pathlib import Path
from compiler.lib.driver import compile_file
from compiler.lib.runtime import run_artifact,BufferRef
from compiler.lib.build import build_tools,REPO
from compiler.lib.artifact import load_artifact
from compiler.lib.diagnostics import CompileError
from compiler.tests.cases import make_case,cpu_f32


class E2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools=build_tools(Path(os.environ.get('NUTCC_TOOLS_DIR',str(REPO/'build/nutcc-tools'))))
        cls.out=Path(os.environ.get('NUTCC_CASE_DIR',str(REPO/'build/nutcc-cases')))
        cls.out.mkdir(parents=True,exist_ok=True)
        cls.sequence=itertools.count()

    def execute(self,source,buffers,args,expected,grid=(1,1,1),block=(1,1,1),floating=False,trace=False):
        directory=self.out/(self._testMethodName+'-'+str(next(self.sequence)))
        directory.mkdir(parents=True,exist_ok=True)
        s=directory/'kernel.cpp';s.write_text(source)
        (directory/'expected.bin').write_bytes(expected)
        for name,data in buffers.items():(directory/(name+'.input.bin')).write_bytes(data)
        (directory/'launch.json').write_text(json.dumps(dict(format='nutcc-launch-v1',grid=grid,block=block,
            buffers={n:n+'.input.bin' for n in buffers},args=[dict(buffer=a.name,offset=a.offset) if isinstance(a,BufferRef) else a for a in args])))
        results=[]
        for level in (0,1):
            mode=directory/f'O{level}'
            m=compile_file(s,'k',mode/'kernel',opt_level=level)
            meta,code=load_artifact(m)
            subprocess.run([str(self.tools['decode_check']),str(mode/meta['code_file']),str(meta['regs_u32'])],check=True,capture_output=True)
            for backend in ('func','cycle'):
                with self.subTest(opt_level=level,backend=backend,case=str(directory)):
                    r=run_artifact(m,backend,buffers,args,grid,block,mode/backend,trace=trace and backend=='func')
                    self.assertTrue(r.completed);actual=r.buffers['out']
                    if floating:
                        self.assertEqual(len(actual),len(expected))
                        for (a,),(b,) in zip(struct.iter_unpack('<I',actual),struct.iter_unpack('<I',expected)):
                            if b&0x7fffffff>0x7f800000:self.assertGreater(a&0x7fffffff,0x7f800000)
                            else:self.assertEqual(a,b)
                    else:self.assertEqual(actual,expected)
                    results.append(r)
        return results

    def test_e2e_sizes(self):
        for n in (1,31,32,33,65,257):
            for name in ('u32_arithmetic','i32_arithmetic','i32_compare','f32_add','u8_copy','u8_promote'):
                with self.subTest(n=n,name=name):
                    c=make_case(name,n,20260924,self.tools['oracle'])
                    self.execute(c['source'],c['buffers'],c['args'],c['expected'],c['grid'],c['block'],name=='f32_add')

    def test_f32_specials_and_comparisons(self):
        for name in ('f32_sub','f32_mul','f32_eq','f32_ne','f32_lt','f32_le','f32_gt','f32_ge','f32_not_lt','f32_neg'):
            c=make_case(name,65,20260924,self.tools['oracle'])
            self.execute(c['source'],c['buffers'],c['args'],c['expected'],c['grid'],c['block'],name in ('f32_sub','f32_mul','f32_neg'))
        self.execute('NT_KERNEL void k(float* out){out[0]=(1.00000011920928955078125f*0.99999988079071044921875f)-1.0f;out[1]=-0.0f;}',
                     {'out':bytes(8)},[BufferRef('out')],struct.pack('<II',0,0x80000000),floating=True)

    def test_condition_snapshot_nested_and_short_circuit(self):
        self.execute('NT_KERNEL void k(int* out){int c=1;int r=0;if(c){c=0;r=1;}else{r=2;}out[0]=r;}',
                     {'out':bytes(4)},[BufferRef('out')],struct.pack('<i',1))
        source='NT_KERNEL void k(const int* a,int* out){unsigned int i=threadIdx.x;int r=9;'
        source+='if(i<8U){if(i<3U && a[i]>0){r=1;}else{r=2;}}else{if(i>=8U || a[i]>0){r=3;}}out[i]=r;}'
        result=self.execute(source,{'a':struct.pack('<3i',1,-1,2),'out':bytes(128)},
                            [BufferRef('a'),BufferRef('out')],struct.pack('<32i',1,2,1,*([2]*5),*([3]*24)),block=(32,1,1),trace=True)
        for func in result[::2]:
            loads=[event for event in func.trace if event[1]==0x60]
            self.assertTrue(loads)
            self.assertTrue(all(mask&~7==0 for _,_,mask in loads),loads)

    def test_fixed_loops(self):
        for n in (0,1,3,256):
            self.execute('NT_KERNEL void k(int* out){int s=0;for(int j=0;j<'+str(n)+';++j){s+=j;}out[0]=s;}',
                         {'out':bytes(4)},[BufferRef('out')],struct.pack('<i',sum(range(n))))
        self.execute('NT_KERNEL void k(int* out){int s=0;for(int i=0;i<3;++i){for(int j=0;j<4;++j){s+=i*j;}}out[0]=s;}',
                     {'out':bytes(4)},[BufferRef('out')],struct.pack('<i',18))

    def test_constant_unsigned_wraparound(self):
        self.execute('NT_KERNEL void k(int* out){if(0xffffffffU+1U){out[0]=2;}else{out[0]=1;}if(-1U<1U){out[1]=2;}else{out[1]=1;}}',
                     {'out':bytes(8)},[BufferRef('out')],struct.pack('<ii',1,1))

    def test_loop_local_requires_initialization_each_iteration(self):
        p=self.out/'uninitialized_iteration.cpp'
        p.write_text('NT_KERNEL void k(int* out){for(int i=0;i<2;++i){int v;if(i==0){v=7;}else{out[0]=v;}}}')
        with self.assertRaisesRegex(CompileError,'E_UNINITIALIZED'):compile_file(p,'k',self.out/'bad')

    def test_multidimensional_coordinates(self):
        for block,grid in (((5,3,1),(2,3,1)),((3,2,2),(2,1,2))):
            nx,ny,nz=[b*g for b,g in zip(block,grid)]
            source='NT_KERNEL void k(unsigned int* out){unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;unsigned int z=blockIdx.z*blockDim.z+threadIdx.z;unsigned int nx=gridDim.x*blockDim.x;unsigned int ny=gridDim.y*blockDim.y;out[(z*ny+y)*nx+x]=x+100U*y+10000U*z;}'
            values=[x+100*y+10000*z for z in range(nz) for y in range(ny) for x in range(nx)]
            self.execute(source,{'out':bytes(4*len(values))},[BufferRef('out')],struct.pack('<'+'I'*len(values),*values),grid,block)

    def test_alias_inplace_sequence(self):
        self.execute('NT_KERNEL void k(int* p,int* q){unsigned int i=threadIdx.x;p[i]+=2;q[i]=p[i]*3;p[i]-=1;}',
                     {'out':struct.pack('<32i',*range(32))},[BufferRef('out'),BufferRef('out')],
                     struct.pack('<32i',*[(x+2)*3-1 for x in range(32)]),block=(32,1,1))

    def test_parenthesized_compound_lvalue_evaluated_once(self):
        self.execute('NT_KERNEL void k(unsigned int* out){unsigned int i=0U;((out[i=i+1U]))+=7U;out[3]=i;}',
                     {'out':struct.pack('<4I',10,20,30,0)},[BufferRef('out')],struct.pack('<4I',10,27,30,1))
        self.execute('NT_KERNEL void k(unsigned int* out){unsigned int i=threadIdx.x;if(i<3U){((out[i]))+=7U;}}',
                     {'out':struct.pack('<3I',10,20,30)},[BufferRef('out')],struct.pack('<3I',17,27,37),block=(32,1,1))

    def test_sequenced_operand_snapshots(self):
        cases=[('unsigned int i=0U;out[i=1U]=i;out[2]=i;',(10,0,1)),
               ('unsigned int i=0U;out[i=1U]+=i;out[2]=i;',(10,20,1)),
               ('unsigned int i=3U;out[0]=i<<(i=1U);out[1]=i;',(6,1,30))]
        for body,expected in cases:
            self.execute('NT_KERNEL void k(unsigned int* out){'+body+'}',
                         {'out':struct.pack('<3I',10,20,30)},[BufferRef('out')],struct.pack('<3I',*expected))
        self.execute('NT_KERNEL void k(unsigned int* out){unsigned int i=threadIdx.x;if(i<3U){unsigned int v=3U;out[i]=v<<(v=1U);}}',
                     {'out':bytes(12)},[BufferRef('out')],struct.pack('<3I',6,6,6),block=(32,1,1))

    def test_constant_bitwise_loop_bounds(self):
        for bound,count in [('3&1',1),('2|1',3),('5^6',3),('~(-4)',3)]:
            source='constexpr int N='+bound+';NT_KERNEL void k(int* out){int v=0;for(int j=0;j<N;++j){v+=1;}out[0]=v;}'
            self.execute(source,{'out':bytes(4)},[BufferRef('out')],struct.pack('<i',count))

    def test_valid_global_constexpr_values(self):
        self.execute('constexpr float A=1.25f;constexpr float B=A+0.5f;NT_KERNEL void k(float* out){out[0]=B;}',
                     {'out':bytes(4)},[BufferRef('out')],struct.pack('<f',1.75),floating=True)

    def test_resource_pressure(self):
        def source(n):
            declarations=''.join('int v'+str(i)+'=a['+str(i)+'];' for i in range(n))
            return 'NT_KERNEL void k(const int* a,int* out){'+declarations+'out[0]='+'+'.join('v'+str(i) for i in range(n))+';}'
        self.execute(source(80),{'a':struct.pack('<80i',*range(80)),'out':bytes(4)},[BufferRef('a'),BufferRef('out')],struct.pack('<i',sum(range(80))))
        p=self.out/'too_many_registers.cpp';p.write_text(source(270))
        with self.assertRaisesRegex(CompileError,'E_REGISTERS'):compile_file(p,'k',self.out/'bad')
        p.write_text('NT_KERNEL void k(int* out){int s=0;for(int i=0;i<256;++i){for(int j=0;j<256;++j){s+=i*j;}}out[0]=s;}')
        with self.assertRaisesRegex(CompileError,'E_CODE_SIZE'):compile_file(p,'k',self.out/'bad')

    def test_oracle_input_rejection(self):
        for text in ('bad 0 0\n','add xyz 0\n','add 100000000 0\n','add 0 0 extra\n'):
            r=subprocess.run([str(self.tools['oracle'])],input=text,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
