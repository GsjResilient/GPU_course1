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
"""Deterministic, independent CPU expectations for actual compiled kernels."""
import random,struct,subprocess
from compiler.lib.runtime import BufferRef


def pack(values,kind='I'):
    return struct.pack('<'+kind*len(values),*values)


def cpu_f32(operations,oracle):
    text=''.join(f'{op} {a:08x} {b:08x}\n' for op,a,b in operations)
    r=subprocess.run([str(oracle)],input=text,text=True,capture_output=True,timeout=10)
    if r.returncode:raise RuntimeError('CPU oracle failed: '+r.stderr)
    values=[int(x,16) for x in r.stdout.splitlines()]
    if len(values)!=len(operations):raise RuntimeError('CPU oracle result count mismatch')
    return values


def make_case(name,n,seed,oracle):
    rng=random.Random(seed);out_kind='I';out_type='unsigned int';body='';floating=False
    if name=='u32_arithmetic':
        kind='I';ctype='unsigned int'
        a=[rng.randrange(1<<32) for _ in range(n)];b=[rng.randrange(1<<32) for _ in range(n)]
        body='unsigned int v=(a[i]+b[i])*3U-7U;out[i]=((v>>3U)^(a[i]<<2U))|((a[i]&b[i])^0xffU);'
        expected=[(((((x+y)&0xffffffff)*3-7)&0xffffffff)>>3 ^ ((x<<2)&0xffffffff))|((x&y)^255) for x,y in zip(a,b)]
    elif name=='i32_arithmetic':
        kind='i';ctype='int';out_type='int';out_kind='i'
        a=[rng.randrange(-10000,10001) for _ in range(n)];b=[rng.randrange(-10000,10001) for _ in range(n)]
        body='out[i]=-(a[i]+b[i])*3-a[i];'
        expected=[-(x+y)*3-x for x,y in zip(a,b)]
    elif name=='i32_compare':
        kind='i';ctype='int';out_type='int';out_kind='i'
        a=[rng.randrange(-10000,10001) for _ in range(n)];b=[rng.randrange(-10000,10001) for _ in range(n)]
        for i,x in enumerate((0,-1,1,-2147483648,2147483647)):
            if i<n:a[i]=x;b[i]=x if i%2==0 else 0
        body='int v=0;'
        comparisons=[('==',lambda x,y:x==y),('!=',lambda x,y:x!=y),('<',lambda x,y:x<y),('<=',lambda x,y:x<=y),('>',lambda x,y:x>y),('>=',lambda x,y:x>=y)]
        expected=[0]*n
        for index,(symbol,op) in enumerate(comparisons):
            body+=f'if(a[i]{symbol}b[i]){{v+={1<<index};}}'
            expected=[v+((1<<index) if op(x,y) else 0) for v,x,y in zip(expected,a,b)]
        body+='out[i]=v;'
    elif name in ('u8_copy','u8_promote'):
        kind='B';ctype='unsigned char';a=[i%256 for i in range(n)];b=[0]*n
        if name=='u8_copy':out_kind='B';out_type='unsigned char';body='out[i]=a[i];';expected=a
        else:out_kind='i';out_type='int';body='out[i]=a[i]+1;';expected=[x+1 for x in a]
    elif name.startswith('f32_'):
        kind='I';ctype='float';operation=name[4:]
        special=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc12345,0x7f7fffff]
        a=[special[i] if i<len(special) else rng.randrange(1<<32) for i in range(n)]
        b=[special[(i+5)%len(special)] if i<len(special) else (a[i] if i%3==0 else rng.randrange(1<<32)) for i in range(n)]
        symbols=dict(add='+',sub='-',mul='*',eq='==',ne='!=',lt='<',le='<=',gt='>',ge='>=')
        expected=cpu_f32([('lt' if operation=='not_lt' else operation,x,y) for x,y in zip(a,b)],oracle)
        if operation in ('add','sub','mul','neg'):
            out_type='float';floating=True
            body='out[i]=-a[i];' if operation=='neg' else 'out[i]=a[i]'+symbols[operation]+'b[i];'
        else:
            expression='!(a[i]<b[i])' if operation=='not_lt' else 'a[i]'+symbols[operation]+'b[i]'
            body='if('+expression+'){out[i]=1U;}else{out[i]=0U;}'
            if operation=='not_lt':expected=[1-x for x in expected]
    else:raise ValueError(name)
    source=f'NT_KERNEL void k(const {ctype}* a,const {ctype}* b,{out_type}* out,unsigned int n){{unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){{'+body+'}}'
    expected=pack(expected,out_kind)
    return dict(source=source,kernel='k',buffers=dict(a=pack(a,kind),b=pack(b,kind),out=bytes(len(expected))),
                args=[BufferRef('a'),BufferRef('b'),BufferRef('out'),n],grid=((n+31)//32,1,1),block=(32,1,1),expected=expected)
