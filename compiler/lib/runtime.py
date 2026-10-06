# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
"""Host validation and launch adapter; all kernel arithmetic runs in the simulator."""
from dataclasses import dataclass
from pathlib import Path
import os,struct,subprocess
from .artifact import load_artifact,integer
from .build import build_tools,REPO


class RunError(Exception):pass


@dataclass(frozen=True)
class BufferRef:
    name: str
    offset: int=0


@dataclass
class RunResult:
    buffers: dict
    instructions: int
    cycles: int
    completed: bool
    trace: list


def check_span(base,size):
    if not integer(base,0,0xffffffff) or not integer(size,0,0x100000000) or base+size>0x100000000:
        raise RunError('buffer exceeds low32 address profile')


def dimensions(value,limit):
    if not isinstance(value,(tuple,list)) or len(value)!=3:raise RunError('expected three launch dimensions')
    product=1
    for x in value:
        if not integer(x,1,limit):raise RunError('invalid launch dimension')
        product*=x
    if product>limit:raise RunError('launch dimensions exceed limit')
    return tuple(value)


def run_artifact(manifest,backend,buffers,args,grid,block,out,timeout_s=20.0,max_steps=10000000,trace=False):
    if backend not in ('func','cycle') or trace and backend!='func':raise RunError('invalid backend/trace mode')
    if not integer(max_steps,1,1000000000) or not isinstance(timeout_s,(int,float)) or not 0<timeout_s<=3600:raise RunError('invalid execution budget')
    grid=dimensions(grid,0x7fffffff);block=dimensions(block,1024)
    meta,code=load_artifact(manifest)
    if type(buffers) is not dict or len(buffers)>512 or any(type(k) is not str or type(v) is not bytes for k,v in buffers.items()):raise RunError('invalid buffers')
    if sum(map(len,buffers.values()))>256*1024*1024:raise RunError('input exceeds 256 MiB budget')
    if not isinstance(args,(list,tuple)) or len(args)!=len(meta['params']):raise RunError('argument count mismatch')
    names=list(buffers);records=[]
    for param,value in zip(meta['params'],args):
        kind=param['kind']
        if kind.startswith('ptr_'):
            if not isinstance(value,BufferRef) or value.name not in buffers or not integer(value.offset,0,len(buffers[value.name])):raise RunError('invalid buffer reference')
            alignment=1 if kind=='ptr_u8' else 4
            if value.offset%alignment:raise RunError('unaligned buffer reference')
            records.append(struct.pack('<HBBQQ',param['offset'],1,0,names.index(value.name),value.offset))
        else:
            if kind=='f32':
                if type(value) is not float:raise RunError('f32 argument requires float')
                try:bits=struct.unpack('<I',struct.pack('<f',value))[0]
                except (OverflowError,struct.error) as e:raise RunError('f32 scalar out of range') from e
            else:
                if not integer(value,-(1<<31) if kind=='i32' else 0,(1<<31)-1 if kind=='i32' else 0xffffffff):raise RunError('integer scalar out of range')
                bits=value&0xffffffff
            records.append(struct.pack('<HBBQQ',param['offset'],0,0,bits,0))
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    request=b'NTRUN01\0'+struct.pack('<11IQ',meta['instruction_count'],meta['regs_u32'],meta['param_size'],*grid,*block,len(names),len(records),max_steps)+code
    for data in buffers.values():request+=struct.pack('<Q',len(data))+data
    request+=b''.join(records)
    req=out/'request.bin';res=out/'response.bin';tracefile=out/'trace.tsv'
    req.write_bytes(request)
    if res.exists():res.unlink()
    tools=build_tools(Path(os.environ.get('NUTCC_TOOLS_DIR',str(REPO/'build/nutcc-tools'))))
    cmd=[str(tools['bridge']),backend,str(req),str(res)]+([str(tracefile)] if trace else [])
    try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout_s)
    except subprocess.TimeoutExpired as e:raise RunError('simulator wall timeout') from e
    (out/'execution.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RunError(r.stderr.strip() or 'simulator exited '+str(r.returncode))
    try:
        data=res.read_bytes();at=0
        def take(n):
            nonlocal at
            if at+n>len(data):raise RunError('truncated response')
            part=data[at:at+n];at+=n;return part
        if take(8)!=b'NTOUT01\0':raise RunError('invalid response magic')
        instructions,cycles,count=struct.unpack('<QQI',take(20))
        if count!=len(names):raise RunError('response buffer count mismatch')
        result={}
        for name in names:
            size,=struct.unpack('<Q',take(8))
            if size!=len(buffers[name]):raise RunError('response buffer size mismatch')
            result[name]=take(size)
        if at!=len(data):raise RunError('trailing response bytes')
        events=[tuple(map(int,line.split())) for line in tracefile.read_text().splitlines()] if trace else []
        return RunResult(result,instructions,cycles,True,events)
    except (OSError,ValueError,struct.error) as e:raise RunError('invalid simulator response: '+str(e)) from e
