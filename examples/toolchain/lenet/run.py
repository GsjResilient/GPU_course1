# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
"""Run the SAME TinyLeNet .ntcu sources through NTCC or native CUDA NVRTC."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASELINE=ROOT/'examples/tiny_lenet'
sys.path.insert(0,str(ROOT/'toolchain'));sys.path.insert(0,str(BASELINE))
from ntcc import compile_file
from ntasm import assemble
from ntruntime import Session
from ntdb import Debugger
from test_example import reference


def assets():
    base=BASELINE/'assets';meta=json.loads((base/'model.json').read_text())
    raw=(base/'weights.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==meta['weights_sha256']
    w={v['name']:np.frombuffer(raw,dtype='<f4',count=v['nbytes']//4,offset=v['offset_bytes']).copy().reshape(v['shape']) for v in meta['tensors']}
    return w,np.fromfile(base/'input.bin',dtype='<f4').reshape(-1,1,28,28),np.fromfile(base/'labels.bin',dtype='u1')


class NativeCUDA:
    def __init__(self):
        from cuda_api import CUDA
        self.cuda=CUDA()
        sources=[ROOT/'examples/toolchain/vector_add.ntcu']+[HERE/f'{n}.ntcu' for n in ('conv','pool','dense','argmax')]
        self.cuda.compile('typedef unsigned int uint;\n'+'\n'.join('extern "C" {\n'+p.read_text()+'\n}' for p in sources))

    def execute(self,name,args,shape,grid,block):
        from cuda_api import P,I,U
        pointers=[];values=[]
        try:
            for value in args.values():
                if isinstance(value,np.ndarray):
                    pointer=self.cuda.upload(np.ascontiguousarray(value));pointers.append(pointer);values.append(pointer)
                else:values.append(I(value))
            params=(P*len(values))(*(C.cast(C.byref(v),P) for v in values))
            function=P();self.cuda.call('cuModuleGetFunction',C.byref(function),self.cuda.module,name.encode())
            g=list(grid)+[1]*(3-len(grid));b=list(block)+[1]*(3-len(block))
            self.cuda.call('cuLaunchKernel',function,*(I(v) for v in g+b),I(0),self.cuda.stream,params,P())
            output_index=list(args).index('output' if 'output' in args else 'out')
            result=np.empty(shape,dtype='i4' if name=='classify' else 'f4')
            self.cuda.download(values[output_index],result)
            return result,{'reason':'completed','device':self.cuda.name}
        finally:
            for pointer in pointers:self.cuda.call('cuMemFree_v2',pointer)

    def close(self):self.cuda.close()


def debug_layer(session,source,transcript):
    line=next(i for i,t in enumerate(source.splitlines(),1) if 'output[' in t and '=' in t)
    d=Debugger(session)
    def command(text):
        value=d.command(text);transcript.append({'command':text,'result':value});return value
    command(f'break {line}');event=command('continue')
    if event['reason']!='breakpoint':raise AssertionError('expected source breakpoint')
    command('where')
    names={'conv':['x','y','co','oi','sum'],'pool':['x','y','c','oi','best'],
           'dense':['co','count','sum'],'classify':['count','index','best']}[session.meta['kernel']]
    for name in names:command('print '+name)
    command('stack');command(f'delete {line}')
    command('step');return command('continue')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['func','cycle','cuda'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--samples',type=int,default=3)
    p.add_argument('--debug',action='store_true')
    a=p.parse_args()
    if a.debug and a.backend!='func':p.error('debug requires func backend')
    if not 1<=a.samples<=3:p.error('samples must be 1..3')
    a.out.mkdir(parents=True,exist_ok=True)
    w,inputs,labels=assets();refs=reference();modules={};compilation={}
    for name in ('conv','pool','dense','argmax'):
        module=compile_file(HERE/f'{name}.ntcu',a.out/'compiled'/name);modules[name]=module
        binary=module.with_name(name+'.ntas').read_bytes()
        assert assemble(module.with_name(name+'.ntasm').read_text())==binary
        meta=json.loads(module.read_text());compilation[name]={'registers':meta['registers'],'instructions':len(binary)//8,'sha256':meta['code_sha256'],'assembly_roundtrip':True}
        print('COMPILED',name,json.dumps(compilation[name]),flush=True)
    cuda=NativeCUDA() if a.backend=='cuda' else None
    records=[];predictions=[];debug=[]
    try:
        for sample in range(a.samples):
            x=inputs[sample].copy()
            def step(kernel,layer,shape,arguments,grid,block):
                nonlocal records
                start=time.perf_counter();dtype='i4' if kernel=='argmax' else 'f4'
                arguments['output']=np.zeros(shape,dtype=dtype)
                # ABI follows declaration order; metadata determines parameter offsets.
                meta=json.loads(modules[kernel].read_text())
                arguments={par['name']:arguments[par['name']] for par in meta['parameters']}
                transcript=[]
                if cuda:
                    actual,event=cuda.execute(meta['kernel'],arguments,shape,grid,block)
                else:
                    with Session(modules[kernel],a.backend) as s:
                        params={key:s.upload(np.ascontiguousarray(v).tobytes()) if isinstance(v,np.ndarray) else v for key,v in arguments.items()}
                        s.launch(grid,block,params)
                        if a.debug and sample==0:
                            event=debug_layer(s,(HERE/f'{kernel}.ntcu').read_text(),transcript)
                            if event['reason']!='completed':raise AssertionError(event)
                        else:event=s.synchronize(100000000)
                        actual=np.frombuffer(s.download(params['output'],int(np.prod(shape))*4),dtype='<'+dtype).reshape(shape).copy()
                elapsed=time.perf_counter()-start
                actual.tofile(a.out/f'{sample}_{layer}.bin')
                if layer=='argmax':
                    expected=int(refs['logits'][sample].argmax());assert int(actual[0])==expected
                    error=0.;predictions.append(int(actual[0]))
                else:
                    ref=refs[layer][sample];np.testing.assert_allclose(actual,ref,atol=1e-4,rtol=1e-4,err_msg=f'{sample}/{layer}')
                    error=float(np.abs(actual-ref).max())
                row={'sample':sample,'layer':layer,'shape':list(shape),'max_abs_error':error,'host_layer_seconds':elapsed,'execution':event}
                records.append(row);print(json.dumps(row),flush=True)
                if transcript:debug.append({'sample':sample,'layer':layer,'commands':transcript})
                return actual
            for number,ci,co,h in [(1,1,8,28),(2,8,16,14)]:
                x=step('conv',f'relu{number}',(co,h,h),{'input':x,'weight':w[f'conv{number}.weight'],'bias':w[f'conv{number}.bias'],'ci':ci,'h':h},(h,co),(h,))
                x=step('pool',f'pool{number}',(co,h//2,h//2),{'input':x,'h':h,'out_h':h//2},(h//2,co),(h//2,))
            logits=step('dense','logits',(10,),{'input':x,'weight':w['fc.weight'],'bias':w['fc.bias'],'count':784},(1,),(10,))
            step('argmax','argmax',(1,),{'input':logits,'count':10},(1,),(1,))
    finally:
        if cuda:cuda.close()
    result={'backend':a.backend,'debug':a.debug,'samples':a.samples,'predictions':predictions,'labels':labels[:a.samples].tolist(),
            'accuracy_on_fixture':float(np.mean(np.array(predictions)==labels[:a.samples])),'compilation':compilation,'layers':records,
            'total_cycles':sum(r['execution'].get('cycles',0) for r in records)}
    (a.out/'results.json').write_text(json.dumps(result,indent=2))
    (a.out/'debug.json').write_text(json.dumps(debug,indent=2))
    print('PREDICTIONS',predictions,'LABELS',labels[:a.samples].tolist(),'ALL_LAYER_CHECKS_PASS',flush=True)


if __name__=='__main__':main()
