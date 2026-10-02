"""Versioned machine-code artifacts, published with a manifest commit marker."""
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
from .diagnostics import CompileError,SourceLoc
from .toolchain import supported_frontend

VERSION=(Path(__file__).resolve().parents[1]/'VERSION').read_text().strip()
PROFILE='ntas1-sim-v1.10-low32-predicated-v1'
FORMAT='nutcc-artifact-v1'
UNSUPPORTED=['dynamic loops','device calls','early return','shared/atomic/barrier',
             '64-bit addressing','integer/float conversion','spilling','RTL execution']
KINDS={'i32','u32','f32','ptr_i32','ptr_u32','ptr_f32','ptr_u8'}


def fail(path,message):
    raise CompileError('E_ARTIFACT',message,SourceLoc(str(path),1,1))


def unique_object(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('duplicate JSON key: '+k)
        result[k]=v
    return result


def integer(value,lo,hi):
    return type(value) is int and lo<=value<=hi


def validate_manifest(meta,path):
    required={'format','compiler_version','clang_version','source_sha256','dependencies','profile','kernel',
              'entry','code_file','code_sha256','instruction_count','regs_u32','regs_u64','param_size','params','unsupported'}
    if type(meta) is not dict or set(meta)!=required:fail(path,'unknown or missing manifest fields')
    if meta['format']!=FORMAT or meta['profile']!=PROFILE or meta['compiler_version']!=VERSION:
        fail(path,'unsupported artifact version/profile')
    for key in ('source_sha256','code_sha256'):
        if type(meta[key]) is not str or not re.fullmatch('[0-9a-f]{64}',meta[key]):fail(path,'invalid '+key)
    if not isinstance(meta['dependencies'],dict) or not all(type(k) is str and type(v) is str and re.fullmatch('[0-9a-f]{64}',v) for k,v in meta['dependencies'].items()):
        fail(path,'invalid dependency hashes')
    if not supported_frontend(meta['clang_version']):fail(path,'unsupported frontend version')
    if type(meta['kernel']) is not str or not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',meta['kernel']):fail(path,'invalid kernel name')
    if type(meta['code_file']) is not str or not meta['code_file'].endswith('.ntbin') or Path(meta['code_file']).name!=meta['code_file'] or '\\' in meta['code_file']:
        fail(path,'code_file must be a local basename')
    for key,lo,hi in [('entry',0,0),('regs_u64',0,0),('regs_u32',2,255),('instruction_count',1,4096),('param_size',0,4096)]:
        if not integer(meta[key],lo,hi):fail(path,'invalid '+key)
    params=meta['params']
    if not isinstance(params,list) or len(params)>512 or meta['param_size']!=8*len(params):fail(path,'invalid parameter size')
    names=set()
    for index,p in enumerate(params):
        if type(p) is not dict or set(p)!={'name','kind','offset','size','readonly'}:fail(path,'invalid parameter fields')
        if not isinstance(p['name'],str) or not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',p['name']) or p['name'] in names:fail(path,'invalid or duplicate parameter name')
        names.add(p['name'])
        if p['kind'] not in KINDS or type(p['readonly']) is not bool:fail(path,'invalid parameter type')
        size=8 if p['kind'].startswith('ptr_') else 4
        if not integer(p['offset'],index*8,index*8) or not integer(p['size'],size,size):fail(path,'invalid parameter layout')
    if meta['unsupported']!=UNSUPPORTED:fail(path,'profile capability declaration does not match')


def load_artifact(manifest):
    manifest=Path(manifest)
    try:
        meta=json.loads(manifest.read_text(),object_pairs_hook=unique_object,
                        parse_constant=lambda s:(_ for _ in ()).throw(ValueError('invalid JSON number '+s)))
        validate_manifest(meta,manifest)
        path=manifest.parent/meta['code_file']
        if path.resolve().parent!=manifest.parent.resolve():fail(manifest,'code path escapes artifact directory')
        code=path.read_bytes()
        if len(code)!=8*meta['instruction_count']:fail(manifest,'code length does not match')
        if hashlib.sha256(code).hexdigest()!=meta['code_sha256']:fail(manifest,'machine-code hash mismatch')
        return meta,code
    except CompileError:raise
    except (OSError,ValueError,TypeError,KeyError) as e:fail(manifest,str(e))


def write_artifact(machine,unit,prefix):
    prefix=Path(prefix)
    code=b''.join(struct.pack('<Q',w) for w in machine.words)
    code_path=prefix.with_name(prefix.name+'.ntbin')
    manifest=prefix.with_name(prefix.name+'.json')
    meta=dict(format=FORMAT,compiler_version=VERSION,clang_version=unit.clang_version,
              source_sha256=hashlib.sha256(unit.source_bytes).hexdigest(),dependencies=unit.dependencies,
              profile=PROFILE,kernel=machine.name,entry=0,code_file=code_path.name,
              code_sha256=hashlib.sha256(code).hexdigest(),instruction_count=len(machine.words),
              regs_u32=machine.regs_u32,regs_u64=0,param_size=machine.param_size,
              params=machine.params,unsupported=UNSUPPORTED)
    validate_manifest(meta,manifest)
    try:
        prefix.parent.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(prefix='.nutcc-',dir=prefix.parent) as directory:
            root=Path(directory)
            (root/code_path.name).write_bytes(code)
            (root/'listing').write_text('\n'.join(machine.listing)+'\n')
            (root/manifest.name).write_text(json.dumps(meta,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
            load_artifact(root/manifest.name)
            os.replace(root/code_path.name,code_path)
            os.replace(root/'listing',prefix.with_name(prefix.name+'.ntasm'))
            os.replace(root/manifest.name,manifest)
    except OSError as e:fail(manifest,str(e))
    return manifest
