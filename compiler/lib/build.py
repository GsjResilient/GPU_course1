"""Content-addressed, at-most-two-worker C++ build using original sources."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]


def build_tools(out,cxx='clang++'):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    src=REPO/'NutShellGPU_sim'
    flags=[cxx,'-std=c++17','-O2','-frounding-math','-ffp-contract=off','-I'+str(src)]
    version=subprocess.check_output([cxx,'--version'],text=True)
    files=sorted((REPO/'compiler/runtime').glob('*'))+[REPO/'compiler/tests/decode_check.cpp',src/'func/func_sim.cpp',src/'timing/cycle_sim.cpp']+sorted(src.rglob('*.hpp'))
    oracle=REPO/'compiler/tests/oracle.cpp'
    if oracle.exists():files.append(oracle)
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    key=hashlib.sha256(json.dumps([flags,version,hashes],sort_keys=True).encode()).hexdigest()[:20]
    target=out/key;target.mkdir(exist_ok=True)
    names=['bridge','decode_check']+(['oracle'] if oracle.exists() else [])
    outputs={n:target/n for n in names}
    stamp=target/'build.json'
    if stamp.exists():
        previous=json.loads(stamp.read_text())
        if all(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==previous.get('binaries',{}).get(n) for n,p in outputs.items()):
            return outputs
    commands=[]
    def execute(pair):
        name,cmd=pair
        r=subprocess.run(cmd,cwd=target,capture_output=True,text=True,timeout=180)
        (target/(name+'.log')).write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('C++ build failed: '+str(target/(name+'.log')))
        return cmd
    objects=[('func',src/'func/func_sim.cpp'),('cycle',src/'timing/cycle_sim.cpp')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        commands.extend(pool.map(execute,[(n,flags+['-c',str(p),'-o',str(target/(n+'.o'))]) for n,p in objects]))
    commands.append(execute(('bridge',flags+[str(REPO/'compiler/runtime/bridge.cpp'),str(target/'func.o'),str(target/'cycle.o'),'-o',str(outputs['bridge'])])))
    commands.append(execute(('decode_check',flags+[str(REPO/'compiler/tests/decode_check.cpp'),'-o',str(outputs['decode_check'])])))
    if oracle.exists():commands.append(execute(('oracle',flags+[str(oracle),'-o',str(outputs['oracle'])])))
    stamp.write_text(json.dumps(dict(version=version,flags=flags,sources=hashes,commands=commands,
                                   binaries={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in outputs.items()}),indent=2)+'\n')
    return outputs
