#!/usr/bin/env python3
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
"""Fail-closed test release gate. Every mandatory suite must execute and pass."""
import argparse,hashlib,json,os,platform,struct,subprocess,sys,unittest,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from compiler.lib.build import build_tools,REPO
from compiler.lib.runtime import RunError
from compiler.lib.toolchain import frontend_compiler,native_compiler,native_command,supported_frontend,SUPPORTED_FRONTENDS

BASE='c1b2aed4634bf2b82e0a2674caeabf2068c08c76'
BRANCH='test/nutcc-v0.1.2'
PROTECTED=['NutShellGPU_sim','NutShellGPU_hw','NutShellGPU_spec','examples']
REQUIRED_SUITES={'environment','build','compiler','upstream','protected_sources','manual','source_stability'}
ORDER=['environment','build','compiler','upstream','protected_sources','manual','source_stability']


def all_passed(results):
    return (len(results)==len(REQUIRED_SUITES) and {r['name'] for r in results}==REQUIRED_SUITES
            and all(r['status']=='PASS' for r in results))


def verify_source_unchanged(before,after):
    if before!=after:raise RunError('source files changed during validation; release forbidden')


def check_unittest_result(result):
    if result.get('tests',0)<=0 or any(result.get(k,1)!=0 for k in
            ('failures','errors','skipped','expected_failures','unexpected_successes')):
        raise RunError('unittest includes failed, skipped, missing or nonexecuted tests: '+str(result))


def optimization_coverage(root):
    sources=list(Path(root).glob('*/kernel.cpp'))
    if not sources:raise RunError('no end-to-end source cases executed')
    counts={}
    for level in ('O0','O1'):
        for source in sources:
            mode=source.parent/level
            for required in ('kernel.json','func/response.bin','cycle/response.bin'):
                if not (mode/required).is_file():
                    raise RunError(f'missing {level} end-to-end result: {mode/required}')
        counts[level]=dict(kernels=len(sources),backend_runs=2*len(sources))
    return counts


def execute_suites(actions):
    results=[]
    for name in ORDER:
        if name not in actions:
            results.append(dict(name=name,status='NOT_RUN',reason='mandatory action missing'));continue
        try:
            detail=actions[name]()
            results.append(dict(name=name,status='PASS',details=detail))
        except Exception as e:results.append(dict(name=name,status='FAIL',reason=type(e).__name__+': '+str(e)))
    return results


def finish_report(out,report):
    report['passed']=all_passed(report['results'])
    Path(out).mkdir(parents=True,exist_ok=True)
    path=Path(out)/'report.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    return 0 if report['passed'] else 1


def command(cmd,log,records,cwd=REPO,env=None,timeout=180,expected=0):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    record=dict(command=[str(x) for x in cmd],cwd=str(cwd),log=str(log),expected_exit=expected)
    records.append(record)
    try:r=subprocess.run(record['command'],cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
    except Exception as e:
        record['exit_code']=None;record['error']=str(e);log.write_text(str(e)+'\n');raise
    record['exit_code']=r.returncode;log.write_text(r.stdout+r.stderr)
    if r.returncode!=expected:raise RunError(f'command exited {r.returncode}, expected {expected}: {log}')
    return r.stdout


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def source_hashes():
    names=git('ls-files','--cached','--others','--exclude-standard','--','compiler',*PROTECTED).splitlines()
    return {n:hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in sorted(set(names)) if (REPO/n).is_file()}


def compiler_worker(path):
    suite=unittest.defaultTestLoader.discover(str(REPO/'compiler/tests'),top_level_dir=str(REPO))
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    data=dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
              expected_failures=len(r.expectedFailures),unexpected_successes=len(r.unexpectedSuccesses))
    Path(path).write_text(json.dumps(data,indent=2)+'\n')
    try:check_unittest_result(data);return 0
    except RunError:return 1


def run_gate(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    run=out/('run-'+uuid.uuid4().hex[:12]);run.mkdir()
    commands=[];tools={};before=source_hashes();initial_commit=git('rev-parse','HEAD')
    report=dict(format='nutcc-test-report-v1',git=dict(commit=initial_commit,branch=git('branch','--show-current'),
                dirty=bool(git('status','--porcelain'))),seed=20260924,os=platform.platform(),arch=platform.machine(),
                source_hashes=before,commands=commands,run_directory=str(run),counts={})
    env=dict(os.environ,NUTCC_TOOLS_DIR=str(run/'tools'),NUTCC_CASE_DIR=str(run/'cases'))
    def environment():
        if sys.version_info<(3,9):raise RunError('Python 3.9+ required')
        versions={name:command(cmd,run/(name+'.log'),commands) for name,cmd in
                  [('clang',[frontend_compiler(),'--version']),('cxx',[native_compiler(),'--version']),('python',[sys.executable,'--version'])]}
        if not supported_frontend(versions['clang'].splitlines()[0]):raise RunError('requires '+SUPPORTED_FRONTENDS)
        if report['git']['branch']!=BRANCH:raise RunError('requires testing branch '+BRANCH)
        report['versions']=versions
        report['toolchain']=dict(frontend=frontend_compiler(),native=native_compiler(),native_command=native_command())
        return versions
    def build():
        tools.update(build_tools(run/'tools'))
        if set(tools)!={'bridge','decode_check','oracle'}:raise RunError('missing native build target')
        stamp=json.loads((tools['bridge'].parent/'build.json').read_text())
        return dict(tools={n:str(p) for n,p in tools.items()},record=stamp)
    def compiler():
        path=run/'unittest.json'
        command([sys.executable,__file__,'--compiler-worker',str(path)],run/'compiler.log',commands,env=env,timeout=300)
        detail=json.loads(path.read_text());check_unittest_result(detail)
        report['counts']['compiler_tests']=detail['tests']
        coverage=optimization_coverage(run/'cases')
        report['counts']['e2e_by_optimization']=coverage
        report['counts']['e2e_source_cases']=coverage['O0']['kernels']
        report['counts']['e2e_kernels']=sum(v['kernels'] for v in coverage.values())
        report['counts']['e2e_backend_runs']=sum(v['backend_runs'] for v in coverage.values())
        detail['optimization_coverage']=coverage
        return detail
    def upstream():
        src=REPO/'NutShellGPU_sim';root=run/'upstream';root.mkdir()
        flags=native_command()+['-std=c++17','-O2','-frounding-math','-ffp-contract=off','-I'+str(src)]
        names=['t_decode','t_saxpy','t_diverge','t_runtime_v2','t_timing_v11','t_repairs','t_stats_time','cycle_builtin']
        def one(name):
            directory=root/name;directory.mkdir()
            source=src/('timing/main.cpp' if name=='cycle_builtin' else 'tests/'+name+'.cpp')
            objects=[str(tools['bridge'].parent/'func.o'),str(tools['bridge'].parent/'cycle.o')]
            executable=directory/name
            command(flags+[str(source),*objects,'-o',str(executable)],directory/'build.log',commands)
            command([str(executable)]+(['--test'] if name=='cycle_builtin' else []),directory/'test.log',commands,cwd=directory)
            return dict(name=name,status='PASS')
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,names))
        report['counts']['upstream_tests']=len(results)
        return results
    def protected():
        diff=git('diff','--name-only',BASE,'--',*PROTECTED)
        additions=git('ls-files','--others','--exclude-standard','--',*PROTECTED)
        if diff or additions:raise RunError('protected upstream changed: '+diff+' '+additions)
        return dict(baseline=BASE,changed_files=[])
    def manual():
        measurements={}
        for level in ('O0','O1'):
            root=run/'manual'/level;root.mkdir(parents=True);prefix=root/'vector_add'
            command([sys.executable,'compiler/nutcc.py','compiler/examples/vector_add.cpp','-'+level,'--kernel','vector_add','--out',str(prefix)],root/'compile.log',commands,env=env)
            meta=json.loads(prefix.with_suffix('.json').read_text())
            metrics=dict(static_instructions=meta['instruction_count'],regs_u32=meta['regs_u32'])
            for backend in ('func','cycle'):
                command([sys.executable,'compiler/examples/run_vector_add.py','--artifact',str(prefix)+'.json','--backend',backend,'--n','65'],root/(backend+'.log'),commands,env=env)
                response=(root/('run-'+backend)/'response.bin').read_bytes()
                if response[:8]!=b'NTOUT01\0':raise RunError('invalid manual response')
                instructions,cycles=struct.unpack_from('<QQ',response,8)
                metrics[backend]=dict(instructions=instructions,cycles=cycles)
            bad=root/'unsupported.cpp';bad.write_text((REPO/'compiler/examples/vector_add.cpp').read_text().replace('float','double'))
            old={p.name:p.read_bytes() for p in root.glob('vector_add.*')}
            command([sys.executable,'compiler/nutcc.py',str(bad),'-'+level,'--kernel','vector_add','--out',str(prefix)],root/'expected-rejection.log',commands,env=env,expected=2)
            if old!={p.name:p.read_bytes() for p in root.glob('vector_add.*')}:raise RunError('failed compile changed previous artifact')
            measurements[level]=metrics
        report['counts']['manual_checks']=8
        report['vector_add_n65']=measurements
        return dict(by_optimization=measurements,unsupported_type_rejected=True)
    def stability():
        verify_source_unchanged(before,source_hashes())
        if initial_commit!=git('rev-parse','HEAD') or report['git']['branch']!=git('branch','--show-current'):
            raise RunError('git revision changed during validation')
        return dict(unchanged=True,files=len(before))
    report['results']=execute_suites(dict(environment=environment,build=build,compiler=compiler,
                                        upstream=upstream,protected_sources=protected,manual=manual,source_stability=stability))
    report['artifact_hashes']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in run.rglob('*') if p.is_file() and p.suffix in ('.ntbin','.ntasm','.json')}
    code=finish_report(out,report)
    for result in report['results']:print(result['status']+' '+result['name']+(': '+result['reason'] if 'reason' in result else ''))
    print(('PASS' if code==0 else 'FAIL')+' report='+str(out/'report.json'))
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--all',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--compiler-worker',type=Path,help=argparse.SUPPRESS);a=p.parse_args()
    if a.compiler_worker:return compiler_worker(a.compiler_worker)
    if not a.all or not a.out:p.error('--all and --out are required')
    try:return run_gate(a.out)
    except Exception as e:
        finish_report(a.out,dict(format='nutcc-test-report-v1',results=[dict(name='environment',status='FAIL',reason=str(e))]))
        print('FAIL: '+str(e),file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
