# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
"""Build the native transport. C++17 compiler or Zig is required."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SIM = PROJECT / 'NutShellGPU_sim'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cxx', help='C++ compiler or zig executable')
    p.add_argument('--tests', action='store_true', help='also build and run baseline C++ regressions')
    p.add_argument('--model-runner', action='store_true', help='build inherited fixture runner')
    p.add_argument('--build-dir', help='output/cache directory (default: project/build/toolchain)')
    args = p.parse_args()
    cxx = args.cxx or shutil.which('c++') or shutil.which('clang++') or shutil.which('g++') or shutil.which('zig')
    if not cxx:
        p.error('set --cxx to a C++17 compiler or Zig executable')
    cxx = shutil.which(cxx) or str(Path(cxx).resolve())
    build = Path(args.build_dir).resolve() if args.build_dir else PROJECT / 'build/toolchain'
    build.mkdir(parents=True,exist_ok=True)
    env = os.environ.copy()
    env['ZIG_GLOBAL_CACHE_DIR'] = str(build / 'zig-global')
    env['ZIG_LOCAL_CACHE_DIR'] = str(build / 'zig-local')
    command = [cxx] + (['c++'] if Path(cxx).stem == 'zig' else [])
    flags = ['-std=c++17', '-O2', '-frounding-math', '-ffp-contract=off', '-fno-strict-aliasing', '-I'+str(SIM)]
    suffix = '.exe' if os.name == 'nt' else ''
    targets = [('ntsim', [ROOT/'native/ntsim.cpp', SIM/'func/func_sim.cpp', SIM/'timing/cycle_sim.cpp'])]
    if args.model_runner:
        targets.append(('model_runner',[SIM/'tests/model_runner.cpp',SIM/'func/func_sim.cpp',SIM/'timing/cycle_sim.cpp']))
    if args.tests:
        for name in ('t_decode', 't_saxpy', 't_diverge', 't_runtime_v2', 't_timing_v11', 't_repairs', 't_stats_time'):
            targets.append((name, [SIM/'tests'/f'{name}.cpp', SIM/'func/func_sim.cpp', SIM/'timing/cycle_sim.cpp']))
    for name, sources in targets:
        output = build / (name+suffix)
        subprocess.run(command+flags+[str(s) for s in sources]+['-o',str(output)], check=True, env=env, cwd=ROOT)
        print('Built', output, flush=True)
        if name not in ('ntsim','model_runner'):
            subprocess.run([str(output)], check=True, cwd=build)


if __name__ == '__main__':
    main()
