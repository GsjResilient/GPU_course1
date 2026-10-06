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
import argparse
import json
from pathlib import Path
from ntruntime import Session, prepare, collect


def main():
    p = argparse.ArgumentParser(description='Run an NTCC module with a JSON host launch description')
    p.add_argument('--version', action='version', version='ntrun 1.2.0')
    p.add_argument('module'); p.add_argument('launch')
    p.add_argument('--backend', choices=['func', 'cycle'], default='func')
    p.add_argument('--budget', type=int, default=10000000)
    p.add_argument('-o', '--output')
    a = p.parse_args()
    with Session(a.module, a.backend) as s:
        buffers = prepare(s, json.loads(Path(a.launch).read_text(encoding='utf-8')))
        event = s.synchronize(a.budget)
        result = {'execution': event, 'buffers': collect(s, buffers)}
    text = json.dumps(result, indent=2)
    if a.output:
        Path(a.output).write_text(text+'\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
