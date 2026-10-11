# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
"""NTDB 1.2.0: source debugger for simulator 1.2.0. Author: Tonghui Ming."""
import argparse
import json
from pathlib import Path
import shlex
import struct
from ntruntime import Session, prepare, collect

__author__ = 'Tonghui Ming'


class Debugger:
    def __init__(self, session):
        self.s = session
        self.sm, self.warp, self.lane = 0, 0, 0
        self.line_by_pc = {r['pc']: r['line'] for r in session.meta['line_table']}
        self.breakpoints = {}

    def state(self):
        return self.s.command(f'state {self.sm} {self.warp} {self.lane}')

    def breakpoint(self, line):
        # Break at each contiguous range entry, including separate loop-condition ranges.
        pcs = [pc for pc, ln in self.line_by_pc.items() if ln == line and self.line_by_pc.get(pc-8) != line]
        if not pcs:
            raise ValueError('no executable instruction at that source line')
        for pc in pcs:
            self.s.command(f'break {pc}')
        self.breakpoints[line] = pcs
        return {'line': line, 'pcs': pcs}

    def run(self, step=False):
        e = self.s.command('step 1' if step else 'run 10000000')
        if 'pc' in e:
            self.sm, self.warp = e['sm'], e['warp']
            e['line'] = self.line_by_pc.get(e['pc'])
        return e

    def source_step(self):
        # Global scheduler stays fair. Stop when the selected warp reaches another
        # line, any breakpoint fires, the selected warp exits, or the budget ends.
        focus = self.sm, self.warp
        initial = self.state()
        start = self.line_by_pc.get(initial['pc'])
        for _ in range(100000):
            e = self.s.command('step 1')
            if e['reason'] in ('completed', 'breakpoint'):
                if 'pc' in e:
                    self.sm,self.warp=e['sm'],e['warp'];e['line']=self.line_by_pc.get(e['pc'])
                return e
            self.sm, self.warp = focus
            try:
                state = self.state()
            except RuntimeError:
                return {'reason':'focus_exited'}
            if state['cta'] != initial['cta'] or state['warp_in_cta'] != initial['warp_in_cta']:
                return {'reason':'focus_exited'}
            if self.line_by_pc.get(state['pc']) != start:
                return {'reason':'source_step', 'sm':self.sm, 'warp':self.warp, 'pc':state['pc'], 'line':self.line_by_pc.get(state['pc'])}
        return {'reason':'budget'}

    def variable(self, name):
        state = self.state(); pc = state['pc']
        candidates = [s for s in self.s.meta['symbols'] if s['name'] == name and s['start_pc'] <= pc < s['end_pc']]
        if not candidates:
            raise ValueError('variable unavailable: not initialized, outside scope, or unknown')
        sym = candidates[-1]; r = sym['reg']; raw = state['registers'][r]; typ = sym['type']
        if sym.get('mask_reg') is not None and not state['registers'][sym['mask_reg']]:
            raise ValueError('variable unavailable in this lane: declaration was predicated off')
        if typ.endswith('*'):
            value = raw | (state['registers'][r+1] << 32)
        elif typ == 'float':
            value = struct.unpack('<f', struct.pack('<I', raw))[0]
        elif typ == 'int':
            value = struct.unpack('<i', struct.pack('<I', raw))[0]
        else:
            value = raw
        live = state['registers'][self.s.meta['live_mask_register']]
        return {'name':name, 'type':typ, 'register':r, 'value':value, 'lane':self.lane, 'active':bool(live and state['active_mask'] & (1 << self.lane))}

    def command(self, text):
        args = shlex.split(text)
        if not args:
            return None
        cmd = args[0]
        if cmd in ('break', 'b') and len(args)==2:
            return self.breakpoint(int(args[1]))
        if cmd == 'delete' and len(args)==2:
            for pc in self.breakpoints.pop(int(args[1])):
                self.s.command(f'delete {pc}')
            return {'deleted':int(args[1])}
        if cmd in ('continue','c'):
            return self.run()
        if cmd in ('stepi','si'):
            return self.run(True)
        if cmd in ('step','s'):
            return self.source_step()
        if cmd == 'lane' and len(args)==2:
            old = self.lane; self.lane=int(args[1])
            try: self.state()
            except Exception: self.lane=old; raise
            return {'lane':self.lane}
        if cmd == 'warp' and len(args)==3:
            old=self.sm,self.warp;self.sm,self.warp=map(int,args[1:])
            try:self.state()
            except Exception:self.sm,self.warp=old;raise
            return {'sm':self.sm,'warp':self.warp}
        if cmd in ('print','p') and len(args)==2:
            return self.variable(args[1])
        if cmd == 'warps':
            return self.s.command('warps')
        if cmd in ('regs','stack','where'):
            state=self.state()
            if cmd=='regs':return {'registers':state['registers'],'predicates':state['predicates']}
            if cmd=='stack':return {'stack':state['stack'],'active_mask':state['active_mask']}
            line=self.line_by_pc.get(state['pc']);source=self.s.meta['source_text'].splitlines()
            return {'pc':state['pc'],'line':line,'text':source[line-1] if line and line<=len(source) else '<exit>'}
        if cmd == 'memory' and len(args)==3:
            return self.s.command(f'read {int(args[1],0)} {int(args[2],0)}')
        if cmd == 'disasm':
            return [{'pc':entry['pc'],'line':entry['line'],**ins} for entry,ins in zip(self.s.meta['line_table'],self.s.meta['instructions'])]
        if cmd == 'help':
            return 'break LINE; delete LINE; continue; stepi; step; where; warps; warp SM SLOT; lane N; print NAME; regs; stack; memory ADDRESS BYTES; disasm; quit'
        raise ValueError('unknown command or invalid arguments; use help')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',action='version',version='NTDB 1.2.0 | Author: '+__author__)
    p.add_argument('module');p.add_argument('launch');p.add_argument('--commands',help='batch command file')
    a=p.parse_args()
    with Session(a.module) as s:
        buffers=prepare(s,json.loads(Path(a.launch).read_text(encoding='utf-8')))
        d=Debugger(s)
        if a.commands:
            for line in Path(a.commands).read_text(encoding='utf-8').splitlines():
                if line.strip()=='quit':break
                if not line.strip() or line.lstrip().startswith('#'):continue
                print(json.dumps({'command':line,'result':d.command(line)},ensure_ascii=False))
        else:
            print('NTDB 1.2.0 / simulator 1.2.0 / Author: '+__author__+'. Type help for commands.')
            while True:
                try:
                    line=input('(ntdb) ')
                    if line.strip()=='quit':break
                    print(json.dumps(d.command(line),ensure_ascii=False))
                except (EOFError,KeyboardInterrupt):break
                except (ValueError,RuntimeError,KeyError) as e:print('error:',e)


if __name__=='__main__':
    main()
