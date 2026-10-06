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
from dataclasses import dataclass
from .diagnostics import CompileError


@dataclass
class Allocation:
    registers: dict
    regs_u32: int


def allocate(program,limit=255):
    intervals={}
    for index,ins in enumerate(program.instructions):
        for reg in ([ins.dst] if ins.dst else [])+list(ins.args)+([ins.guard] if ins.guard else []):
            if reg.id not in intervals:intervals[reg.id]=[index,index]
            intervals[reg.id][1]=index
    active=[];mapping={};free=set(range(2,limit));highest=1
    for id_,(first,last) in sorted(intervals.items(),key=lambda pair:(pair[1][0],pair[0])):
        keep=[]
        for end,reg in active:
            if end<first:free.add(reg)
            else:keep.append((end,reg))
        active=keep
        if not free:
            raise CompileError('E_REGISTERS',f'register allocation exceeds {limit} slots; spilling is unsupported',program.instructions[first].loc)
        physical=min(free);free.remove(physical)
        mapping[id_]=physical;highest=max(highest,physical)
        active.append((last,physical))
    return Allocation(mapping,highest+1)
