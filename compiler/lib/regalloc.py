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
