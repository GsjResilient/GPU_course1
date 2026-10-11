# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
from dataclasses import dataclass
from .ntas import encode_i,encode_r,encode_mi
from .regalloc import allocate
from .diagnostics import CompileError


@dataclass
class MachineProgram:
    name: str
    words: list
    listing: list
    params: list
    param_size: int
    regs_u32: int


# These expansions preserve P0 and the active-lane mask. New operations must be
# reviewed before joining this set; comparisons use P1, then a GPR destination.
P0_PRESERVING_OPS=frozenset(('const','param','sr','copy','add','sub','mul','and','or','xor',
                            'shl','shr','cmp_eq','cmp_ne','cmp_lt','cmp_le','cmp_gt','cmp_ge',
                            'load','store'))


def generate(program,opt_level=1):
    if type(opt_level) is not int or opt_level not in (0,1):
        raise CompileError('E_OPT_LEVEL','optimization level must be 0 or 1',program.instructions[-1].loc)
    allocation=allocate(program)
    words=[];listing=[]
    def add(word,text,loc):
        if len(words)>=4096:
            raise CompileError('E_CODE_SIZE','machine program exceeds 4096 instructions',loc)
        listing.append(f'{len(words)*8:08x}  {word:016x}  {text}')
        words.append(word)
    reg=lambda v:allocation.registers[v.id]
    loc=program.instructions[-1].loc
    add(encode_i(2,0,1),'MOV32I R0, 1',loc)
    cached_guard=None
    for ins in program.instructions:
        loc=ins.loc
        guard=None
        reusable=opt_level==1 and ins.op in P0_PRESERVING_OPS
        if not reusable or ins.guard is None:
            cached_guard=None
        if ins.guard is not None:
            key=(ins.guard.id,reg(ins.guard))
            if cached_guard!=key:
                add(encode_r(0x40,0,reg(ins.guard),255,x=0x11,y=0x83),f'SETP.NE.U32 P0, R{reg(ins.guard)}, RZ',loc)
            cached_guard=key if reusable else None
            guard=(0,False)
        d=reg(ins.dst) if ins.dst else 0
        args=[reg(v) for v in ins.args]
        a=args[0] if args else 0;c=args[1] if len(args)>1 else 0
        op=ins.op
        prefix='@P0 ' if guard else ''
        def r(opcode,x=0,y=0):
            add(encode_r(opcode,d,a,c,x=x,y=y,guard=guard),prefix+f'{op.upper()} R{d}, R{a}, R{c}',loc)
        if op=='const':add(encode_i(2,d,ins.imm,guard=guard),prefix+f'MOV32I R{d}, 0x{ins.imm:08x}',loc)
        elif op=='param':add(encode_mi(4,d,0,0,0,ins.imm,guard),f'LDC.PARAM.U32 R{d}, [{ins.imm}]',loc)
        elif op=='sr':add(encode_r(3,d,x=ins.imm,guard=guard),prefix+f'S2R R{d}, SR[{ins.imm}]',loc)
        elif op=='copy':r(1)
        elif op in ('add','sub','mul'):
            fp=ins.dst.kind=='f32'
            codes={'add':0x30 if fp else 0x10,'sub':0x31 if fp else 0x14,'mul':0x32 if fp else 0x13}
            x=0 if fp else (int(ins.dst.kind=='i32') if op=='mul' else 4)
            r(codes[op],x)
        elif op in ('and','or','xor'):r(0x16,{'and':0,'or':1,'xor':2}[op])
        elif op in ('shl','shr'):r(0x18 if op=='shl' else 0x19)
        elif op.startswith('cmp_'):
            typ={'i32':0,'u32':1,'u8':1,'bool':1,'f32':2}[ins.args[0].kind]
            cmp=('eq','ne','lt','le','gt','ge').index(op[4:])
            add(encode_r(0x40,1,a,c,x=(typ<<4)|cmp,y=0x83,guard=guard),prefix+f'SETP.{op[4:].upper()}.{ins.args[0].kind} P1, R{a}, R{c}',loc)
            add(encode_r(9,d,0,255,y=1,guard=guard),prefix+f'SELP R{d}, R0, RZ, P1',loc)
        elif op in ('load','store'):
            width={'u8':0,'i32':4,'u32':4,'bool':4,'f32':7}[ins.dst.kind if op=='load' else ins.args[1].kind]
            data=d if op=='load' else c
            add(encode_mi(0x60 if op=='load' else 0x61,data,a,width<<3,0,0,guard),prefix+f'{op.upper()}.{width} R{data}, [R{a}]',loc)
        elif op=='exit':add(encode_r(0x57,0),'EXIT',loc)
        else:raise CompileError('E_BACKEND','unknown IR operation: '+op,loc)
        # Even a predicated partial write changes the cached mask's value.
        if cached_guard is not None and ins.dst is not None and d==cached_guard[1]:
            cached_guard=None
    return MachineProgram(program.name,words,listing,program.params,program.param_size,allocation.regs_u32)
