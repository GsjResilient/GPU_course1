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
"""NTASM 1.2.0: explicit NTAS1 fields, with simulator capability boundaries."""
import argparse
from pathlib import Path
import re
import struct

OPS = {0: 'NOP', 1: 'MOV', 2: 'MOV32I', 3: 'S2R', 4: 'LDC', 5: 'CVTA', 6: 'SHFL', 7: 'VOTE', 8: 'PRMT', 9: 'SELP', 16: 'IADD', 17: 'IADD3', 18: 'IMAD', 19: 'IMUL', 20: 'ISUB', 21: 'IMNMX', 22: 'LOP', 23: 'SHF', 24: 'SHL', 25: 'SHR', 26: 'SAR', 27: 'BFE', 28: 'BFI', 29: 'POPC', 30: 'BREV', 31: 'IABS', 32: 'INEG', 33: 'IDIV', 34: 'IREM', 35: 'LOP3', 36: 'BMSK', 37: 'FIND', 48: 'FADD', 49: 'FSUB', 50: 'FMUL', 51: 'FFMA', 52: 'FMNMX', 53: 'FSET', 54: 'F2F', 55: 'XCVT', 56: 'RCP', 57: 'RSQ', 58: 'MUFU', 59: 'FRND', 60: 'FABS', 61: 'FNEG', 64: 'SETP', 65: 'SETPI', 80: 'BRA', 81: 'BRX', 82: 'CALL', 83: 'RET', 84: 'SSY', 85: 'BAR', 86: 'MEMBAR', 87: 'EXIT', 88: 'YIELD', 89: 'TRAP', 90: 'BRKPT', 96: 'LD', 97: 'ST', 98: 'LDU', 99: 'ATOM', 100: 'RED', 101: 'PREFETCH', 102: 'LD128', 103: 'ST128'}
# Rejected in both frontends: missing, simplified, or mismatched simulator semantics.
UNSUPPORTED = {
    'FCMP':'missing execution', 'DADD':'missing execution', 'PLOP':'missing execution', 'PSET2':'missing execution',
    'F2F':'incomplete half conversion', 'XCVT':'unsupported modes silently copy bits',
    'SETPI':'I format loses source register', 'LOP3':'I format loses source registers',
    'BRX':'incomplete divergent target ordering', 'RET':'divergent return is simplified',
    'CALL':'matching general return is unavailable', 'SHFL':'boundary/direction semantics differ from ISA',
    'PRMT':'source/selector encoding differs from ISA',
    'TRAP':'no exception effect', 'BRKPT':'no instruction breakpoint effect',
    'YIELD':'no scheduling hint implementation', 'PREFETCH':'no prefetch implementation',
    'MEMBAR':'scope ordering is not implemented',
}
OPS={op:name for op,name in OPS.items() if name not in UNSUPPORTED}

I_OPS = {'MOV32I', 'SETPI', 'TRAP', 'LOP3'}
B_OPS = {'BRA', 'CALL', 'SSY'}
MEM_OPS = {'LD', 'ST', 'LDU', 'PREFETCH', 'LD128', 'ST128'}
ZERO_OPS = {'NOP', 'EXIT', 'RET', 'YIELD', 'BRKPT', 'MEMBAR'}


def encode(ins):
    op, kind = ins['op'], ins['kind']
    d,a,c,x,y,b = (ins.get(k,0) for k in ('d','a','c','x','y','b'))
    imm=ins.get('imm',0)
    word=(op<<57)|(ins.get('guard',0)<<56)|(ins.get('negate',0)<<55)|(ins.get('gp',0)<<52)|(ins.get('m',0)<<48)
    if kind in ('R','M'):word |= d<<40 | a<<32 | c<<24 | x<<16 | y<<8 | b
    elif kind=='I':word |= d<<40 | (imm&0xffffffff)<<8 | x
    elif kind=='B':word |= (imm&0x3ffffff)<<22 | x<<16
    elif kind=='MI':word |= d<<40 | a<<32 | x<<24 | y<<16 | (imm&0xffff)
    else:raise ValueError('invalid instruction format')
    return word


def decode(word):
    op=word>>57; name=OPS[op]
    kind='I' if name in I_OPS else 'B' if name in B_OPS else 'MI' if name=='LDC' else ('M' if word&(1<<23) else 'MI') if name in MEM_OPS else 'R'
    ins=dict(op=op,kind=kind,guard=(word>>56)&1,negate=(word>>55)&1,gp=(word>>52)&7,m=(word>>48)&15,d=0,a=0,c=0,x=0,y=0,b=0,imm=0)
    def signed(v,n):return v-(1<<n) if v&(1<<(n-1)) else v
    if kind in ('R','M'):
        for k,shift in zip(('d','a','c','x','y','b'),(40,32,24,16,8,0)):ins[k]=(word>>shift)&255
    elif kind=='I':ins.update(d=(word>>40)&255,imm=(word>>8)&0xffffffff,x=word&255)
    elif kind=='B':ins.update(imm=signed((word>>22)&0x3ffffff,26),x=(word>>16)&63)
    else:ins.update(d=(word>>40)&255,a=(word>>32)&255,x=(word>>24)&255,y=(word>>16)&255,imm=signed(word&65535,16))
    return ins


def validate(ins):
    name=OPS[ins['op']];k=ins['kind'];d=ins['d'];a=ins['a'];c=ins['c'];x=ins['x'];y=ins['y']
    def pair(r,words=2):
        if r!=255 and (r%2 or r+words>255):raise ValueError('unaligned/out-of-range register group')
    if name=='S2R' and x not in {0,1,2,*range(8,15),*range(16,23),24,32,33,34,41}:raise ValueError('invalid special register')
    if name=='IADD3' and x:raise ValueError('unused x must be zero')
    if name=='CVTA':
        pair(d);pair(a)
        if x&7>3 or x&0xf0:raise ValueError('invalid CVTA space/mode')
    if name in ('IADD','ISUB') and x&7==6:pair(d);pair(a);pair(c)
    if name in ('IMUL','IMAD') and (x>>3)&7==2:pair(d)
    if name=='S2R' and x in (32,33,34):pair(d)
    if ins.get('m',0) and (name not in ('FADD','FSUB','FMUL','FFMA') or ins['m']>3):raise ValueError('unsupported header modifier')
    if name in ('ATOM','RED') and ((x>>3)&15!=4 or x&7 not in (0,1,2) or y>11):raise ValueError('atomics require U32 global/shared/local, operation 0..11')
    if name=='MOV32I' and x&1:pair(d)
    if name=='LDC':
        if x&0x90 or x&15>7 or (x>>5)&3==3:raise ValueError('invalid LDC bank/width')
        if (x>>5)&3==1:pair(d)
    if name in MEM_OPS:
        width=(x>>3)&15
        if x&7>4 or width>10 or y>4:raise ValueError('invalid memory space/width/hint bits')
        if width in (6,8,10) or name in ('LD128','ST128'):
            pair(d,4 if width==10 or name in ('LD128','ST128') else 2);pair(a)
            if k=='M':pair(c)
    if name=='BAR' and (x&0xc0 or x&0x30==0x30):raise ValueError('invalid barrier mode/id bits')
    zero_x={'NOP','MOV','SHL','SHR','SAR','BFI','POPC','BREV','IABS','INEG','IDIV','IREM','BMSK','FADD','FSUB','FMUL','FFMA','RCP','RSQ','FABS','FNEG','EXIT'}
    if name in zero_x and x:raise ValueError('unsupported x modifier')
    ranges={'MOV32I':1,'VOTE':3,'IMNMX':3,'LOP':7,'BFE':1,'FIND':2,'FMNMX':1,'MUFU':3,'FRND':3,'BRA':1,'SSY':1}
    if name in ranges and x>ranges[name]:raise ValueError('unsupported x modifier')
    if name in ('IADD','ISUB') and x not in (0,6):raise ValueError('unsupported integer width')
    if name in ('IMUL','IMAD') and x not in (0,1,8,9,16,17):raise ValueError('unsupported multiply mode')
    if name=='SETP' and (d>6 or x>>4>2 or ((x>>4)<2 and (x&15)>9) or y>3):raise ValueError('unsupported predicate type/comparison')
    if name=='FSET' and (x>15 or y>3):raise ValueError('unsupported float comparison')
    if name=='SELP' and (x or y>15):raise ValueError('unsupported select mode')
    if name=='VOTE' and y>7:raise ValueError('invalid vote predicate')
    if name not in {'IADD3','VOTE','SELP','SETP','FSET','ATOM','RED',*MEM_OPS} and y:raise ValueError('unused y must be zero')
    if ins.get('b',0) and name!='ATOM' and name!='RED':raise ValueError('unused b must be zero')
    if name in ('ATOM','RED') and y!=8 and ins.get('b',0):raise ValueError('b is reserved for atomic CAS')



def parse(source):
    rows=[];labels={}
    for line,text in enumerate(source.splitlines(),1):
        text=text.split('//',1)[0].strip()
        if not text:continue
        if ':' in text:
            label,text=map(str.strip,text.split(':',1))
            if not re.fullmatch(r'[A-Za-z_]\w*',label) or label in labels:raise ValueError(f'line {line}: invalid/duplicate label')
            labels[label]=len(rows)
        if text:rows.append((line,text))
    result=[];reverse={v:k for k,v in OPS.items()}
    for index,(line,text) in enumerate(rows):
        try:
            ins=dict(kind='R',d=0,a=0,c=0,x=0,y=0,b=0,imm=0,guard=0,negate=0,gp=0,m=0)
            if text.startswith('@'):
                pred,text=text.split(None,1);match=re.fullmatch(r'@(!?)(P[0-6]|PT)',pred)
                if not match:raise ValueError('expected @P0..@P6 or @PT')
                ins.update(guard=1,negate=int(bool(match[1])),gp=7 if match[2]=='PT' else int(match[2][1]))
            parts=text.split(None,1);mnemonic=parts[0].split('.m');name=mnemonic[0]
            if len(mnemonic)>1:ins['m']=int(mnemonic[1],0)
            if len(mnemonic)>2 or not 0<=ins['m']<=15:raise ValueError('invalid header modifier')
            if name in UNSUPPORTED:raise ValueError(f'{name}: unsupported simulator capability ({UNSUPPORTED[name]})')
            if name not in reverse:raise ValueError('unknown opcode')
            ins['op']=reverse[name]
            fields=[s.strip() for s in parts[1].split(',')] if len(parts)>1 else []
            def reg(s):
                if s=='RZ':return 255
                if not re.fullmatch(r'R\d+',s) or not 0<=int(s[1:])<=255:raise ValueError('invalid register')
                return int(s[1:])
            def number(s,lo=0,hi=255):
                n=int(s,0)
                if not lo<=n<=hi:raise ValueError('field out of range')
                return n
            if name in ZERO_OPS and not fields:pass
            elif name in B_OPS:
                if len(fields) not in (1,2):raise ValueError('expected target[, x]')
                x=number(fields[1],0,63) if len(fields)==2 else 0
                target=labels.get(fields[0])
                if target is None:raise ValueError('unknown branch target')
                imm=target if x&1 else target-index
                if not -(1<<25)<=imm<(1<<25):raise ValueError('branch offset out of range')
                ins.update(kind='B',imm=imm,x=x)
            elif name in I_OPS:
                if len(fields)!=3:raise ValueError('expected Rd, imm32, x')
                ins.update(kind='I',d=reg(fields[0]),imm=number(fields[1],-(1<<31),(1<<32)-1),x=number(fields[2]))
            elif name in MEM_OPS or name=='LDC':
                if len(fields)!=4:raise ValueError('expected Rd, [Ra+offset], x, y')
                addr=re.fullmatch(r'\[(R\d+|RZ)(?:([+-])((?:R\d+|RZ)|(?:0x[0-9a-fA-F]+|\d+)))?\]',fields[1].replace(' ',''))
                if not addr:raise ValueError('invalid address')
                x=number(fields[2]);y=number(fields[3]);offset=addr[3] or '0'
                ins.update(d=reg(fields[0]),a=reg(addr[1]),x=x,y=y)
                if offset.startswith('R'):
                    if name=='LDC' or addr[2]=='-':raise ValueError('register offset is unavailable here')
                    ins.update(kind='M',c=reg(offset),x=x|128)
                else:
                    if name!='LDC' and x&128:raise ValueError('MI requires x[7]=0')
                    ins.update(kind='MI',imm=number(('-' if addr[2]=='-' else '')+offset,-32768,32767))
            else:
                if len(fields) not in (5,6):raise ValueError('expected Rd, Ra, Rc, x, y[, b]')
                ins.update(d=reg(fields[0]),a=reg(fields[1]),c=reg(fields[2]),x=number(fields[3]),y=number(fields[4]),b=(reg(fields[5]) if fields[5].startswith('R') else number(fields[5])) if len(fields)==6 else 0)
            validate(ins);result.append(ins)
        except (ValueError,KeyError) as e:raise ValueError(f'line {line}: {e}') from e
    return result


def assemble(source):
    return b''.join(struct.pack('<Q',encode(ins)) for ins in parse(source))


def render(meta):
    lines=['// NTAS1 explicit fields; .mN is the header modifier.']
    for row,ins in zip(meta['line_table'],meta['instructions']):
        pc=row['pc'];name=OPS[ins['op']];kind=ins['kind']
        d,a,c=(f"R{ins[k]}" for k in ('d','a','c'));x,y=ins['x'],ins['y']
        prefix=('@'+('!' if ins['negate'] else '')+('PT' if ins.get('gp',0)==7 else 'P'+str(ins.get('gp',0)))+' ') if ins['guard'] else ''
        if kind=='I':operands=f"{d}, 0x{ins['imm']&0xffffffff:08x}, {x}"
        elif kind=='B':operands=f"L{(ins['imm']*8 if x&1 else pc+8*ins['imm']):06x}, {x}"
        elif kind=='MI':operands=f"{d}, [{a}{ins['imm']:+d}], {x}, {y}"
        elif kind=='M':operands=f"{d}, [{a}+{c}], {x}, {y}"
        else:operands=f"{d}, {a}, {c}, {x}, {y}, {ins.get('b',0)}"
        if ins.get('m',0):name+='.m'+str(ins['m'])
        lines.append(f"L{pc:06x}: {prefix}{name} {operands} // source line {row['line']}")
    lines.append(f"L{len(meta['instructions'])*8:06x}:")
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('-o',required=True)
    a=p.parse_args();binary=assemble(Path(a.source).read_text(encoding='utf-8'))
    Path(a.o).write_bytes(binary);print(f'assembled {len(binary)//8} instructions, {len(binary)} bytes -> {a.o}')
