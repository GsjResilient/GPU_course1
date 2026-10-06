# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
"""NTCC 1.2.0: .ntcu teaching language -> NTAS1. Author: Tonghui Ming."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

VERSION = '1.2.0'
__author__ = 'Tonghui Ming'


class CompileError(Exception):
    pass


class Parser:
    pattern = re.compile(r'(?P<skip>\s+|//[^\n]*|/\*[\s\S]*?\*/)|(?P<string>"(?:\\.|[^"\\])*")|(?P<num>(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fF]?|\d+[eE][+-]?\d+[fF]?|0[xX][0-9a-fA-F]+[uU]?|\d+[uU]?)|(?P<id>[A-Za-z_]\w*)|(?P<op><<=|>>=|\+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|&&|\|\||==|!=|<=|>=|[{}()\[\];:,.+*\-/=<>%&|^!~])')

    def __init__(self, source):
        self.tokens = []
        pos, line = 0, 1
        while pos < len(source):
            m = self.pattern.match(source, pos)
            if not m:
                raise CompileError(f'line {line}: unsupported character {source[pos]!r}')
            if m.lastgroup != 'skip':
                self.tokens.append((m.group(), line))
            line += m.group().count('\n')
            pos = m.end()
        self.tokens.append(('<eof>', line))
        self.i = 0

    def peek(self):
        return self.tokens[self.i][0]

    def take(self, expected=None):
        t = self.tokens[self.i]
        if expected and t[0] != expected:
            raise CompileError(f'line {t[1]}: expected {expected!r}, got {t[0]!r}')
        self.i += 1
        return t

    def ident(self):
        t, line = self.take()
        if not re.fullmatch(r'[A-Za-z_]\w*', t) or t in {'if', 'else', 'while', 'for', 'break', 'continue', 'asm', 'return', 'void', 'int', 'uint', 'float', '__global__'}:
            raise CompileError(f'line {line}: expected identifier')
        return t

    def typ(self):
        t, line = self.take()
        if t not in ('int', 'uint', 'float'):
            raise CompileError(f'line {line}: supported types: int, uint, float and their pointers')
        if self.peek() == '*':
            self.take('*')
            t += '*'
        return t

    def parse(self):
        self.take('__global__'); self.take('void')
        name = self.ident()
        self.take('(')
        params = []
        while self.peek() != ')':
            line = self.tokens[self.i][1]
            t = self.typ(); n = self.ident()
            params.append((n, t, line))
            if self.peek() != ',':
                break
            self.take(',')
        self.take(')')
        body = self.block()
        self.take('<eof>')
        return name, params, body

    def body(self):
        return self.block() if self.peek() == '{' else [self.statement()]

    def block(self):
        self.take('{')
        result = []
        while self.peek() != '}':
            result.append(self.statement())
        self.take('}')
        return result

    def statement(self):
        k, line = self.tokens[self.i]
        if k == '{':
            return 'scope', line, self.block()
        if k == ';':
            self.take(); return 'scope', line, []
        if k == 'for':
            self.take(); self.take('(')
            init=[] if self.peek()==';' else [self.simple()]
            self.take(';')
            cond=('literal',line,'1') if self.peek()==';' else self.expr()
            self.take(';')
            update=[] if self.peek()==')' else [self.simple()]
            self.take(')')
            return 'scope',line,init+[('while',line,cond,self.body(),update)]
        if k in ('if', 'while'):
            self.take(); self.take('('); cond = self.expr(); self.take(')')
            body = self.body(); other = []
            if k == 'if' and self.peek() == 'else':
                self.take(); other = self.body()
            return k, line, cond, body, other
        if k in ('return', 'break', 'continue'):
            self.take(); self.take(';')
            return k, line
        if k == 'asm':
            self.take(); self.take('(')
            template=''
            while self.peek().startswith('"'):
                template+=ast.literal_eval(self.take()[0])
            outputs=[];inputs=[]
            for items in (outputs,inputs):
                if self.peek()!=':':break
                self.take(':')
                while self.peek() not in (':',')'):
                    constraint=ast.literal_eval(self.take()[0])
                    self.take('(');exprs=[self.expr()]
                    while self.peek()==',':
                        self.take(',');exprs.append(self.expr())
                    self.take(')')
                    items.append((constraint,exprs))
                    if self.peek()!=',':break
                    self.take(',')
            self.take(')');self.take(';')
            return 'asm',line,template,outputs,inputs
        result=self.simple();self.take(';');return result

    def simple(self):
        k,line=self.tokens[self.i]
        if k in ('int','uint','float'):
            typ=self.typ();name=self.ident();self.take('=');expr=self.expr()
            return 'decl',line,name,typ,expr
        lhs=self.expr()
        if self.peek() in ('=','+=','-=','*=','/=','%=','&=','|=','^=','<<=','>>='):
            op=self.take()[0];rhs=self.expr()
            return 'assign',line,lhs,rhs,op
        return 'exprstmt',line,lhs

    def expr(self, minimum=0):
        t, line = self.take()
        if t == '(':
            if self.peek() in ('int','uint','float'):
                typ=self.typ();self.take(')');lhs=('cast',line,typ,self.expr(30))
            else:
                lhs = self.expr(); self.take(')')
        elif t in ('-','+','!','~','++','--'):
            lhs = ('unary', line, t, self.expr(30))
        elif re.match(r'\d|\.', t):
            lhs = ('literal', line, t)
        elif re.fullmatch(r'[A-Za-z_]\w*', t):
            lhs = ('var', line, t)
            if self.peek() == '.':
                self.take(); lhs = ('builtin', line, t + '.' + self.ident())
            elif self.peek() == '(':
                self.take(); args = []
                while self.peek() != ')':
                    args.append(self.expr())
                    if self.peek() != ',':
                        break
                    self.take(',')
                self.take(')'); lhs = ('call', line, t, args)
        else:
            raise CompileError(f'line {line}: expected expression, got {t!r}')
        if self.peek() == '[':
            self.take(); idx = self.expr(); self.take(']')
            lhs = ('index', line, lhs, idx)
        if self.peek() in ('++','--'):
            lhs=('postfix',line,self.take()[0],lhs)
        precedence = {'||':1,'&&':2,'|':3,'^':4,'&':5,'==':6,'!=':6,'<':7,'<=':7,'>':7,'>=':7,'<<':8,'>>':8,'+':10,'-':10,'*':20,'/':20,'%':20}
        while precedence.get(self.peek(), -1) >= minimum:
            op = self.take()[0]
            lhs = ('binary', line, op, lhs, self.expr(precedence[op] + 1))
        return lhs


class Compiler:
    def __init__(self):
        self.code, self.map, self.ir = [], [], []
        self.symbols, self.env, self.params = [], {}, []
        self.next_reg, self.depth = 0, 0
        self.active = None
        self.live = None
        self.guard_temp = None
        self.loops = []
        self.has_return = False

    def fail(self, line, message):
        raise CompileError(f'line {line}: {message}')

    def reg(self, pair=False):
        if pair and self.next_reg % 2:
            self.next_reg += 1
        r = self.next_reg
        self.next_reg += 2 if pair else 1
        if self.next_reg > 254:
            raise CompileError('register budget exceeds R0..R253; split the kernel (no spilling in 1.2.0)')
        return r

    def enabled(self, line):
        masks=list(dict.fromkeys([r for r in [self.active,self.live,*[loop[1] for loop in self.loops]] if r is not None]))
        if not masks:return None
        enabled=masks[0]
        for mask in masks[1:]:
            self.raw_emit(0x13,line,self.guard_temp,enabled,mask);enabled=self.guard_temp
        return enabled

    def emit(self, op, line, *args, **kwargs):
        enabled=self.enabled(line)
        if enabled is not None:
            self.raw_emit(0x40, line, a=enabled, c=255, x=0x11, y=3)
            kwargs['guard'] = 1
        return self.raw_emit(op, line, *args, **kwargs)

    def raw_emit(self, op, line, d=0, a=0, c=0, x=0, y=0, kind='R', imm=0, guard=0, negate=0, gp=0, m=0, b=0):
        from ntasm import encode, validate
        ins=dict(op=op,d=d,a=a,c=c,x=x,y=y,b=b,m=m,gp=gp,kind=kind,imm=imm,guard=guard,negate=negate)
        validate(ins)
        self.code.append(encode(ins))
        self.map.append({'pc':len(self.map)*8,'line':line})
        self.ir.append(ins)
        return len(self.code)-1

    def patch(self, at, target):
        rel = target-at
        self.code[at] = (self.code[at] & ~(((1 << 26)-1) << 22)) | ((rel & 0x3ffffff) << 22)
        self.ir[at]['imm'] = rel

    def literal(self, value, typ, line):
        r = self.reg()
        bits = struct.unpack('<I', struct.pack('<f', value))[0] if typ == 'float' else value
        self.emit(2, line, d=r, kind='I', imm=bits)
        return r, typ

    def compatible(self, a, b, line):
        if a == b:
            return a
        if {a, b} <= {'int', 'uint'}:
            return 'uint'
        self.fail(line, f'type mismatch: {a} and {b}; implicit floating conversions are unsupported')

    def address(self, e):
        _, line, base, index = e
        b, typ = self.expr(base)
        if not typ.endswith('*'):
            self.fail(line, 'subscript needs a pointer')
        i, it = self.expr(index)
        if it not in ('int', 'uint'):
            self.fail(line, 'subscript needs an integer')
        four, _ = self.literal(4, 'uint', line)
        off = self.reg(True)
        self.emit(0x13, line, off, i, four, x=0x10 | (it == 'int'))
        addr = self.reg(True)
        self.emit(0x10, line, addr, b, off, x=6)
        return addr, typ[:-1]

    def expr(self, e):
        k, line = e[:2]
        if k == 'literal':
            s = e[2]
            if not s.lower().startswith('0x') and any(ch in s for ch in '.eE'):
                return self.literal(float(s.rstrip('fF')), 'float', line)
            v = int(s.rstrip('uU'), 16 if s.lower().startswith('0x') else 10)
            if v > 0xffffffff:
                self.fail(line, 'integer literal exceeds 32 bits')
            return self.literal(v, 'uint' if s[-1] in 'uU' or v > 0x7fffffff else 'int', line)
        if k == 'var':
            if e[2] not in self.env:
                self.fail(line, f'unknown variable {e[2]}')
            s = self.env[e[2]]
            return s['reg'], s['type']
        if k == 'builtin':
            sr = {f'{name}.{axis}': ids[j] for name, ids in [('threadIdx',(0,1,2)),('blockIdx',(8,9,10)),('blockDim',(11,12,13)),('gridDim',(14,19,20))] for j,axis in enumerate('xyz')}
            if e[2] not in sr:
                self.fail(line, f'unknown builtin {e[2]}')
            r = self.reg(); self.emit(3, line, r, x=sr[e[2]])
            return r, 'uint'
        if k == 'index':
            addr, typ = self.address(e); r = self.reg()
            self.emit(0x60, line, r, addr, x=0x38 if typ == 'float' else 0x20, kind='MI')
            return r, typ
        if k in ('unary','postfix'):
            op=e[2]
            if op in ('++','--'):
                target=self.lvalue(e[3]);a,typ=self.load_target(target,line)
                if typ.endswith('*'):self.fail(line,'pointer increment unsupported')
                old=a
                if k=='postfix':
                    old=self.reg();self.emit(1,line,old,a)
                one=self.literal(1.0 if typ=='float' else 1,typ,line)
                new=self.binary('+' if op=='++' else '-',(a,typ),one,line)
                self.store_target(target,new,line)
                return (old,typ) if k=='postfix' else new
            a,typ=self.expr(e[3])
            if typ.endswith('*'):self.fail(line,'unary pointer operation unsupported')
            if op=='+':return a,typ
            if op=='!':return self.compare('==',(a,typ),self.literal(0,typ,line),line)
            if op=='~':
                if typ=='float':self.fail(line,'bitwise operation requires integers')
                return self.binary('^',(a,typ),self.literal(0xffffffff,typ,line),line)
            r=self.reg();self.emit(0x3d if typ=='float' else 0x20,line,r,a);return r,typ
        if k=='cast':
            a,typ=self.expr(e[3]);dst=e[2]
            if typ==dst or {typ,dst}<={'int','uint'}:return a,dst
            self.fail(line,'floating casts unavailable: simulator XCVT is incomplete')
        if k=='call':return self.intrinsic(e)
        if k=='binary':
            op=e[2]
            if op in ('&&','||'):
                left=self.truth(self.expr(e[3]),line);result=self.reg()
                self.raw_emit(1,line,result,left)
                parent=self.active
                if op=='&&':self.active=left
                else:
                    one,_=self.literal(1,'int',line);mask=self.reg();self.raw_emit(0x14,line,mask,one,left);self.active=mask
                # Combine with the surrounding active mask, including nested branches.
                if parent is not None:
                    mask=self.reg();self.raw_emit(0x13,line,mask,self.active,parent);self.active=mask
                right=self.truth(self.expr(e[4]),line);self.emit(1,line,result,right)
                self.active=parent
                return result,'int'
            return self.binary(op,self.expr(e[3]),self.expr(e[4]),line)
        self.fail(line, 'unsupported expression')

    def compare(self, op, left, right, line):
        a,at=left;b,bt=right;typ=self.compatible(at,bt,line)
        if typ.endswith('*'):self.fail(line,'pointer comparison unsupported')
        cmp={'==':0,'!=':1,'<':2,'<=':3,'>':4,'>=':5}[op]
        one,_=self.literal(1,'int',line)
        enabled=self.enabled(line)
        # SETP and SELP must be adjacent: emit() would overwrite P0.
        self.raw_emit(0x40,line,0,a,b,x=cmp|({'int':0,'uint':1,'float':2}[typ]<<4),y=3)
        r=self.reg();self.raw_emit(9,line,r,one,255)
        if enabled is not None:self.raw_emit(0x13,line,r,r,enabled)
        return r,'int'

    def truth(self,value,line):
        return self.compare('!=',value,self.literal(0,value[1],line),line)[0]

    def condition(self,e):
        value=self.expr(e)
        if e[0]=='binary' and e[2] in ('==','!=','<','<=','>','>='):return value[0]
        return self.truth(value,e[1])

    def binary(self,op,left,right,line):
        if op in ('==','!=','<','<=','>','>='):return self.compare(op,left,right,line)
        a,at=left;b,bt=right;typ=self.compatible(at,bt,line)
        if typ.endswith('*'):self.fail(line,'use pointer subscripting instead of pointer arithmetic')
        if typ=='float':
            if op not in ('+','-','*'):self.fail(line,'float operators supported: + - *; use rcpf explicitly for reciprocal')
            opcode={'+':0x30,'-':0x31,'*':0x32}[op];x=0
        else:
            if op in ('/','%') and typ!='int':self.fail(line,'IDIV/IREM support signed int only')
            opcode={'+':0x10,'-':0x14,'*':0x13,'/':0x21,'%':0x22,'&':0x16,'|':0x16,'^':0x16,'<<':0x18,'>>':0x1a if at=='int' else 0x19}[op]
            x={'&':0,'|':1,'^':2}.get(op,0)
        r=self.reg();self.emit(opcode,line,r,a,b,x=x);return r,typ

    def lvalue(self,e):
        if e[0]=='var':return ('reg',*self.expr(e))
        if e[0]=='index':return ('mem',*self.address(e))
        self.fail(e[1],'assignment needs a variable or subscript')

    def load_target(self,target,line):
        kind,r,typ=target
        if kind=='reg':return r,typ
        dst=self.reg();self.emit(0x60,line,dst,r,x=0x38 if typ=='float' else 0x20,kind='MI');return dst,typ

    def store_target(self,target,value,line):
        kind,dst,typ=target;src,actual=value;self.compatible(typ,actual,line)
        if kind=='mem':self.emit(0x61,line,src,dst,x=0x38 if typ=='float' else 0x20,y=1,kind='MI')
        else:
            self.emit(1,line,dst,src)
            if typ.endswith('*'):self.emit(1,line,dst+1,src+1)

    def intrinsic(self,e):
        _,line,name,args=e
        unary={
            'abs':(0x1f,0,'int','int'),'fabsf':(0x3c,0,'float','float'),
            'rcpf':(0x38,0,'float','float'),'rsqrtf':(0x39,0,'float','float'),
            'sinf':(0x3a,0,'float','float'),'cosf':(0x3a,1,'float','float'),
            'exp2f':(0x3a,2,'float','float'),'log2f':(0x3a,3,'float','float'),
            'floorf':(0x3b,0,'float','float'),'ceilf':(0x3b,1,'float','float'),
            'truncf':(0x3b,2,'float','float'),'nearbyintf':(0x3b,3,'float','float'),
            '__popc':(0x1d,0,'integer','uint'),'__brev':(0x1e,0,'integer','uint'),
            '__clz':(0x25,1,'integer','uint'),'__ctz':(0x25,2,'integer','uint'),
            '__find_msb':(0x25,0,'integer','uint'),'__bmsk':(0x24,0,'integer','uint'),
        }
        if name=='__syncthreads':
            if args or self.depth or self.has_return:self.fail(line,'__syncthreads requires uniform top-level control without return')
            self.raw_emit(0x55,line)
            return 255,'int'
        values=[self.expr(v) for v in args]
        if name in unary:
            op,x,want,out=unary[name]
            if len(values)!=1 or (values[0][1] not in ('int','uint') if want=='integer' else values[0][1]!=want):self.fail(line,f'{name} requires one {want} argument')
            r=self.reg();self.emit(op,line,r,values[0][0],x=x);return r,out
        if name in ('min','max','fminf','fmaxf'):
            if len(values)!=2:self.fail(line,'min/max require two arguments')
            typ=self.compatible(values[0][1],values[1][1],line)
            if typ not in (('float',) if name.startswith('f') else ('int','uint')):self.fail(line,'min/max argument types are invalid')
            r=self.reg();self.emit(0x34 if typ=='float' else 0x15,line,r,values[0][0],values[1][0],x=int('max' in name)|(2 if typ=='uint' else 0));return r,typ
        if name in ('fmaf','__imad'):
            want='float' if name=='fmaf' else 'integer'
            if len(values)!=3 or any(t!='float' if want=='float' else t not in ('int','uint') for _,t in values):self.fail(line,f'{name} requires three {want} arguments')
            typ=values[0][1]
            for _,t in values:typ=self.compatible(typ,t,line)
            r=self.reg();self.emit(1,line,r,values[2][0]);self.emit(0x33 if want=='float' else 0x12,line,r,values[0][0],values[1][0]);return r,typ
        atomic={'atomicExch':0,'atomicAdd':1,'atomicMin':2,'atomicMax':3,'atomicAnd':9,'atomicOr':10,'atomicXor':11,'atomicCAS':8}
        if name in atomic:
            if len(values)!=(3 if name=='atomicCAS' else 2) or values[0][1] not in ('int*','uint*'):self.fail(line,'atomic requires int*/uint* and matching scalar operands')
            typ=values[0][1][:-1]
            if any(t!=typ for _,t in values[1:]):self.fail(line,'atomic scalar types must match pointer element type')
            op=atomic[name]+(2 if typ=='uint' and name in ('atomicMin','atomicMax') else 0)
            r=self.reg();self.emit(0x63,line,r,values[0][0],values[-1][0],x=0x20,y=op,b=values[1][0] if name=='atomicCAS' else 0);return r,typ
        self.fail(line,f'unsupported intrinsic {name}')

    def inline_asm(self,st):
        from ntasm import parse
        _,line,template,outputs,inputs=st
        if self.depth or self.has_return:self.fail(line,'asm requires top-level control without return; use native NTAS1 branches inside the template')
        bindings=[];groups=[];copyback=[]
        for is_output,operands in ((True,outputs),(False,inputs)):
            for constraint,exprs in operands:
                allowed=('=r','+r','=f','+f','=l','+l','=q','+q') if is_output else ('r','f','l','q')
                if constraint not in allowed:self.fail(line,'unsupported asm constraint')
                kind=constraint[-1];count=4 if kind=='q' else 1
                if len(exprs)!=count:self.fail(line,'q requires four scalar operands; other constraints require one')
                values=[];targets=[]
                for expr in exprs:
                    if is_output:
                        if expr[0]!='var':self.fail(line,'asm outputs must be local/parameter variables')
                        target=self.lvalue(expr);targets.append(target);value=self.load_target(target,line)
                    else:value=self.expr(expr)
                    typ=value[1]
                    if not (typ.endswith('*') if kind=='l' else typ=='float' if kind=='f' else typ in ('int','uint') if kind=='r' else typ in ('int','uint','float')):self.fail(line,'asm constraint/type mismatch')
                    values.append(value)
                start=self.reg(kind in ('l','q'))
                if kind=='q':self.reg();self.reg()
                regs=list(range(start,start+(4 if kind=='q' else 2 if kind=='l' else 1)))
                if not is_output or constraint.startswith('+'):
                    for j,reg in enumerate(regs):self.emit(1,line,reg,values[j if kind=='q' else 0][0]+(j if kind=='l' else 0))
                if is_output:
                    for j,target in enumerate(targets):copyback.append((target,(start+j if kind=='q' else start,target[2])))
                bindings.append(start)
                groups.append((start,len(regs),is_output))
        # Numeric register tokens are forbidden, except RZ and predicate destinations.
        # Predicate destinations use Pn syntax, converted to the ISA's d field.
        if re.search(r'\bR[0-9]+\b',template):self.fail(line,'asm uses %N operands, RZ and Pn; fixed registers are forbidden')
        def operand(match):
            idx=int(match[1])
            if idx>=len(bindings):self.fail(line,'asm operand index out of range')
            return 'R'+str(bindings[idx])
        text=re.sub(r'%(\d+)',operand,template).replace(';','\n')
        text=re.sub(r'(?m)(SETP(?:\.m\d+)?\s+)P([0-6])\b',r'\1R\2',text)
        try:instructions=parse(text)
        except ValueError as exc:self.fail(line,str(exc))
        from ntasm import OPS, MEM_OPS
        def access(reg,width=1,write=False):
            if reg==255:return
            if not any(start<=reg and reg+width<=start+count and (not write or output) for start,count,output in groups):
                self.fail(line,'asm register group exceeds its constraint or writes an input; use +l/=l or +q/=q for wide outputs')
        unary={'MOV','CVTA','INEG','IABS','POPC','BREV','BMSK','FIND','FABS','FNEG','RCP','RSQ','MUFU','FRND'}
        for ins in instructions:
            name=OPS[ins['op']];d,a,c=ins['d'],ins['a'],ins['c'];x=ins['x']
            if name in ('NOP','BRA','SSY','BAR','EXIT'):continue
            if name in MEM_OPS:
                width=4 if name in ('LD128','ST128') or (x>>3)&15==10 else 2 if (x>>3)&15 in (6,8) else 1
                access(d,width,name not in ('ST','ST128'));access(a,2 if width>1 else 1)
                if ins['kind']=='M':access(c,2 if width>1 else 1)
            elif name=='LDC':access(d,2 if (x>>5)&3==1 else 1,True)
            elif name=='MOV32I':access(d,2 if x&1 else 1,True)
            elif name=='S2R':access(d,2 if x in (32,33,34) else 1,True)
            elif name in ('ATOM','RED'):
                if name=='ATOM':access(d,1,True)
                access(a);access(c)
                if ins['y']==8:access(ins['b'])
            elif name=='SETP':access(a);access(c)
            else:
                pair=name=='CVTA' or name in ('IADD','ISUB') and x==6
                width=2 if pair or name in ('IMUL','IMAD') and (x>>3)&7==2 else 1
                access(d,width,True)
                if name!='VOTE':access(a,2 if pair else 1)
                if name not in unary and name!='VOTE' and not(name=='LOP' and x&4):access(c,2 if pair else 1)
        base=len(self.code)
        for ins in instructions:
            if ins['kind']=='B' and ins['x']&1:ins['imm']+=base
            self.raw_emit(line=line,**ins)
        for target,value in copyback:self.store_target(target,value,line)

    def block(self, statements):
        saved = self.env.copy()
        locals_here = []
        for st in statements:
            k, line = st[:2]
            if k == 'decl':
                name, typ, expr = st[2:]
                if name in self.env or name in ('threadIdx', 'blockIdx', 'blockDim', 'gridDim', 'fmaf'):
                    self.fail(line, f'duplicate/reserved name {name} (shadowing unsupported)')
                src, actual = self.expr(expr); self.compatible(typ, actual, line)
                dst = self.reg(typ.endswith('*'))
                self.emit(1, line, dst, src)
                if typ.endswith('*'):
                    self.emit(1, line, dst+1, src+1)
                declaration_mask=self.active
                if self.loops:
                    enabled=self.enabled(line);declaration_mask=self.reg()
                    self.raw_emit(1,line,declaration_mask,enabled)
                sym = {'name': name, 'type': typ, 'reg': dst, 'start_pc': len(self.code)*8, 'line': line, 'mask_reg': declaration_mask}
                self.env[name] = sym; self.symbols.append(sym); locals_here.append(sym)
            elif k=='scope':self.block(st[2])
            elif k=='exprstmt':
                expr=st[2]
                if expr[0]=='postfix':expr=('unary',*expr[1:])
                self.expr(expr)
            elif k=='asm':self.inline_asm(st)
            elif k == 'assign':
                target=self.lvalue(st[2]);op=st[4] if len(st)>4 else '='
                if op=='=':value=self.expr(st[3])
                else:
                    old=self.load_target(target,line)
                    snapshot=self.reg();self.emit(1,line,snapshot,old[0])
                    value=self.binary(op[:-1],(snapshot,old[1]),self.expr(st[3]),line)
                self.store_target(target,value,line)
            elif k in ('break','continue'):
                if not self.loops:self.fail(line,f'{k} outside a loop')
                enabled=self.enabled(line);snapshot=self.reg();self.raw_emit(1,line,snapshot,enabled)
                running,iteration=self.loops[-1]
                if k=='break':self.raw_emit(0x14,line,running,running,snapshot)
                self.raw_emit(0x14,line,iteration,iteration,snapshot)
            elif k in ('if', 'while'):
                if self.depth >= 8:
                    self.fail(line, 'control-flow nesting exceeds supported depth 8')
                self.depth += 1
                parent = self.active
                if k == 'while':
                    mask = self.reg()
                    if parent is None:
                        self.raw_emit(2, line, mask, kind='I', imm=1)
                    else:
                        self.raw_emit(1, line, mask, parent)
                    self.active = mask
                    test = len(self.code)
                    cond = self.condition(st[2])
                    self.raw_emit(1, line, mask, cond)
                    # VOTE.ANY produces a uniform decision across all live lanes;
                    # lane-local loop masks stay false after a lane exits the loop.
                    self.raw_emit(0x40, line, a=mask, c=255, x=0x11, y=3)
                    vote = self.reg(); self.raw_emit(7, line, vote, x=1, y=0)
                    self.raw_emit(0x40, line, a=vote, c=255, x=0x11, y=3)
                    leave = self.raw_emit(0x50, line, kind='B', guard=1, negate=1)
                    iteration=self.reg();self.raw_emit(1,line,iteration,mask)
                    self.loops.append((mask,iteration))
                    self.block(st[3])
                    self.loops.pop()
                    self.active=mask
                    self.block(st[4])
                    back = self.raw_emit(0x50, line, kind='B')
                    self.patch(back, test)
                    self.patch(leave, len(self.code))
                else:
                    cond = self.condition(st[2])
                    self.active = cond; self.block(st[3])
                    if st[4]:
                        other = self.reg()
                        if parent is None:
                            one = self.reg(); self.raw_emit(2, line, one, kind='I', imm=1)
                        else:
                            one = parent
                        self.raw_emit(0x14, line, other, one, cond)
                        self.active = other; self.block(st[4])
                self.active = parent
                self.depth -= 1
            elif k == 'return':
                # Logical per-lane return. A single final EXIT retires the warp;
                # no partially predicated EXIT is required by the baseline ABI.
                enabled=self.enabled(line)
                self.raw_emit(0x14,line,self.live,self.live,enabled)
        for s in locals_here:
            s['end_pc'] = len(self.code)*8
        self.env = saved

    def compile(self, source, path):
        name, params, body = Parser(source).parse()
        self.has_return=any(t[0]=='return' for t in Parser(source).tokens)
        offset = 0
        for n, t, line in params:
            if n in self.env or n in ('threadIdx', 'blockIdx', 'blockDim', 'gridDim', 'fmaf'):
                self.fail(line, 'duplicate/reserved parameter name')
            size = 8 if t.endswith('*') else 4
            offset = (offset+size-1)//size*size
            reg = self.reg(size == 8)
            self.emit(4, line, reg, x=0x20 if size == 8 else (0x40 if t == 'float' else 0), imm=offset, kind='MI')
            p = {'name': n, 'type': t, 'offset': offset, 'size': size}
            self.params.append(p)
            s = {'name': n, 'type': t, 'reg': reg, 'line': line, 'start_pc': len(self.code)*8}
            self.symbols.append(s); self.env[n] = s
            offset += size
        self.live = self.reg(); self.guard_temp = self.reg()
        self.raw_emit(2, body[0][1] if body else 1, self.live, kind='I', imm=1)
        self.active = self.live
        self.block(body)
        self.raw_emit(0x57, source.count('\n')+1)
        for s in self.symbols:
            s.setdefault('end_pc', len(self.code)*8)
        binary = b''.join(struct.pack('<Q', w) for w in self.code)
        meta = {'format': 'NTMODULE-1', 'compiler_version': VERSION, 'compiler_author': __author__, 'isa': 'NTAS1', 'simulator_min': '1.1.0',
                'kernel': name, 'registers': self.next_reg, 'parameter_size': offset, 'parameters': self.params,
                'source': str(Path(path).resolve()), 'source_text': source, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'code_sha256': hashlib.sha256(binary).hexdigest(), 'line_table': self.map, 'symbols': self.symbols,
                'optimization': 'O0', 'live_mask_register': self.live, 'instructions': self.ir}
        return binary, meta


def compile_file(source, output):
    source, output = Path(source), Path(output)
    if source.suffix.lower() != '.ntcu':
        raise CompileError('NTCC requires .ntcu input; .cu is reserved for CUDA source')
    binary, meta = Compiler().compile(source.read_text(encoding='utf-8-sig'), source)
    output.parent.mkdir(parents=True, exist_ok=True)
    code_path = output.with_suffix('.ntas')
    meta['code_file'] = code_path.name
    code_path.write_bytes(binary)
    output.with_suffix('.ntm.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [f'{m["pc"]:06x} {w:016x} line {m["line"]} {json.dumps(i)}' for w, m, i in zip(struct.unpack('<'+'Q'*(len(binary)//8), binary), meta['line_table'], meta['instructions'])]
    output.with_suffix('.listing').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    from ntasm import render
    output.with_suffix('.ntasm').write_text(render(meta),encoding='utf-8')
    return output.with_suffix('.ntm.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', action='version', version='NTCC '+VERSION+' | Author: '+__author__)
    p.add_argument('source'); p.add_argument('-o', '--output', required=True)
    p.add_argument('-g', action='store_true', help='source debug data (always enabled in 1.2.0)')
    args = p.parse_args()
    try:
        print(compile_file(args.source, args.output))
    except (CompileError, OSError, ValueError, OverflowError) as e:
        print(f'ntcc: {e}', file=sys.stderr); return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
