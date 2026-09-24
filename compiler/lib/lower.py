"""Typed, ordered lowering. Mutable locals preserve inactive lane values."""
from fractions import Fraction
from .ir import VReg, IRInst, Program
from .validate import kind_of, walk, const_int, loop_contract

OPS = {'+':'add','-':'sub','*':'mul','&':'and','|':'or','^':'xor',
       '<<':'shl','>>':'shr','==':'cmp_eq','!=':'cmp_ne','<':'cmp_lt',
       '<=':'cmp_le','>':'cmp_gt','>=':'cmp_ge'}
SR = {'threadIdx':(0,1,2),'blockIdx':(8,9,10),'blockDim':(11,12,13),'gridDim':(14,19,20)}


def f32_literal_bits(text):
    """Exact rational to IEEE binary32 round-to-nearest, ties-to-even."""
    text = text.lower()
    sign = 0x80000000 if text.startswith('-') else 0
    text = text.lstrip('+-')
    if text.startswith('0x'):
        mant, exponent = text[2:].split('p')
        whole, _, frac = mant.partition('.')
        q = Fraction(int(whole + frac, 16), 16 ** len(frac)) * Fraction(2) ** int(exponent)
    else:
        q = Fraction(text)
    if not q:
        return sign
    def rounded(v):
        a, rem = divmod(v.numerator, v.denominator)
        return a + (2 * rem > v.denominator or (2 * rem == v.denominator and a % 2))
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if q < Fraction(2) ** e:
        e -= 1
    if e < -126:
        return sign | rounded(q * (2 ** 149))
    mantissa = rounded(q * Fraction(2) ** (23 - e))
    if mantissa == 1 << 24:
        e += 1
        mantissa >>= 1
    if e > 127:
        raise ValueError('floating literal overflows binary32')
    return sign | ((e + 127) << 23) | (mantissa - (1 << 23))


class Builder:
    def __init__(self, kernel):
        self.k = kernel
        self.unit = kernel.unit
        params = []
        for i, param in enumerate(kernel.parameters):
            p = {key: value for key, value in param.items() if key != 'id'}
            p.update(offset=i*8, size=8 if p['kind'].startswith('ptr_') else 4)
            params.append(p)
        self.program = Program(kernel.name, params, len(params)*8, [])
        self.counter, self.locals, self.initialized = 0, {}, set()
        self.statement_visits = 0
        self.constant_decls = {n['id']: n for n in walk(kernel.unit.ast)
                               if n.get('kind') == 'VarDecl' and n.get('constexpr')}
        self.loop_values = {}
        for src, param in zip(kernel.parameters, params):
            reg = self.emit('param', param['kind'], (), param['offset'], None, kernel.node)
            self.locals[src['id']] = reg
            self.initialized.add(src['id'])

    def error(self, node, code, msg):
        self.unit.error(node, code, msg)

    def fresh(self, kind):
        r = VReg(self.counter, kind)
        self.counter += 1
        return r

    def emit(self, op, kind, args=(), imm=None, guard=None, node=None, dst=None):
        if len(self.program.instructions) >= 4096:
            self.error(node or self.k.node, 'E_CODE_SIZE', 'expanded IR exceeds 4096 instructions')
        reg = dst if dst is not None else (self.fresh(kind) if kind else None)
        self.program.instructions.append(IRInst(op, reg, tuple(args), imm, guard,
                                                self.unit.location(node or self.k.node)))
        return reg

    def constant(self, value, kind='i32', guard=None, node=None):
        return self.emit('const', kind, imm=value & 0xffffffff, guard=guard, node=node)

    def address(self, node, guard):
        if node.get('kind') == 'ParenExpr':
            return self.address(node['inner'][0], guard)
        if node.get('kind') != 'ArraySubscriptExpr':
            self.error(node, 'E_LVALUE', 'expected scalar variable or array subscript')
        base = self.expr(node['inner'][0], guard)
        index = self.expr(node['inner'][1], guard)
        if not base.kind.startswith('ptr_') or index.kind not in ('i32','u32'):
            self.error(node, 'E_ADDRESS', 'invalid array base or index')
        if base.kind != 'ptr_u8':
            scale = self.constant(4, index.kind, guard, node)
            index = self.emit('mul', index.kind, (index,scale), guard=guard, node=node)
        return self.emit('add', base.kind, (base,index), guard=guard, node=node)

    def assign(self, lhs, value, guard):
        if lhs.get('kind') == 'ParenExpr':
            return self.assign(lhs['inner'][0], value, guard)
        if lhs.get('kind') == 'DeclRefExpr':
            id_ = lhs['referencedDecl']['id']
            if id_ in self.loop_values:
                self.error(lhs, 'E_LOOP', 'loop variable cannot be modified in body')
            self.emit('copy', value.kind, (value,), guard=guard, node=lhs, dst=self.locals[id_])
            self.initialized.add(id_)
        else:
            addr = self.address(lhs, guard)
            self.emit('store', None, (addr,value), guard=guard, node=lhs)
        return value

    def boolean(self, n, guard):
        val = self.expr(n,guard)
        # Always snapshot; the source variable may be assigned by either branch.
        zero = self.constant(0,val.kind,guard,n)
        return self.emit('cmp_ne','bool',(val,zero),guard=guard,node=n)

    def mask(self,parent,condition,invert,node):
        if invert:
            condition=self.emit('xor','bool',(condition,self.constant(1,'bool',node=node)),node=node)
        if parent is not None:
            condition=self.emit('and','bool',(parent,condition),node=node)
        return condition

    def expr(self, n, guard=None):
        kind = n['kind']
        children = n.get('inner', [])
        if kind in ('ParenExpr','ConstantExpr'):
            return self.expr(children[0], guard)
        if kind == 'ImplicitCastExpr':
            val = self.expr(children[0], guard)
            if n['castKind'] in ('IntegralToBoolean','FloatingToBoolean'):
                zero = self.constant(0, val.kind, guard, n)
                return self.emit('cmp_ne', 'bool', (val,zero), guard=guard, node=n)
            target = kind_of(n, self.unit)
            if target == val.kind:
                return val
            return self.emit('copy', target, (val,), guard=guard, node=n)
        if kind in ('IntegerLiteral','FloatingLiteral','CXXBoolLiteralExpr'):
            typ = kind_of(n, self.unit)
            value = f32_literal_bits(n['value']) if typ == 'f32' else int(n['value'])
            return self.constant(value, typ, guard, n)
        if kind == 'DeclRefExpr':
            id_ = n['referencedDecl']['id']
            if id_ in self.loop_values:
                return self.constant(self.loop_values[id_], 'i32', guard, n)
            if id_ in self.constant_decls:
                return self.expr(self.constant_decls[id_]['inner'][-1], guard)
            if id_ not in self.initialized:
                self.error(n, 'E_UNINITIALIZED', 'variable may be uninitialized: ' + n['referencedDecl'].get('name',''))
            return self.locals[id_]
        if kind == 'MemberExpr':
            ref = next(c for c in walk(n) if c.get('kind') == 'DeclRefExpr')
            sr = SR[ref['referencedDecl']['name']]['xyz'.index(n['name'])]
            return self.emit('sr', 'u32', imm=sr, guard=guard, node=n)
        if kind == 'ArraySubscriptExpr':
            return self.emit('load', kind_of(n,self.unit), (self.address(n,guard),), guard=guard, node=n)
        if kind in ('BinaryOperator','CompoundAssignOperator'):
            op = n['opcode']
            if op == '=':
                return self.assign(children[0], self.expr(children[1],guard), guard)
            if op in ('&&','||'):
                left=self.boolean(children[0],guard)
                rhs_mask=self.mask(guard,left,op=='||',n)
                before=set(self.initialized)
                right=self.boolean(children[1],rhs_mask)
                self.initialized=before  # RHS may not execute.
                rhs_value=self.emit('and','bool',(rhs_mask,right),node=n)
                if op=='&&': return rhs_value
                lhs_value=self.mask(guard,left,False,n)
                return self.emit('or','bool',(lhs_value,rhs_value),node=n)
            if kind == 'CompoundAssignOperator':
                # C++17 evaluates RHS before LHS, and the LHS address exactly once.
                right = self.expr(children[1],guard)
                lhs = children[0]
                if lhs['kind'] == 'ArraySubscriptExpr':
                    addr = self.address(lhs,guard)
                    left = self.emit('load',kind_of(lhs,self.unit),(addr,),guard=guard,node=n)
                    value = self.emit(OPS[op[:-1]],left.kind,(left,right),guard=guard,node=n)
                    self.emit('store',None,(addr,value),guard=guard,node=n)
                    return value
                left = self.expr(lhs,guard)
                value = self.emit(OPS[op[:-1]],left.kind,(left,right),guard=guard,node=n)
                return self.assign(lhs,value,guard)
            a, b = self.expr(children[0],guard), self.expr(children[1],guard)
            return self.emit(OPS[op],kind_of(n,self.unit),(a,b),guard=guard,node=n)
        if kind == 'UnaryOperator':
            op = n['opcode']
            val = self.expr(children[0],guard)
            if op == '+': return val
            if op == '-':
                if val.kind == 'f32':
                    sign = self.constant(0x80000000,'u32',guard,n)
                    return self.emit('xor','f32',(val,sign),guard=guard,node=n)
                return self.emit('sub',val.kind,(self.constant(0,val.kind,guard,n),val),guard=guard,node=n)
            if op == '~':
                return self.emit('xor',val.kind,(val,self.constant(0xffffffff,val.kind,guard,n)),guard=guard,node=n)
            if op == '!':
                return self.emit('cmp_eq','bool',(val,self.constant(0,val.kind,guard,n)),guard=guard,node=n)
            self.error(n,'E_LOOP','increment only allowed in canonical for-loop header')
        self.error(n,'E_EXPRESSION','unsupported expression: '+kind)

    def statement(self,n,guard=None):
        self.statement_visits += 1
        if self.statement_visits > 65536:
            self.error(n,'E_EXPANSION_LIMIT','compile-time expansion exceeds 65536 statement visits')
        kind=n.get('kind')
        if kind in ('CompoundStmt','DeclStmt'):
            for c in n.get('inner',[]): self.statement(c,guard)
        elif kind == 'VarDecl':
            self.locals[n['id']] = self.fresh(kind_of(n,self.unit))
            if n.get('inner'):
                value=self.expr(n['inner'][-1],guard)
                self.emit('copy',value.kind,(value,),guard=guard,node=n,dst=self.locals[n['id']])
                self.initialized.add(n['id'])
        elif kind in ('NullStmt','ReturnStmt'):
            return
        elif kind == 'IfStmt':
            condition,then,*other=n['inner']
            try:
                known=bool(const_int(condition,self.constant_decls))
            except ValueError:
                known=None
            if known is not None:
                if known: self.statement(then,guard)
                elif other: self.statement(other[0],guard)
                return
            saved=self.boolean(condition,guard)
            then_mask=self.mask(guard,saved,False,n)
            else_mask=self.mask(guard,saved,True,n)
            before=set(self.initialized)
            self.statement(then,then_mask)
            after_then=set(self.initialized)
            self.initialized=set(before)
            if other: self.statement(other[0],else_mask)
            self.initialized &= after_then
        elif kind == 'ForStmt':
            id_,count,body=loop_contract(n,self.unit,self.constant_decls)
            # An empty loop is side-effect free, so avoid expanding it.
            if all(c.get('kind') in ('CompoundStmt','NullStmt',None) for c in walk(body)):
                return
            for index in range(count):
                self.loop_values[id_]=index
                self.statement(body,guard)
            self.loop_values.pop(id_,None)
        else:
            self.expr(n,guard)

    def build(self):
        body=next(c for c in self.k.node['inner'] if c['kind']=='CompoundStmt')
        self.statement(body)
        self.emit('exit',None,node=body)
        return self.program


def lower_kernel(kernel):
    return Builder(kernel).build()
