"""Fail-closed language subset validation, including implicit casts."""
from dataclasses import dataclass
from pathlib import Path
from .frontend import SDK

TYPES = {'int': 'i32', 'unsigned int': 'u32', 'float': 'f32',
         'unsigned char': 'u8', 'bool': 'bool', 'void': 'void'}


def kind_of(node, unit):
    t = node.get('type', {})
    text = t.get('desugaredQualType', t.get('qualType', ''))
    if 'volatile' in text:
        unit.error(node, 'E_TYPE', 'volatile is unsupported')
    clean = text.replace('const ', '').replace(' const', '').strip()
    if clean.endswith('*'):
        base = clean[:-1].strip()
        if base in ('int', 'unsigned int', 'unsigned char', 'float'):
            return 'ptr_' + TYPES[base]
    if clean in TYPES:
        return TYPES[clean]
    unit.error(node, 'E_TYPE', 'unsupported type: ' + text)


@dataclass
class KernelAST:
    unit: object
    name: str
    node: dict
    parameters: list


def walk(n):
    yield n
    for c in n.get('inner', []):
        yield from walk(c)


def const_int(n, declarations):
    value=_const_int(n,declarations)
    typ=n.get('type',{}).get('qualType','').replace('const ','')
    if typ=='unsigned int':return value&0xffffffff
    if typ=='bool':return int(bool(value))
    if typ=='int' and not -(1<<31)<=value<(1<<31):
        raise ValueError('signed constant expression overflows')
    return value


def _const_int(n, declarations):
    k = n.get('kind')
    children = n.get('inner', [])
    if k in ('IntegerLiteral','CXXBoolLiteralExpr'):
        return int(n['value'])
    if k in ('ParenExpr','ConstantExpr','ImplicitCastExpr'):
        return const_int(children[0], declarations)
    if k == 'DeclRefExpr' and n['referencedDecl']['id'] in declarations:
        return const_int(declarations[n['referencedDecl']['id']]['inner'][-1], declarations)
    if k == 'UnaryOperator':
        v = const_int(children[0], declarations)
        op = n['opcode']
        if op == '-': return -v
        if op == '+': return v
        if op == '!': return int(not v)
    if k == 'BinaryOperator':
        a,b = [const_int(c,declarations) for c in children]
        op = n['opcode']
        if op == '+': return a+b
        if op == '-': return a-b
        if op == '*': return a*b
        if op == '<': return int(a<b)
        if op == '==': return int(a==b)
    raise ValueError('not a supported integer constant expression')


def loop_contract(n, unit, declarations):
    children=n.get('inner',[])
    try:
        init,empty,cond,step,body=children
        decl,=init['inner']
        id_=decl['id']
        def ref(node):
            while node.get('kind') in ('ImplicitCastExpr','ParenExpr'):
                node=node['inner'][0]
            return node.get('referencedDecl',{}).get('id')
        if (init['kind']!='DeclStmt' or decl['kind']!='VarDecl'
                or kind_of(decl,unit)!='i32' or const_int(decl['inner'][-1],declarations)!=0
                or empty.get('kind') or cond.get('opcode')!='<' or ref(cond['inner'][0])!=id_
                or step.get('opcode')!='++' or step.get('isPostfix') or ref(step['inner'][0])!=id_):
            raise ValueError('expected for(int k=0;k<C;++k)')
        count=const_int(cond['inner'][1],declarations)
        if not 0<=count<=256: raise ValueError('loop count must be 0..256')
        for child in walk(body):
            if child.get('kind')=='UnaryOperator' and child.get('opcode') in ('++','--','&') and ref(child['inner'][0])==id_:
                raise ValueError('loop variable cannot be modified or escape')
            if (child.get('kind')=='CompoundAssignOperator' or
                child.get('kind')=='BinaryOperator' and child.get('opcode')=='=') and ref(child['inner'][0])==id_:
                raise ValueError('loop variable cannot be modified')
        return id_,count,body
    except (KeyError,ValueError,IndexError) as e:
        unit.error(n,'E_LOOP',str(e))


def validate_kernel(unit, name):
    functions, builtin_ids = [], set()
    def top(n):
        if not n or n.get('isImplicit'):
            return
        if Path(n.get('_file', '')) == SDK:
            if n.get('kind') == 'VarDecl' and n.get('name') in ('threadIdx', 'blockIdx', 'blockDim', 'gridDim'):
                builtin_ids.add(n['id'])
            return
        kind = n.get('kind')
        if kind in ('TranslationUnitDecl', 'LinkageSpecDecl'):
            for c in n.get('inner', []): top(c)
        elif kind == 'FunctionDecl':
            functions.append(n)
        elif kind == 'VarDecl' and n.get('constexpr'):
            kind_of(n, unit)
        elif n.get('loc') or n.get('range'):
            unit.error(n, 'E_DECL', 'unsupported top-level declaration: ' + str(kind))
    top(unit.ast)
    matches = [f for f in functions if f.get('name') == name and any(c.get('kind') == 'CompoundStmt' for c in f.get('inner', []))]
    if len(matches) != 1:
        unit.error(unit.ast, 'E_KERNEL', 'expected exactly one kernel definition: ' + name)
    node = matches[0]
    if node['type']['qualType'].split(' (')[0] != 'void' or node.get('variadic'):
        unit.error(node, 'E_KERNEL', 'kernel must return void and cannot be variadic')
    parameters = []
    for p in node.get('inner', []):
        if p.get('kind') == 'ParmVarDecl':
            kind = kind_of(p, unit)
            if kind not in ('i32', 'u32', 'f32', 'ptr_i32', 'ptr_u32', 'ptr_f32', 'ptr_u8'):
                unit.error(p, 'E_PARAMETER', 'unsupported kernel parameter')
            if p.get('inner'):
                unit.error(p, 'E_PARAMETER', 'default parameter values are unsupported')
            parameters.append({'name': p.get('name', ''), 'id': p['id'], 'kind': kind,
                               'readonly': p.get('type', {}).get('qualType', '').startswith('const ')})
    if len(parameters) > 512:
        unit.error(node, 'E_PARAMETERS', 'parameter storage exceeds 4096 bytes')
    body = next(c for c in node['inner'] if c.get('kind') == 'CompoundStmt')
    returns = [n for n in walk(body) if n.get('kind') == 'ReturnStmt']
    if returns and (len(returns) != 1 or body['inner'][-1] is not returns[0] or returns[0].get('inner')):
        unit.error(returns[0], 'E_RETURN', 'only final unconditional return is supported')
    allowed = {'CompoundStmt', 'DeclStmt', 'VarDecl', 'IntegerLiteral', 'FloatingLiteral',
               'CXXBoolLiteralExpr', 'ImplicitCastExpr', 'ParenExpr', 'DeclRefExpr',
               'BinaryOperator', 'CompoundAssignOperator', 'UnaryOperator', 'ArraySubscriptExpr',
               'MemberExpr', 'IfStmt', 'ForStmt', 'NullStmt', 'ReturnStmt', 'ConstantExpr'}
    local_ids = {p['id'] for p in parameters} | {n['id'] for n in walk(body) if n.get('kind') == 'VarDecl'}
    constants = {n['id'] for n in walk(unit.ast) if n.get('kind') == 'VarDecl' and n.get('constexpr')}
    constant_decls = {n['id']: n for n in walk(unit.ast) if n.get('kind') == 'VarDecl' and n.get('constexpr')}
    for n in walk(body):
        if not n.get('kind'): continue  # Clang's empty ForStmt condition-variable slot.
        k = n.get('kind')
        if k not in allowed:
            unit.error(n, 'E_SYNTAX', 'unsupported syntax: ' + str(k))
        if k == 'ForStmt':
            loop_contract(n,unit,constant_decls)
        if k == 'IfStmt' and (n.get('hasInit') or n.get('hasVar') or len(n.get('inner',[])) not in (2,3)):
            unit.error(n,'E_IF','if initializer/condition declarations are unsupported')
        if k == 'VarDecl':
            kind = kind_of(n, unit)
            if kind.startswith('ptr_') or n.get('storageClass'):
                unit.error(n, 'E_DECL', 'local pointers and static/extern variables are unsupported')
        if k == 'DeclRefExpr':
            ref = n.get('referencedDecl', {})
            if ref.get('id') not in local_ids | builtin_ids | constants:
                unit.error(n, 'E_REFERENCE', 'unsupported external reference')
        if k == 'MemberExpr':
            refs = [v.get('referencedDecl', {}).get('id') for v in walk(n) if v.get('kind') == 'DeclRefExpr']
            if n.get('name') not in ('x', 'y', 'z') or len(refs) != 1 or refs[0] not in builtin_ids:
                unit.error(n, 'E_BUILTIN', 'only genuine thread/grid index members are supported')
        if k in ('BinaryOperator', 'CompoundAssignOperator'):
            op = n.get('opcode', '')
            if op not in ('+', '-', '*', '&', '|', '^', '<<', '>>', '==', '!=', '<', '<=', '>', '>=', '=', '+=', '-=', '*=', '&=', '|=', '^=', '<<=', '>>=', '&&', '||'):
                unit.error(n, 'E_OPERATOR', 'unsupported operator: ' + op)
            typ = kind_of(n, unit)
            if typ.startswith('ptr_'):
                unit.error(n, 'E_POINTER', 'only array subscript pointer access is supported')
            if op in ('<<', '>>', '<<=', '>>=') and typ != 'u32':
                unit.error(n, 'E_OPERATOR', 'shifts require unsigned operands')
        if k == 'UnaryOperator' and n.get('opcode') not in ('+', '-', '!', '~', '++'):
            unit.error(n, 'E_OPERATOR', 'unsupported unary operator')
        if k == 'ImplicitCastExpr':
            cast = n.get('castKind')
            child = n.get('inner', [{}])[0]
            if cast in ('LValueToRValue', 'NoOp'):
                continue
            if cast in ('IntegralToBoolean', 'FloatingToBoolean'):
                continue
            if cast == 'IntegralCast' and kind_of(child, unit) == 'u8' and kind_of(n, unit) == 'i32':
                continue
            unit.error(n, 'E_CONVERSION', 'unsupported implicit conversion: ' + str(cast))
    return KernelAST(unit, name, node, parameters)
