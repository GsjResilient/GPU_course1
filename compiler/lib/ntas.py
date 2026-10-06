# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
"""NTAS1 field encoders. Memory and constant-load width fields differ."""


def field(v, bits):
    if type(v) is not int or not 0 <= v < 1 << bits:
        raise ValueError(f'invalid {bits}-bit field: {v!r}')
    return v


def header(op, guard, modifier=0):
    h=field(op,7)<<9 | field(modifier,4)
    if guard is not None:
        p,inv=guard
        if type(p) is not int or not 0<=p<=6 or inv not in (True,False):
            raise ValueError('guard requires P0..P6 and boolean inversion')
        h |= 0x100 | int(inv)<<7 | p<<4
    return h<<48


def encode_r(op,d,a=0,c=0,x=0,y=0,b=0,guard=None,modifier=0):
    return (header(op,guard,modifier)|field(d,8)<<40|field(a,8)<<32|
            field(c,8)<<24|field(x,8)<<16|field(y,8)<<8|field(b,8))


def encode_i(op,d,imm,x=0,guard=None):
    if type(imm) is not int or not -(1<<31)<=imm<(1<<32):
        raise ValueError('immediate does not fit 32 bits')
    return header(op,guard)|field(d,8)<<40|(imm&0xffffffff)<<8|field(x,8)


def encode_mi(op,d,a,x,y,offset,guard=None):
    if type(offset) is not int or not -32768<=offset<=32767:
        raise ValueError('offset does not fit signed 16 bits')
    if y&128:
        raise ValueError('MI Y[7] must be zero')
    return (header(op,guard)|field(d,8)<<40|field(a,8)<<32|field(x,8)<<24|
            field(y,8)<<16|(offset&65535))
