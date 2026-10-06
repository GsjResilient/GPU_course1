// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Author: Di Zhao, Trae CN
// Contact: zhaodi@ucas.ac.cn
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers,
//             General-Purpose Graphics Processor Architecture,
//             Morgan & Claypool, 2018
// September 2026
// Improved by: 明同辉（26秋），王一锴（26秋），崔文君（26秋）
// =============================================================================
#pragma once
#include "isa/ntisa.hpp"
#include <vector>
#include <stdexcept>

inline void validate_code(const std::vector<uint64_t>& words, unsigned regs) {
    using namespace ntisa;
    if(words.empty() || words.size()>4096 || regs<2 || regs>255)
        throw std::runtime_error("invalid code size or registers");
    auto check_reg=[&](unsigned r){if(r!=255 && r>=regs)throw std::runtime_error("register outside resource declaration");};
    for(size_t pc=0;pc<words.size();++pc){
        Inst i;
        if(decode(words[pc],pc*8,i)!=ERR_OK)throw std::runtime_error("instruction decode failed");
        if(i.g && i.gp>6)throw std::runtime_error("PT guards are outside profile");
        if(i.m)throw std::runtime_error("non-RN modifier is outside profile");
        if(i.op==OP_EXIT){
            if(pc+1!=words.size() || words[pc]!=0xae00000000000000ULL)
                throw std::runtime_error("EXIT must be final and unconditional");
            continue;
        }
        if(pc+1==words.size())throw std::runtime_error("missing final EXIT");
        if(i.op==OP_SETP){
            if(i.d>6 || i.cmp_type()>2 || i.cmp_code()>5 || i.y!=0x83)
                throw std::runtime_error("unsupported comparison");
            check_reg(i.a);check_reg(i.c);continue;
        }
        check_reg(i.d);
        switch(i.op){
        case OP_MOV32I:if(i.x)throw std::runtime_error("wide immediate");break;
        case OP_S2R:
            if(!(i.x<=2 || (i.x>=8 && i.x<=14) || i.x==19 || i.x==20))throw std::runtime_error("unsupported SR");break;
        case OP_LDC:
            if(i.x || i.y || i.simm16<0 || i.simm16+4>4096)throw std::runtime_error("unsupported LDC");break;
        case OP_LD:case OP_ST:
            if(i.kind!=FK_MI || (i.x!=0 && i.x!=0x20 && i.x!=0x38) || i.y || i.simm16)
                throw std::runtime_error("unsupported memory instruction");
            check_reg(i.a);break;
        case OP_MOV:check_reg(i.a);break;
        case OP_SELP:
            if(i.y>14 || (i.y&7)>6)throw std::runtime_error("unsupported SELP");
            check_reg(i.a);check_reg(i.c);break;
        case OP_IADD:case OP_ISUB:
            if(i.x!=4)throw std::runtime_error("wide integer op");
            check_reg(i.a);check_reg(i.c);break;
        case OP_IMUL:
            if(i.x>1)throw std::runtime_error("wide multiply");
            check_reg(i.a);check_reg(i.c);break;
        case OP_LOP:
            if(i.x>2)throw std::runtime_error("unsupported LOP");
            check_reg(i.a);check_reg(i.c);break;
        case OP_SHL:case OP_SHR:case OP_FADD:case OP_FSUB:case OP_FMUL:
            check_reg(i.a);check_reg(i.c);break;
        default:throw std::runtime_error("opcode outside compiler profile");
        }
    }
}
