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
#include "protocol.hpp"
#include "code_validation.hpp"
#include "timing/cycle_sim.hpp"
#include <iostream>
#include <memory>
#include <array>
#include <algorithm>

using namespace ntisa;
int main(int argc,char**argv){try{
    if(argc!=4 && argc!=5)throw std::runtime_error("bridge func|cycle request response [trace]");
    std::string backend=argv[1];
    if(backend!="func" && backend!="cycle")throw std::runtime_error("invalid backend");
    if(argc==5 && backend!="func")throw std::runtime_error("trace only supports func");
    Reader r(argv[2]);
    auto magic=r.bytes(8);
    if(std::string(magic.begin(),magic.end())!=std::string("NTRUN01\0",8))throw std::runtime_error("invalid protocol version");
    auto nw=r.number(4),regs=r.number(4),ps=r.number(4);
    std::array<uint32_t,3> grid,block;
    for(auto&x:grid)x=uint32_t(r.number(4));
    for(auto&x:block)x=uint32_t(r.number(4));
    uint64_t gc=1,bc=1;
    for(auto x:grid){if(!x || x>0x7fffffff/gc)throw std::runtime_error("invalid grid");gc*=x;}
    for(auto x:block){if(!x || x>1024/bc)throw std::runtime_error("invalid block");bc*=x;}
    auto nb=r.number(4),na=r.number(4),budget=r.number(8);
    if(!nw || nw>4096 || regs<2 || regs>255 || ps>4096 || na>512 || ps!=na*8 || nb>512 || !budget || budget>1000000000)
        throw std::runtime_error("invalid resource declaration");
    std::vector<uint64_t> words(nw);for(auto&x:words)x=r.number(8);
    validate_code(words,unsigned(regs));
    std::vector<std::vector<uint8_t>> buffers;
    uint64_t total=0;
    for(size_t i=0;i<nb;++i){
        auto size=r.number(8);
        if(size>(uint64_t(256)<<20)-total)throw std::runtime_error("buffer budget exceeded");
        total+=size;buffers.push_back(r.bytes(size));
    }
    struct Arg{uint16_t off;uint8_t kind;uint64_t value,offset;};
    std::vector<Arg> args;
    for(size_t i=0;i<na;++i){
        Arg a;a.off=r.number(2);a.kind=r.number(1);auto reserved=r.number(1);
        a.value=r.number(8);a.offset=r.number(8);
        if(a.off!=8*i || a.kind>1 || reserved || (a.kind==0 && (a.value>0xffffffff || a.offset)) ||
           (a.kind==1 && (a.value>=nb || a.offset>buffers[a.value].size())))
            throw std::runtime_error("invalid argument record");
        args.push_back(a);
    }
    if(r.at!=r.data.size())throw std::runtime_error("trailing request bytes");
    auto cycle=std::make_unique<CycleSim>();
    FuncSim& sim=cycle->func;
    if(backend=="cycle")cycle->init();else sim.init();
    std::vector<uint64_t> bases;
    for(auto&buf:buffers){
        auto base=sim.malloc(buf.size()+64);
        if(base>0xffffffff || uint64_t(buf.size())+64>0x100000000ULL-base)
            throw std::runtime_error("buffer exceeds low32 profile");
        std::vector<uint8_t> padded(buf.size()+64,0xa5);
        std::copy(buf.begin(),buf.end(),padded.begin()+32);
        sim.memcpy_h2d(base,padded.data(),padded.size());bases.push_back(base+32);
    }
    std::vector<uint8_t> params(ps,0);
    for(auto a:args){
        auto value=a.kind?bases[a.value]+a.offset:a.value;
        if(a.kind && value>0xffffffff)throw std::runtime_error("pointer exceeds low32 profile");
        for(unsigned j=0;j<(a.kind?8:4);++j)params[a.off+j]=(value>>(8*j))&255;
    }
    KernelInfo k;k.name="nutcc_kernel";k.code_size=words.size()*8;k.regs_u32=regs;k.param_size=ps;
    sim.load_code(words,k);
    if(!sim.launch(0,grid[0],grid[1],grid[2],block[0],block[1],block[2],0,params,0))
        throw std::runtime_error("launch failed: "+sim.error.msg);
    if(argc==5)sim.enable_trace(true);
    if(backend=="cycle"){
        int error=cycle->run(budget);
        if(error)throw std::runtime_error("cycle execution error "+std::to_string(error)+": "+sim.error.msg);
    }else{
        uint64_t rounds=0;
        while(!sim.is_done() && !sim.has_error() && rounds++<budget){
            if(!sim.step_one_round())break;
            if(sim.trace.size()>100000)throw std::runtime_error("trace exceeds 100000 events");
        }
    }
    if(sim.has_error() || !sim.is_done())throw std::runtime_error("execution failed or step budget exceeded: "+sim.error.msg);
    for(size_t i=0;i<buffers.size();++i){
        std::vector<uint8_t> padded(buffers[i].size()+64);
        sim.memcpy_d2h(padded.data(),bases[i]-32,padded.size());
        if(!std::all_of(padded.begin(),padded.begin()+32,[](uint8_t x){return x==0xa5;}) ||
           !std::all_of(padded.end()-32,padded.end(),[](uint8_t x){return x==0xa5;}))
            throw std::runtime_error("output sentinel was overwritten");
        std::copy(padded.begin()+32,padded.end()-32,buffers[i].begin());
    }
    if(argc==5){std::ofstream trace(argv[4]);
        for(auto&t:sim.trace)trace<<t.pc<<'\t'<<unsigned(t.op)<<'\t'<<t.exec_mask<<'\n';
        if(!trace)throw std::runtime_error("cannot write trace");}
    std::ofstream out(argv[3],std::ios::binary);
    out.write("NTOUT01\0",8);write_number(out,sim.inst_count,8);
    write_number(out,backend=="cycle"?cycle->stats.total_cycles:0,8);write_number(out,buffers.size(),4);
    for(auto&buf:buffers){write_number(out,buf.size(),8);out.write(reinterpret_cast<const char*>(buf.data()),buf.size());}
    if(!out)throw std::runtime_error("cannot write response");
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
