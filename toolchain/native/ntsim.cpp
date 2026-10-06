// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.3
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
// September 2026
// =============================================================================
// NT runtime/debug transport 1.2.0, using the independent simulator 1.2.0.
#include "func/func_sim.hpp"
#include "timing/cycle_sim.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <tuple>
#include <memory>
#include <iomanip>
using namespace ntisa;

static std::string quote(const std::string& s) {
    std::ostringstream o; o << '"';
    for (unsigned char c:s) {
        if(c=='"'||c=='\\') o << '\\' << c;
        else if(c<32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else o << c;
    }
    return o.str()+'"';
}
static std::vector<uint8_t> unhex(const std::string& s) {
    if(s=="-") return {};
    if(s.size()%2) throw std::runtime_error("odd hex size");
    std::vector<uint8_t> v;
    auto digit=[](char c)->int {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;if(c>='A'&&c<='F')return c-'A'+10;throw std::runtime_error("invalid hex");};
    for(size_t i=0;i<s.size();i+=2)v.push_back(digit(s[i])*16+digit(s[i+1]));
    return v;
}
static std::string hex(const std::vector<uint8_t>& v) {
    std::ostringstream o; o<<std::hex<<std::setfill('0');for(auto b:v)o<<std::setw(2)<<int(b);return o.str();
}
template<class T> static T arg(std::istringstream& in) {
    T v{};if(!(in>>v))throw std::runtime_error("missing/invalid command argument");return v;
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--version") {std::cout<<"ntsim 1.2.0; simulator 1.2.0\n";return 0;}
        if(argc!=5)throw std::runtime_error("usage: ntsim code.ntas registers parameter_bytes func|cycle");
        bool timing=std::string(argv[4])=="cycle";
        if(!timing && std::string(argv[4])!="func")throw std::runtime_error("unknown backend");
        auto cycle=std::make_unique<CycleSim>();
        auto& s=cycle->func;
        cycle->cfg.n_clusters=1;cycle->cfg.n_cores_per_cluster=1;s.cfg=cycle->cfg;
        cycle->init();
        std::ifstream f(argv[1],std::ios::binary);
        std::vector<char> raw((std::istreambuf_iterator<char>(f)),{});
        if(raw.empty()||raw.size()%8)throw std::runtime_error("invalid instruction file");
        std::vector<uint64_t> code(raw.size()/8);std::memcpy(code.data(),raw.data(),raw.size());
        for(size_t i=0;i<code.size();++i){Inst ins;if(decode(code[i],i*8,ins))throw std::runtime_error("invalid instruction encoding");}
        int regs=std::stoi(argv[2]), param_bytes=std::stoi(argv[3]);
        if(regs<0||regs>254||param_bytes<0||param_bytes>4096)throw std::runtime_error("invalid kernel resources");
        KernelInfo ki;ki.name="kernel";ki.regs_u32=regs;ki.param_size=param_bytes;s.load_code(code,ki);
        std::set<uint64_t> breaks;
        std::tuple<int,int,uint64_t> stopped{-1,-1,0};
        bool at_stop=false, launched=false;
        std::vector<std::pair<uint64_t,uint64_t>> allocations;
        auto check_range=[&](uint64_t p,uint64_t n){
            if(n>16*1024*1024)throw std::runtime_error("transfer exceeds 16 MiB");
            for(auto a:allocations)if(p>=a.first&&p-a.first<=a.second&&n<=a.second-(p-a.first))return;
            throw std::runtime_error("host transfer outside allocation");
        };
        std::cout<<"{\"ok\":true,\"protocol\":1,\"version\":\"1.2.0\",\"simulator\":\"1.2.0\"}"<<std::endl;
        std::string line;
        while(std::getline(std::cin,line)) {
            try {
                std::istringstream in(line);auto cmd=arg<std::string>(in);
                if(cmd=="quit")break;
                if(cmd=="alloc") {
                    auto n=arg<uint64_t>(in);if(!n||n>16*1024*1024)throw std::runtime_error("allocation size must be 1..16 MiB");
                    auto p=s.malloc(n);allocations.push_back({p,n});std::cout<<"{\"ok\":true,\"address\":"<<p<<"}";
                } else if(cmd=="write") {
                    auto p=arg<uint64_t>(in);auto data=unhex(arg<std::string>(in));check_range(p,data.size());s.memcpy_h2d(p,data.data(),data.size());std::cout<<"{\"ok\":true}";
                } else if(cmd=="read") {
                    auto p=arg<uint64_t>(in),n=arg<uint64_t>(in);check_range(p,n);std::vector<uint8_t> v(n);s.memcpy_d2h(v.data(),p,n);std::cout<<"{\"ok\":true,\"hex\":"<<quote(hex(v))<<"}";
                } else if(cmd=="launch"||cmd=="launch3") {
                    if(launched)throw std::runtime_error("one launch per session; create another session");
                    auto gx=arg<uint32_t>(in);uint32_t gy=1,gz=1,dx=1,dy=1,dz=1;
                    if(cmd=="launch3"){gy=arg<uint32_t>(in);gz=arg<uint32_t>(in);}
                    dx=arg<uint32_t>(in);
                    if(cmd=="launch3"){dy=arg<uint32_t>(in);dz=arg<uint32_t>(in);}
                    auto params=unhex(arg<std::string>(in));
                    if(!gx||!gy||!gz||gx>65535||gy>65535||gz>65535||uint64_t(gx)*gy*gz>65535||!dx||!dy||!dz||dx>1024||dy>1024||dz>1024||uint64_t(dx)*dy*dz>1024||params.size()!=size_t(param_bytes))throw std::runtime_error("invalid launch dimensions/parameters");
                    if(!s.launch_by_name("kernel",gx,gy,gz,dx,dy,dz,0,params))throw std::runtime_error("launch: "+s.error.msg);
                    launched=true;std::cout<<"{\"ok\":true}";
                } else if(cmd=="break"||cmd=="delete") {
                    if(timing)throw std::runtime_error("debug commands require func backend");
                    auto pc=arg<uint64_t>(in);if(pc%8||pc>=raw.size())throw std::runtime_error("breakpoint outside code");
                    if(cmd=="break")breaks.insert(pc);else breaks.erase(pc);std::cout<<"{\"ok\":true}";
                } else if(cmd=="run"||cmd=="step") {
                    if(!launched)throw std::runtime_error("kernel not launched");
                    if(s.has_error())throw std::runtime_error("simulator fault: "+s.error.msg);
                    uint64_t budget=arg<uint64_t>(in);if(!budget||budget>100000000)throw std::runtime_error("budget must be 1..100000000");
                    if(timing) {
                        if(cmd=="step")throw std::runtime_error("debug commands require func backend");
                        int error=cycle->run(budget);
                        if(error && error!=ERR_DEADLOCK)throw std::runtime_error("cycle execution: "+s.error.msg);
                        if(error==ERR_DEADLOCK && s.has_error())throw std::runtime_error("cycle execution: "+s.error.msg);
                        std::string reason=error==ERR_DEADLOCK?"budget":"completed";
                        std::cout<<"{\"ok\":true,\"reason\":"<<quote(reason)<<",\"cycles\":"<<cycle->stats.total_cycles<<"}";
                    } else {
                        uint64_t count=0;bool skip=at_stop;std::string reason="completed";
                        s.debug_before=[&](int sm,int w,uint64_t pc){
                            auto here=std::make_tuple(sm,w,pc);
                            bool skip_here=skip&&here==stopped;skip=false;
                            if(count>=budget||(!skip_here&&breaks.count(pc))) {
                                reason=(!skip_here&&breaks.count(pc))?"breakpoint":(cmd=="step"?"step":"budget");
                                stopped=here;at_stop=true;return true;
                            }
                            ++count;return false;
                        };
                        s.debug_paused=false;
                        while(!s.is_done()&&!s.has_error()&&!s.debug_paused){
                            bool progress=s.step_one_round();
                            if(!progress&&!s.is_done()&&!s.debug_paused&&!s.has_error())throw std::runtime_error("scheduler deadlock");
                        }
                        s.debug_before={};
                        if(s.has_error())throw std::runtime_error("execution code="+std::to_string(s.error.code)+" pc="+std::to_string(s.error.pc)+": "+s.error.msg);
                        if(s.is_done())at_stop=false;
                        std::cout<<"{\"ok\":true,\"reason\":"<<quote(reason)<<",\"instructions\":"<<s.inst_count;
                        if(at_stop)std::cout<<",\"sm\":"<<std::get<0>(stopped)<<",\"warp\":"<<std::get<1>(stopped)<<",\"pc\":"<<std::get<2>(stopped);
                        std::cout<<"}";
                    }
                } else if(cmd=="warps") {
                    std::cout<<"{\"ok\":true,\"warps\":[";bool first=true;
                    for(size_t sm=0;sm<s.nsms.size();++sm)for(size_t wi=0;wi<s.nsms[sm].warps.size();++wi){auto& w=s.nsms[sm].warps[wi];if(!w.valid)continue;
                        if(!first)std::cout<<",";first=false;
                        std::cout<<"{\"sm\":"<<sm<<",\"warp\":"<<wi<<",\"cta\":"<<w.ctaid_x<<",\"pc\":"<<(w.stack.empty()?0:w.stack[w.tos].nextpc)<<",\"exited\":"<<(w.exit_done?"true":"false")<<"}";
                    }std::cout<<"]}";
                } else if(cmd=="state") {
                    auto sm=arg<size_t>(in),wi=arg<size_t>(in),lane=arg<size_t>(in);
                    if(sm>=s.nsms.size()||wi>=s.nsms[sm].warps.size()||lane>=32)throw std::runtime_error("invalid focus");
                    auto& w=s.nsms[sm].warps[wi];if(!w.valid||w.exit_done||!(w.lane_valid&(1u<<lane)))throw std::runtime_error("focus is not a live lane");
                    std::cout<<"{\"ok\":true,\"pc\":"<<w.stack[w.tos].nextpc<<",\"cta\":["<<w.ctaid_x<<","<<w.ctaid_y<<","<<w.ctaid_z<<"],\"warp_in_cta\":"<<w.warp_id_in_cta<<",\"active_mask\":"<<w.stack[w.tos].mask<<",\"lane_valid\":"<<w.lane_valid<<",\"registers\":[";
                    for(int r=0;r<256;++r){if(r)std::cout<<",";std::cout<<static_cast<const WarpState&>(w).reg(lane,r);}std::cout<<"],\"predicates\":[";
                    for(int p=0;p<7;++p){if(p)std::cout<<",";std::cout<<int(w.pred(lane,p));}std::cout<<"],\"stack\":[";
                    for(size_t i=0;i<w.stack.size();++i){auto e=w.stack[i];if(i)std::cout<<",";std::cout<<"{\"rpc\":"<<e.rpc<<",\"nextpc\":"<<e.nextpc<<",\"mask\":"<<e.mask<<"}";}std::cout<<"]}";
                } else throw std::runtime_error("unknown command");
                std::cout<<std::endl;
            } catch(const std::exception& e) {s.debug_before={};std::cout<<"{\"ok\":false,\"error\":"<<quote(e.what())<<"}"<<std::endl;}
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
