// =============================================================================
// NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
// Version 1.4
// Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
// Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
// References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
// September 2026
// =============================================================================
#include "../runtime/code_validation.hpp"
#include <fstream>
#include <iterator>
#include <iostream>

int main(int argc,char**argv){try{
    if(argc!=3)throw std::runtime_error("decode_check code.ntbin regs");
    std::ifstream f(argv[1],std::ios::binary);
    if(!f)throw std::runtime_error("missing code");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)),{});
    if(bytes.empty() || bytes.size()%8)throw std::runtime_error("truncated code");
    std::vector<uint64_t> words(bytes.size()/8);
    for(size_t i=0;i<bytes.size();++i)words[i/8]|=uint64_t(bytes[i])<<((i%8)*8);
    validate_code(words,std::stoul(argv[2]));
    std::cout<<"PASS decode "<<words.size()<<" instructions\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
