// Independent host binary32 reference: no compiler or simulator headers.
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#pragma STDC FENV_ACCESS ON

static uint32_t word(const std::string& s) {
    if(s.empty() || s.size()>8 || s.find_first_not_of("0123456789abcdefABCDEF")!=std::string::npos)
        throw std::runtime_error("expected u32 hex word");
    return static_cast<uint32_t>(std::stoul(s,nullptr,16));
}
int main() {
    try {
        if(std::fesetround(FE_TONEAREST)!=0) throw std::runtime_error("rounding mode unavailable");
        std::string line;
        while(std::getline(std::cin,line)) {
            std::istringstream input(line);std::string op,sa,sb,extra;
            if(!(input>>op>>sa>>sb) || (input>>extra)) throw std::runtime_error("invalid oracle record");
            uint32_t ua=word(sa),ub=word(sb),answer=0;float fa,fb;
            std::memcpy(&fa,&ua,4);std::memcpy(&fb,&ub,4);
            volatile float a=fa,b=fb,r=0;
            bool arithmetic=true;
            if(op=="add")r=a+b;
            else if(op=="sub")r=a-b;
            else if(op=="mul")r=a*b;
            else if(op=="neg")r=-a;
            else {
                arithmetic=false;
                if(op=="eq")answer=a==b;
                else if(op=="ne")answer=a!=b;
                else if(op=="lt")answer=a<b;
                else if(op=="le")answer=a<=b;
                else if(op=="gt")answer=a>b;
                else if(op=="ge")answer=a>=b;
                else throw std::runtime_error("unknown operation");
            }
            if(arithmetic){float value=r;std::memcpy(&answer,&value,4);}
            std::cout<<std::hex<<std::setw(8)<<std::setfill('0')<<answer<<'\n';
        }
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
