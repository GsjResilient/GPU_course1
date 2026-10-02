#pragma once
#include <cstdint>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iterator>

struct Reader {
    std::vector<uint8_t> data;
    size_t at=0;
    explicit Reader(const char* path){
        std::ifstream f(path,std::ios::binary|std::ios::ate);
        if(!f || f.tellg()<0 || uint64_t(f.tellg())>(uint64_t(256)<<20)+65536)
            throw std::runtime_error("invalid request file size");
        f.seekg(0);data.assign(std::istreambuf_iterator<char>(f),{});
    }
    uint64_t number(unsigned n){
        if(at>data.size() || n>data.size()-at)throw std::runtime_error("truncated request");
        uint64_t v=0;for(unsigned i=0;i<n;++i)v|=uint64_t(data[at++])<<(8*i);return v;
    }
    std::vector<uint8_t> bytes(size_t n){
        if(at>data.size() || n>data.size()-at)throw std::runtime_error("truncated buffer");
        auto begin=data.begin()+at;at+=n;return {begin,begin+n};
    }
};
inline void write_number(std::ostream& f,uint64_t value,unsigned n){
    for(unsigned j=0;j<n;++j)f.put(char((value>>(8*j))&255));
}
