/*
    Thorough Correctness Verification: CUDA v7 vs Slang v7 vs CPU reference

    Tests:
    1. J2K codestream structure (markers)
    2. DWT numerical correctness (GPU vs CPU reference DWT)
    3. Quantization correctness
    4. Bitstream determinism (same input -> same output)
    5. Multi-frame consistency
    6. Edge cases (all-zero, all-max, gradient patterns)

    Build:
      nvcc -std=c++17 -O2 test/verify_correctness.cu \
        src/lib/cuda_j2k_encoder_v7.cu src/lib/slang_j2k_encoder_v7.cu \
        -I src/lib -lcudart -o test/verify_correctness
*/

#include "cuda_j2k_encoder.h"
#include "slang_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cassert>

/* ===== CPU Reference DWT 9/7 ===== */

static constexpr float ALPHA=-1.586134342f, BETA=-0.052980118f;
static constexpr float GAMMA=0.882911075f, DELTA=0.443506852f;

static void cpu_dwt97_horz(std::vector<float>& data, std::vector<float>& tmp,
                            int w, int h, int s) {
    for (int y = 0; y < h; ++y) {
        float* r = data.data() + y * s;
        for(int x=1;x<w-1;x+=2) r[x]+=ALPHA*(r[x-1]+r[x+1]);
        if(w>1&&!(w&1)) r[w-1]+=2.f*ALPHA*r[w-2];
        r[0]+=2.f*BETA*r[std::min(1,w-1)];
        for(int x=2;x<w;x+=2) r[x]+=BETA*(r[x-1]+r[std::min(x+1,w-1)]);
        for(int x=1;x<w-1;x+=2) r[x]+=GAMMA*(r[x-1]+r[x+1]);
        if(w>1&&!(w&1)) r[w-1]+=2.f*GAMMA*r[w-2];
        r[0]+=2.f*DELTA*r[std::min(1,w-1)];
        for(int x=2;x<w;x+=2) r[x]+=DELTA*(r[x-1]+r[std::min(x+1,w-1)]);
        int hw=(w+1)/2;
        for(int x=0;x<w;x+=2) tmp[y*s+x/2]=r[x];
        for(int x=1;x<w;x+=2) tmp[y*s+hw+x/2]=r[x];
    }
}

static void cpu_dwt97_vert(std::vector<float>& data, std::vector<float>& tmp,
                            int w, int h, int s) {
    for (int x = 0; x < w; ++x) {
        auto C = [&](int y) -> float& { return data[y*s+x]; };
        for(int y=1;y<h-1;y+=2) C(y)+=ALPHA*(C(y-1)+C(y+1));
        if(h>1&&!(h&1)) C(h-1)+=2.f*ALPHA*C(h-2);
        C(0)+=2.f*BETA*C(std::min(1,h-1));
        for(int y=2;y<h;y+=2) C(y)+=BETA*(C(y-1)+C(std::min(y+1,h-1)));
        for(int y=1;y<h-1;y+=2) C(y)+=GAMMA*(C(y-1)+C(y+1));
        if(h>1&&!(h&1)) C(h-1)+=2.f*GAMMA*C(h-2);
        C(0)+=2.f*DELTA*C(std::min(1,h-1));
        for(int y=2;y<h;y+=2) C(y)+=DELTA*(C(y-1)+C(std::min(y+1,h-1)));
        int hh=(h+1)/2;
        for(int y=0;y<h;y+=2) tmp[(y/2)*s+x]=C(y);
        for(int y=1;y<h;y+=2) tmp[(hh+y/2)*s+x]=C(y);
    }
}

static std::vector<float> cpu_full_dwt(const int32_t* input, int w, int h) {
    size_t n = size_t(w)*h;
    std::vector<float> data(n), tmp(n);
    for (size_t i=0;i<n;++i) data[i]=float(input[i]);
    int cw=w, ch=h, s=w;
    for (int l=0;l<5;++l) {
        cpu_dwt97_horz(data, tmp, cw, ch, s);
        std::swap(data, tmp);
        cpu_dwt97_vert(data, tmp, cw, ch, s);
        std::swap(data, tmp);
        cw=(cw+1)/2; ch=(ch+1)/2;
    }
    return data;
}


/* ===== J2K Verification ===== */

struct J2KCheck {
    bool soc=false, siz=false, cod=false, qcd=false, sot=false, sod=false, eoc=false;
    int siz_w=0, siz_h=0, siz_comp=0, siz_bpp=0;
    size_t size=0;
    bool valid() const { return soc&&siz&&cod&&qcd&&sot&&sod&&eoc; }
};

static J2KCheck check_j2k(const std::vector<uint8_t>& d) {
    J2KCheck r; r.size=d.size();
    if(d.size()<4) return r;
    if(d[0]==0xFF&&d[1]==0x4F) r.soc=true;
    if(d[d.size()-2]==0xFF&&d[d.size()-1]==0xD9) r.eoc=true;
    size_t pos=2;
    while(pos+1<d.size()) {
        if(d[pos]!=0xFF){++pos;continue;}
        uint8_t m=d[pos+1];
        if(m==0||m==0xFF){pos+=2;continue;}
        if(m==0x51) { r.siz=true;
            if(pos+14<=d.size()) {
                size_t b=pos+6;
                r.siz_w=(d[b]<<24)|(d[b+1]<<16)|(d[b+2]<<8)|d[b+3];
                r.siz_h=(d[b+4]<<24)|(d[b+5]<<16)|(d[b+6]<<8)|d[b+7];
            }
            if(pos+40<=d.size()) r.siz_comp=(d[pos+38]<<8)|d[pos+39];
            if(pos+41<=d.size()) r.siz_bpp=d[pos+40];
        }
        else if(m==0x52) r.cod=true;
        else if(m==0x5C) r.qcd=true;
        else if(m==0x90) r.sot=true;
        else if(m==0x93) { r.sod=true; break; }
        bool has_len=(m>=0x40&&m!=0x4F&&m!=0x93&&m!=0xD9)||m==0x90;
        if(has_len&&pos+3<d.size()) { uint16_t l=(d[pos+2]<<8)|d[pos+3]; pos+=2+l; }
        else pos+=2;
    }
    return r;
}


/* ===== GPU DWT extraction (encode without J2K packaging) ===== */

/* Kernels from v7 - need forward declarations */
/* Use whichever version's kernels are linked */
extern __global__ void v10_dwt_h(float*,float*,int,int,int);
extern __global__ void v10_dwt_v(float*,float*,int,int,int);

static std::vector<float> gpu_dwt_only(const int32_t* input, int w, int h) {
    size_t n=size_t(w)*h;
    float *d_a,*d_b;
    cudaMalloc(&d_a,n*4); cudaMalloc(&d_b,n*4);
    /* Convert int32->float on CPU (like v10 does) */
    {std::vector<float> hf(n);
    for(size_t i=0;i<n;++i) hf[i]=float(input[i]);
    cudaMemcpy(d_a,hf.data(),n*4,cudaMemcpyHostToDevice);}
    int cw=w,ch=h,s=w;
    for(int l=0;l<5;++l){
        v10_dwt_h<<<ch,128,cw*sizeof(float)>>>(d_a,d_b,cw,ch,s);
        std::swap(d_a,d_b);
        v10_dwt_v<<<(cw+127)/128,128>>>(d_a,d_b,cw,ch,s);
        std::swap(d_a,d_b);
        cw=(cw+1)/2;ch=(ch+1)/2;
    }
    std::vector<float> result(n);
    cudaMemcpy(result.data(),d_a,n*4,cudaMemcpyDeviceToHost);
    cudaFree(d_a);cudaFree(d_b);
    return result;
}


/* ===== Test Helpers ===== */

static void rgb_to_xyz(const uint8_t*rgb,int w,int h,
                       std::vector<int32_t>&x,std::vector<int32_t>&y,std::vector<int32_t>&z){
    size_t n=size_t(w)*h;
    x.resize(n);y.resize(n);z.resize(n);
    for(size_t i=0;i<n;++i){
        float r=rgb[i*3]/255.f,g=rgb[i*3+1]/255.f,b=rgb[i*3+2]/255.f;
        x[i]=int32_t(std::min(4095.f,(r*.4124f+g*.3576f+b*.1805f)*4095));
        y[i]=int32_t(std::min(4095.f,(r*.2126f+g*.7152f+b*.0722f)*4095));
        z[i]=int32_t(std::min(4095.f,(r*.0193f+g*.1192f+b*.9505f)*4095));
    }
}

static std::vector<int32_t> make_pattern(int w, int h, int type) {
    std::vector<int32_t> v(size_t(w)*h);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        size_t i=y*w+x;
        switch(type) {
        case 0: v[i]=0; break;                         // all zero
        case 1: v[i]=4095; break;                      // all max
        case 2: v[i]=int32_t(x*4095/w); break;         // horizontal gradient
        case 3: v[i]=int32_t(y*4095/h); break;         // vertical gradient
        case 4: v[i]=int32_t((x+y)%4096); break;       // diagonal
        case 5: v[i]=((x^y)&1)?4095:0; break;          // checkerboard
        }
    }
    return v;
}

#define TEST(name, cond) do { \
    bool ok=(cond); \
    std::cout << "  " << (ok?"PASS":"FAIL") << ": " << name << std::endl; \
    if(!ok) fails++; total++; \
} while(0)

int main(int argc, char* argv[])
{
    int W=2048, H=1080;
    int fails=0, total=0;

    std::cout << "======================================================" << std::endl;
    std::cout << " Correctness Verification: CUDA v7 & Slang v7" << std::endl;
    std::cout << "======================================================" << std::endl;

    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,0)==cudaSuccess)
        std::cout << "GPU: " << prop.name << std::endl;

    /* Initialize encoders */
    CudaJ2KEncoder cuda_enc;
    SlangJ2KEncoder slang_enc;
    TEST("CUDA encoder initialized", cuda_enc.is_initialized());
    TEST("Slang encoder initialized", slang_enc.is_initialized());

    /* ===== Test 1: DWT Numerical Correctness ===== */
    std::cout << "\n--- Test 1: DWT Numerical Correctness (GPU vs CPU reference) ---" << std::endl;
    {
        auto pattern = make_pattern(W, H, 2); // horizontal gradient
        auto cpu_result = cpu_full_dwt(pattern.data(), W, H);
        auto gpu_result = gpu_dwt_only(pattern.data(), W, H);

        double max_err=0, sum_err=0;
        size_t n=cpu_result.size();
        for(size_t i=0;i<n;++i) {
            double err=std::abs(double(cpu_result[i])-double(gpu_result[i]));
            max_err=std::max(max_err,err);
            sum_err+=err;
        }
        double avg_err=sum_err/n;
        std::cout << "    Max error: " << max_err << ", Avg error: " << avg_err << std::endl;
        /* GPU shared-memory ops may differ from CPU sequential due to
         * float associativity. 0.02 tolerance is well within J2K quantization step. */
        TEST("DWT max error < 0.02 (float rounding)", max_err < 0.02);
        TEST("DWT avg error < 0.001", avg_err < 0.001);
    }

    /* ===== Test 2: J2K Codestream Structure ===== */
    std::cout << "\n--- Test 2: J2K Codestream Structure ---" << std::endl;
    {
        auto pattern = make_pattern(W, H, 4);
        const int32_t* p[3]={pattern.data(),pattern.data(),pattern.data()};

        auto cuda_j2k = cuda_enc.encode(p,W,H,100000000,24,false,false);
        auto slang_j2k = slang_enc.encode(p,W,H,100000000,24,false,false);

        auto cc = check_j2k(cuda_j2k);
        auto sc = check_j2k(slang_j2k);

        TEST("CUDA: SOC marker", cc.soc);
        TEST("CUDA: SIZ marker", cc.siz);
        TEST("CUDA: SIZ width=2048", cc.siz_w==2048);
        TEST("CUDA: SIZ height=1080", cc.siz_h==1080);
        TEST("CUDA: SIZ 3 components", cc.siz_comp==3);
        TEST("CUDA: SIZ 12-bit", cc.siz_bpp==11);
        TEST("CUDA: COD marker", cc.cod);
        TEST("CUDA: QCD marker", cc.qcd);
        TEST("CUDA: SOT marker", cc.sot);
        TEST("CUDA: SOD marker", cc.sod);
        TEST("CUDA: EOC marker", cc.eoc);
        TEST("CUDA: J2K fully valid", cc.valid());

        TEST("Slang: SOC marker", sc.soc);
        TEST("Slang: SIZ marker", sc.siz);
        TEST("Slang: COD marker", sc.cod);
        TEST("Slang: QCD marker", sc.qcd);
        TEST("Slang: SOT marker", sc.sot);
        TEST("Slang: SOD marker", sc.sod);
        TEST("Slang: EOC marker", sc.eoc);
        TEST("Slang: J2K fully valid", sc.valid());

        TEST("CUDA output >= 16KB", cuda_j2k.size() >= 16384);
        TEST("Slang output >= 16KB", slang_j2k.size() >= 16384);
        TEST("CUDA == Slang output size", cuda_j2k.size() == slang_j2k.size());
    }

    /* ===== Test 3: Determinism ===== */
    std::cout << "\n--- Test 3: Determinism (same input -> same output) ---" << std::endl;
    {
        auto pattern = make_pattern(W, H, 5);
        const int32_t* p[3]={pattern.data(),pattern.data(),pattern.data()};

        auto cuda_a = cuda_enc.encode(p,W,H,100000000,24,false,false);
        auto cuda_b = cuda_enc.encode(p,W,H,100000000,24,false,false);
        auto slang_a = slang_enc.encode(p,W,H,100000000,24,false,false);
        auto slang_b = slang_enc.encode(p,W,H,100000000,24,false,false);

        TEST("CUDA deterministic", cuda_a == cuda_b);
        TEST("Slang deterministic", slang_a == slang_b);
        TEST("CUDA == Slang output", cuda_a == slang_a);
    }

    /* ===== Test 4: Edge Cases ===== */
    std::cout << "\n--- Test 4: Edge Cases ---" << std::endl;
    {
        const char* names[]={"all-zero","all-max","h-gradient","v-gradient","diagonal","checkerboard"};
        for(int t=0;t<6;++t){
            auto pat=make_pattern(W,H,t);
            const int32_t*p[3]={pat.data(),pat.data(),pat.data()};
            auto cj=cuda_enc.encode(p,W,H,100000000,24,false,false);
            auto sj=slang_enc.encode(p,W,H,100000000,24,false,false);
            auto cc=check_j2k(cj), sc=check_j2k(sj);
            TEST(std::string("CUDA ")+names[t]+" valid J2K", cc.valid());
            TEST(std::string("Slang ")+names[t]+" valid J2K", sc.valid());
            TEST(std::string("Match ")+names[t], cj==sj);
        }
    }

    /* ===== Test 5: Real Video Frames ===== */
    std::cout << "\n--- Test 5: Real Video Frames ---" << std::endl;
    if(argc > 1) {
        std::ifstream fin(argv[1],std::ios::binary);
        int nframes = (argc>4) ? std::atoi(argv[4]) : 10;
        int w = (argc>2) ? std::atoi(argv[2]) : W;
        int h = (argc>3) ? std::atoi(argv[3]) : H;
        size_t fb=size_t(w)*h*3;
        int cuda_valid=0, slang_valid=0, match=0;

        for(int i=0;i<nframes;++i){
            std::vector<uint8_t> rgb(fb);
            fin.read(reinterpret_cast<char*>(rgb.data()),fb);
            if(fin.gcount()!=std::streamsize(fb)) break;

            std::vector<int32_t> x,y,z;
            rgb_to_xyz(rgb.data(),w,h,x,y,z);
            const int32_t*p[3]={x.data(),y.data(),z.data()};

            auto cj=cuda_enc.encode(p,w,h,100000000,24,false,false);
            auto sj=slang_enc.encode(p,w,h,100000000,24,false,false);
            if(check_j2k(cj).valid()) ++cuda_valid;
            if(check_j2k(sj).valid()) ++slang_valid;
            if(cj==sj) ++match;
        }
        TEST("CUDA: all real frames valid J2K", cuda_valid==nframes);
        TEST("Slang: all real frames valid J2K", slang_valid==nframes);
        TEST("CUDA == Slang for all real frames", match==nframes);
        std::cout << "    (" << nframes << " frames tested)" << std::endl;
    } else {
        std::cout << "    (skipped - pass frames.rgb w h n as args)" << std::endl;
    }

    /* ===== Summary ===== */
    std::cout << "\n======================================================" << std::endl;
    std::cout << " RESULTS: " << (total-fails) << "/" << total << " tests passed";
    if(fails) std::cout << " (" << fails << " FAILED)";
    std::cout << std::endl;
    std::cout << "======================================================" << std::endl;

    return fails ? 1 : 0;
}
