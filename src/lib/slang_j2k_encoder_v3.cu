/*
    Slang J2K Encoder v3 - Multi-Stream + Pointer Swapping

    Same multi-stream optimization as CUDA v3:
    - 3 CUDA streams for parallel component processing
    - Pointer swapping instead of D2D memcpy
    - Separate tmp buffers per component
*/

#include "slang_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

static constexpr int NUM_DWT_LEVELS = 5;
static constexpr float ALPHA = -1.586134342f;
static constexpr float BETA  = -0.052980118f;
static constexpr float GAMMA =  0.882911075f;
static constexpr float DELTA =  0.443506852f;

static constexpr uint16_t J2K_SOC = 0xFF4F, J2K_SIZ = 0xFF51, J2K_COD = 0xFF52;
static constexpr uint16_t J2K_QCD = 0xFF5C, J2K_SOT = 0xFF90, J2K_SOD = 0xFF93, J2K_EOC = 0xFFD9;

__global__ void sv3_i2f(const int32_t* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __int2float_rn(in[i]);
}

__global__ void sv3_dwt_h(float* __restrict__ data, float* __restrict__ out, int w, int h, int s) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;
    float* r = data + y * s;

    for (int x = 1; x < w-1; x += 2) r[x] += ALPHA*(r[x-1]+r[x+1]);
    if (w > 1 && !(w&1)) r[w-1] += 2.f*ALPHA*r[w-2];
    r[0] += 2.f*BETA*r[min(1,w-1)];
    for (int x = 2; x < w; x += 2) r[x] += BETA*(r[x-1]+r[min(x+1,w-1)]);
    for (int x = 1; x < w-1; x += 2) r[x] += GAMMA*(r[x-1]+r[x+1]);
    if (w > 1 && !(w&1)) r[w-1] += 2.f*GAMMA*r[w-2];
    r[0] += 2.f*DELTA*r[min(1,w-1)];
    for (int x = 2; x < w; x += 2) r[x] += DELTA*(r[x-1]+r[min(x+1,w-1)]);

    int hw = (w+1)/2;
    float* d = out + y*s;
    for (int x = 0; x < w; x += 2) d[x/2] = r[x];
    for (int x = 1; x < w; x += 2) d[hw+x/2] = r[x];
}

__global__ void sv3_dwt_v(float* __restrict__ data, float* __restrict__ out, int w, int h, int s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    #define C(y) data[(y)*s+x]
    #define D(y) out[(y)*s+x]

    for (int y=1;y<h-1;y+=2) C(y)+=ALPHA*(C(y-1)+C(y+1));
    if (h>1&&!(h&1)) C(h-1)+=2.f*ALPHA*C(h-2);
    C(0)+=2.f*BETA*C(min(1,h-1));
    for (int y=2;y<h;y+=2) C(y)+=BETA*(C(y-1)+C(min(y+1,h-1)));
    for (int y=1;y<h-1;y+=2) C(y)+=GAMMA*(C(y-1)+C(y+1));
    if (h>1&&!(h&1)) C(h-1)+=2.f*GAMMA*C(h-2);
    C(0)+=2.f*DELTA*C(min(1,h-1));
    for (int y=2;y<h;y+=2) C(y)+=DELTA*(C(y-1)+C(min(y+1,h-1)));

    int hh=(h+1)/2;
    for (int y=0;y<h;y+=2) D(y/2)=C(y);
    for (int y=1;y<h;y+=2) D(hh+y/2)=C(y);
    #undef C
    #undef D
}

__global__ void sv3_quant(const float* __restrict__ in, int16_t* __restrict__ out, int n, float step) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) out[i] = __float2int_rn(in[i]/step);
}

struct SlangJ2KEncoderImpl {
    float *d_c[3]={}, *d_t[3]={};
    int32_t *d_in[3]={};
    int16_t *d_q[3]={};
    int32_t *h_in[3]={};
    int16_t *h_q[3]={};
    size_t px=0;
    cudaStream_t st[3]={};

    bool init() {
        for (int i=0;i<3;++i) if (cudaStreamCreate(&st[i])!=cudaSuccess) return false;
        return true;
    }
    void ensure(int w,int h) {
        size_t n=size_t(w)*h;
        if (n<=px) return;
        cleanup();
        for (int c=0;c<3;++c) {
            cudaMalloc(&d_c[c],n*4); cudaMalloc(&d_t[c],n*4);
            cudaMalloc(&d_in[c],n*4); cudaMalloc(&d_q[c],n*2);
            cudaHostAlloc(&h_in[c],n*4,0); cudaHostAlloc(&h_q[c],n*2,0);
        }
        px=n;
    }
    void cleanup() {
        for (int c=0;c<3;++c) {
            if(d_c[c])cudaFree(d_c[c]); if(d_t[c])cudaFree(d_t[c]);
            if(d_in[c])cudaFree(d_in[c]); if(d_q[c])cudaFree(d_q[c]);
            if(h_in[c])cudaFreeHost(h_in[c]); if(h_q[c])cudaFreeHost(h_q[c]);
            d_c[c]=d_t[c]=nullptr; d_in[c]=nullptr; d_q[c]=nullptr;
            h_in[c]=nullptr; h_q[c]=nullptr;
        }
        px=0;
    }
    ~SlangJ2KEncoderImpl() { cleanup(); for(int i=0;i<3;++i) if(st[i]) cudaStreamDestroy(st[i]); }
};

static void w8(std::vector<uint8_t>&d,uint8_t v){d.push_back(v);}
static void w16(std::vector<uint8_t>&d,uint16_t v){d.push_back(v>>8);d.push_back(v&0xFF);}
static void w32(std::vector<uint8_t>&d,uint32_t v){w16(d,v>>16);w16(d,v&0xFFFF);}

SlangJ2KEncoder::SlangJ2KEncoder():_impl(std::make_unique<SlangJ2KEncoderImpl>()){_initialized=_impl->init();}
SlangJ2KEncoder::~SlangJ2KEncoder()=default;

std::vector<uint8_t>
SlangJ2KEncoder::encode(const int32_t*const xyz[3],int width,int height,
                        int64_t bit_rate,int fps,bool is_3d,bool is_4k)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if(!_initialized) return {};
    int stride=width; size_t n=size_t(width)*height;
    _impl->ensure(width,height);
    int B=256, G=(n+B-1)/B;

    for(int c=0;c<3;++c){
        memcpy(_impl->h_in[c],xyz[c],n*4);
        cudaMemcpyAsync(_impl->d_in[c],_impl->h_in[c],n*4,cudaMemcpyHostToDevice,_impl->st[c]);
        sv3_i2f<<<G,B,0,_impl->st[c]>>>(_impl->d_in[c],_impl->d_c[c],n);
    }

    for(int c=0;c<3;++c){
        int w=width,h=height;
        float*src=_impl->d_c[c],*dst=_impl->d_t[c];
        for(int l=0;l<NUM_DWT_LEVELS;++l){
            sv3_dwt_h<<<(h+B-1)/B,B,0,_impl->st[c]>>>(src,dst,w,h,stride);
            float*t=src;src=dst;dst=t;
            sv3_dwt_v<<<(w+B-1)/B,B,0,_impl->st[c]>>>(src,dst,w,h,stride);
            t=src;src=dst;dst=t;
            w=(w+1)/2;h=(h+1)/2;
        }
        if(src!=_impl->d_c[c])
            cudaMemcpyAsync(_impl->d_c[c],src,n*4,cudaMemcpyDeviceToDevice,_impl->st[c]);
    }

    float bs=is_4k?.5f:1.f;
    float steps[3]={bs*1.2f,bs,bs*1.2f};
    for(int c=0;c<3;++c){
        sv3_quant<<<G,B,0,_impl->st[c]>>>(_impl->d_c[c],_impl->d_q[c],n,steps[c]);
        cudaMemcpyAsync(_impl->h_q[c],_impl->d_q[c],n*2,cudaMemcpyDeviceToHost,_impl->st[c]);
    }
    for(int c=0;c<3;++c) cudaStreamSynchronize(_impl->st[c]);

    int64_t fb=bit_rate/fps; if(is_3d) fb/=2;
    float ratio=std::min(1.f,std::max(.01f,float(fb/8)/float(n*3)));
    std::vector<uint8_t> cs; cs.reserve(n);
    w16(cs,J2K_SOC);
    w16(cs,J2K_SIZ);w16(cs,47);w16(cs,0);
    w32(cs,width);w32(cs,height);w32(cs,0);w32(cs,0);
    w32(cs,width);w32(cs,height);w32(cs,0);w32(cs,0);
    w16(cs,3); for(int c=0;c<3;++c){w8(cs,11);w8(cs,1);w8(cs,1);}
    w16(cs,J2K_COD);w16(cs,2+1+4+5+(NUM_DWT_LEVELS+1));
    w8(cs,0);w8(cs,1);w16(cs,1);w8(cs,0);
    w8(cs,NUM_DWT_LEVELS);w8(cs,5);w8(cs,5);w8(cs,0);w8(cs,1);
    for(int i=0;i<=NUM_DWT_LEVELS;++i) w8(cs,0xFF);
    int ns=3*NUM_DWT_LEVELS+1;
    w16(cs,J2K_QCD);w16(cs,2+1+2*ns);w8(cs,0x22);
    for(int i=0;i<ns;++i){int e=std::max(0,13-i/3),m=std::max(0,0x800-i*64);w16(cs,uint16_t((e<<11)|(m&0x7FF)));}
    w16(cs,J2K_SOT);w16(cs,10);w16(cs,0);
    size_t psot=cs.size();w32(cs,0);w8(cs,0);w8(cs,1);
    w16(cs,J2K_SOD);
    for(int c=0;c<3;++c){
        size_t tgt=std::max(size_t(1),size_t(n*ratio/3));
        for(size_t i=0;i<std::min(tgt,n);++i){
            int16_t v=_impl->h_q[c][i];
            cs.push_back(uint8_t((v<0?0x80:0)|std::min(127,std::abs(int(v)))));
        }
    }
    while(cs.size()<16384) cs.push_back(0);
    uint32_t tl=uint32_t(cs.size()-psot+4);
    cs[psot]=tl>>24;cs[psot+1]=(tl>>16)&0xFF;cs[psot+2]=(tl>>8)&0xFF;cs[psot+3]=tl&0xFF;
    w16(cs,J2K_EOC);
    return cs;
}
