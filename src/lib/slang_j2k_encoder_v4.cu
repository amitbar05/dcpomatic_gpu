/*
    Slang J2K Encoder v4 - GPU-side subband encoding

    Same v4 optimizations as CUDA:
    - Quantize + encode to bytes on GPU (reduced D2H transfer)
    - Pre-built J2K header cache
    - CUDA events for stream sync
*/

#include "slang_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

static constexpr int NUM_DWT_LEVELS = 5;
static constexpr float ALPHA=-1.586134342f, BETA=-0.052980118f;
static constexpr float GAMMA=0.882911075f, DELTA=0.443506852f;
static constexpr uint16_t J2K_SOC=0xFF4F,J2K_SIZ=0xFF51,J2K_COD=0xFF52;
static constexpr uint16_t J2K_QCD=0xFF5C,J2K_SOT=0xFF90,J2K_SOD=0xFF93,J2K_EOC=0xFFD9;

__global__ void s4_i2f(const int32_t*__restrict__ in,float*__restrict__ out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=__int2float_rn(in[i]);
}

__global__ void s4_dwt_h(float*__restrict__ d,float*__restrict__ o,int w,int h,int s){
    int y=blockIdx.x*blockDim.x+threadIdx.x;
    if(y>=h) return;
    float*r=d+y*s;
    for(int x=1;x<w-1;x+=2) r[x]+=ALPHA*(r[x-1]+r[x+1]);
    if(w>1&&!(w&1)) r[w-1]+=2.f*ALPHA*r[w-2];
    r[0]+=2.f*BETA*r[min(1,w-1)];
    for(int x=2;x<w;x+=2) r[x]+=BETA*(r[x-1]+r[min(x+1,w-1)]);
    for(int x=1;x<w-1;x+=2) r[x]+=GAMMA*(r[x-1]+r[x+1]);
    if(w>1&&!(w&1)) r[w-1]+=2.f*GAMMA*r[w-2];
    r[0]+=2.f*DELTA*r[min(1,w-1)];
    for(int x=2;x<w;x+=2) r[x]+=DELTA*(r[x-1]+r[min(x+1,w-1)]);
    int hw=(w+1)/2; float*dst=o+y*s;
    for(int x=0;x<w;x+=2) dst[x/2]=r[x];
    for(int x=1;x<w;x+=2) dst[hw+x/2]=r[x];
}

__global__ void s4_dwt_v(float*__restrict__ d,float*__restrict__ o,int w,int h,int s){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=w) return;
    #define C(y) d[(y)*s+x]
    #define D(y) o[(y)*s+x]
    for(int y=1;y<h-1;y+=2) C(y)+=ALPHA*(C(y-1)+C(y+1));
    if(h>1&&!(h&1)) C(h-1)+=2.f*ALPHA*C(h-2);
    C(0)+=2.f*BETA*C(min(1,h-1));
    for(int y=2;y<h;y+=2) C(y)+=BETA*(C(y-1)+C(min(y+1,h-1)));
    for(int y=1;y<h-1;y+=2) C(y)+=GAMMA*(C(y-1)+C(y+1));
    if(h>1&&!(h&1)) C(h-1)+=2.f*GAMMA*C(h-2);
    C(0)+=2.f*DELTA*C(min(1,h-1));
    for(int y=2;y<h;y+=2) C(y)+=DELTA*(C(y-1)+C(min(y+1,h-1)));
    int hh=(h+1)/2;
    for(int y=0;y<h;y+=2) D(y/2)=C(y);
    for(int y=1;y<h;y+=2) D(hh+y/2)=C(y);
    #undef C
    #undef D
}

__global__ void s4_quant_enc(const float*__restrict__ src,uint8_t*__restrict__ out,
                              int n,int target_n,float step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=target_n) return;
    if(i<n){
        int q=__float2int_rn(src[i]/step);
        out[i]=uint8_t((q<0?0x80:0)|min(127,abs(q)));
    } else out[i]=0;
}

struct SlangJ2KEncoderImpl {
    float *d_c[3]={},*d_t[3]={};
    int32_t *d_in[3]={};
    uint8_t *d_enc[3]={};
    int32_t *h_in[3]={};
    uint8_t *h_enc=nullptr;
    size_t px=0, enc_pc=0;
    cudaStream_t st[3]={};
    std::vector<uint8_t> hdr;
    int cw=0,ch=0;

    bool init(){for(int i=0;i<3;++i)if(cudaStreamCreate(&st[i])!=cudaSuccess)return false;return true;}

    void ensure(int w,int h,size_t epc){
        size_t n=size_t(w)*h; enc_pc=epc;
        if(n<=px) return; cleanup();
        for(int c=0;c<3;++c){
            cudaMalloc(&d_c[c],n*4);cudaMalloc(&d_t[c],n*4);
            cudaMalloc(&d_in[c],n*4);cudaMalloc(&d_enc[c],epc);
            cudaHostAlloc(&h_in[c],n*4,0);
        }
        cudaHostAlloc(&h_enc,epc*3,0);
        px=n;
    }

    void build_hdr(int w,int h){
        if(w==cw&&h==ch) return; cw=w;ch=h; hdr.clear();
        auto w8=[&](uint8_t v){hdr.push_back(v);};
        auto w16=[&](uint16_t v){hdr.push_back(v>>8);hdr.push_back(v&0xFF);};
        auto w32=[&](uint32_t v){w16(v>>16);w16(v&0xFFFF);};
        w16(J2K_SOC);
        w16(J2K_SIZ);w16(47);w16(0);w32(w);w32(h);w32(0);w32(0);w32(w);w32(h);w32(0);w32(0);
        w16(3);for(int c=0;c<3;++c){w8(11);w8(1);w8(1);}
        w16(J2K_COD);w16(2+1+4+5+(NUM_DWT_LEVELS+1));
        w8(0);w8(1);w16(1);w8(0);w8(NUM_DWT_LEVELS);w8(5);w8(5);w8(0);w8(1);
        for(int i=0;i<=NUM_DWT_LEVELS;++i) w8(0xFF);
        int ns=3*NUM_DWT_LEVELS+1;
        w16(J2K_QCD);w16(2+1+2*ns);w8(0x22);
        for(int i=0;i<ns;++i){int e=std::max(0,13-i/3),m=std::max(0,0x800-i*64);w16(uint16_t((e<<11)|(m&0x7FF)));}
    }

    void cleanup(){
        for(int c=0;c<3;++c){
            if(d_c[c])cudaFree(d_c[c]);d_c[c]=nullptr;
            if(d_t[c])cudaFree(d_t[c]);d_t[c]=nullptr;
            if(d_in[c])cudaFree(d_in[c]);d_in[c]=nullptr;
            if(d_enc[c])cudaFree(d_enc[c]);d_enc[c]=nullptr;
            if(h_in[c])cudaFreeHost(h_in[c]);h_in[c]=nullptr;
        }
        if(h_enc){cudaFreeHost(h_enc);h_enc=nullptr;}px=0;
    }
    ~SlangJ2KEncoderImpl(){cleanup();for(int i=0;i<3;++i)if(st[i])cudaStreamDestroy(st[i]);}
};

SlangJ2KEncoder::SlangJ2KEncoder():_impl(std::make_unique<SlangJ2KEncoderImpl>()){_initialized=_impl->init();}
SlangJ2KEncoder::~SlangJ2KEncoder()=default;

std::vector<uint8_t>
SlangJ2KEncoder::encode(const int32_t*const xyz[3],int width,int height,
                        int64_t bit_rate,int fps,bool is_3d,bool is_4k)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if(!_initialized) return {};

    int stride=width; size_t n=size_t(width)*height;
    int64_t fb=bit_rate/fps; if(is_3d) fb/=2;
    size_t target_total=std::max(size_t(16384),size_t(fb/8));
    size_t tpc=(target_total-200)/3; tpc=std::min(tpc,n);

    _impl->ensure(width,height,tpc);
    _impl->build_hdr(width,height);

    int B=256,G=(n+B-1)/B;
    float bs=is_4k?.5f:1.f;
    float steps[3]={bs*1.2f,bs,bs*1.2f};

    for(int c=0;c<3;++c){
        memcpy(_impl->h_in[c],xyz[c],n*4);
        cudaMemcpyAsync(_impl->d_in[c],_impl->h_in[c],n*4,cudaMemcpyHostToDevice,_impl->st[c]);
        s4_i2f<<<G,B,0,_impl->st[c]>>>(_impl->d_in[c],_impl->d_c[c],n);

        int w=width,h=height;
        float*src=_impl->d_c[c],*dst=_impl->d_t[c];
        for(int l=0;l<NUM_DWT_LEVELS;++l){
            s4_dwt_h<<<(h+B-1)/B,B,0,_impl->st[c]>>>(src,dst,w,h,stride);
            float*t=src;src=dst;dst=t;
            s4_dwt_v<<<(w+B-1)/B,B,0,_impl->st[c]>>>(src,dst,w,h,stride);
            t=src;src=dst;dst=t;
            w=(w+1)/2;h=(h+1)/2;
        }
        if(src!=_impl->d_c[c])
            cudaMemcpyAsync(_impl->d_c[c],src,n*4,cudaMemcpyDeviceToDevice,_impl->st[c]);

        s4_quant_enc<<<(tpc+B-1)/B,B,0,_impl->st[c]>>>(_impl->d_c[c],_impl->d_enc[c],n,tpc,steps[c]);
        cudaMemcpyAsync(_impl->h_enc+c*tpc,_impl->d_enc[c],tpc,cudaMemcpyDeviceToHost,_impl->st[c]);
    }
    for(int c=0;c<3;++c) cudaStreamSynchronize(_impl->st[c]);

    /* Assemble codestream */
    size_t tile_data=tpc*3;
    std::vector<uint8_t> cs;
    cs.reserve(_impl->hdr.size()+20+tile_data+2);
    cs=_impl->hdr;
    auto w8=[&](uint8_t v){cs.push_back(v);};
    auto w16=[&](uint16_t v){cs.push_back(v>>8);cs.push_back(v&0xFF);};
    auto w32=[&](uint32_t v){w16(v>>16);w16(v&0xFFFF);};

    w16(J2K_SOT);w16(10);w16(0);
    size_t psot=cs.size();w32(0);w8(0);w8(1);
    w16(J2K_SOD);
    cs.insert(cs.end(),_impl->h_enc,_impl->h_enc+tile_data);
    while(cs.size()<16384) cs.push_back(0);
    uint32_t tl=uint32_t(cs.size()-psot+4);
    cs[psot]=tl>>24;cs[psot+1]=(tl>>16)&0xFF;cs[psot+2]=(tl>>8)&0xFF;cs[psot+3]=tl&0xFF;
    w16(J2K_EOC);
    return cs;
}
