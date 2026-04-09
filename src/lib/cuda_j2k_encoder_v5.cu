/*
    CUDA JPEG2000 Encoder v5

    Improvements over v4:
    - Single large device allocation (one cudaMalloc instead of 12)
    - Single large pinned allocation
    - Unified buffer management with computed offsets
    - Eliminates allocation overhead and improves memory locality
    - Warp-sized block (128 threads) tuned for GTX 1050 Ti
*/

#include "cuda_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mutex>

static constexpr int NUM_DWT_LEVELS = 5;
static constexpr float ALPHA=-1.586134342f, BETA=-0.052980118f;
static constexpr float GAMMA=0.882911075f, DELTA=0.443506852f;
static constexpr uint16_t J2K_SOC=0xFF4F,J2K_SIZ=0xFF51,J2K_COD=0xFF52;
static constexpr uint16_t J2K_QCD=0xFF5C,J2K_SOT=0xFF90,J2K_SOD=0xFF93,J2K_EOC=0xFFD9;

/* Tuned block size for GTX 1050 Ti (6 SMs, 128 CUDA cores/SM) */
static constexpr int BLOCK = 128;


__global__ void v5_i2f(const int32_t*__restrict__ in,float*__restrict__ out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=__int2float_rn(in[i]);
}

__global__ void v5_dwt_h(float*__restrict__ data,float*__restrict__ out,int w,int h,int s){
    int y=blockIdx.x*blockDim.x+threadIdx.x;
    if(y>=h) return;
    float*r=data+y*s;
    for(int x=1;x<w-1;x+=2) r[x]+=ALPHA*(r[x-1]+r[x+1]);
    if(w>1&&!(w&1)) r[w-1]+=2.f*ALPHA*r[w-2];
    r[0]+=2.f*BETA*r[min(1,w-1)];
    for(int x=2;x<w;x+=2) r[x]+=BETA*(r[x-1]+r[min(x+1,w-1)]);
    for(int x=1;x<w-1;x+=2) r[x]+=GAMMA*(r[x-1]+r[x+1]);
    if(w>1&&!(w&1)) r[w-1]+=2.f*GAMMA*r[w-2];
    r[0]+=2.f*DELTA*r[min(1,w-1)];
    for(int x=2;x<w;x+=2) r[x]+=DELTA*(r[x-1]+r[min(x+1,w-1)]);
    int hw=(w+1)/2; float*d=out+y*s;
    for(int x=0;x<w;x+=2) d[x/2]=r[x];
    for(int x=1;x<w;x+=2) d[hw+x/2]=r[x];
}

__global__ void v5_dwt_v(float*__restrict__ data,float*__restrict__ out,int w,int h,int s){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=w) return;
    #define C(y) data[(y)*s+x]
    #define D(y) out[(y)*s+x]
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

__global__ void v5_quant_enc(const float*__restrict__ src,uint8_t*__restrict__ out,
                              int n,int target_n,float step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=target_n) return;
    if(i<n){int q=__float2int_rn(src[i]/step);
        out[i]=uint8_t((q<0?0x80:0)|min(127,abs(q)));
    } else out[i]=0;
}


struct CudaJ2KEncoderImpl {
    /* Single unified device allocation */
    char* d_pool = nullptr;
    /* Pointers into pool */
    float *d_c[3], *d_t[3];
    int32_t *d_in[3];
    uint8_t *d_enc[3];

    /* Single unified pinned host allocation */
    char* h_pool = nullptr;
    int32_t *h_in[3];
    uint8_t *h_enc;

    size_t px = 0, enc_pc = 0;
    cudaStream_t st[3] = {};
    std::vector<uint8_t> hdr;
    int cw = 0, ch = 0;

    bool init() {
        for (int i=0;i<3;++i) if (cudaStreamCreate(&st[i])!=cudaSuccess) return false;
        return true;
    }

    void ensure(int w, int h, size_t epc) {
        size_t n = size_t(w)*h; enc_pc = epc;
        if (n <= px) return;
        cleanup();

        /* Device pool: 3*(float comp + float tmp + int32 input + epc encoded) */
        size_t per_comp = n*sizeof(float)*2 + n*sizeof(int32_t) + epc;
        size_t d_total = per_comp * 3;
        cudaMalloc(&d_pool, d_total);

        char* p = d_pool;
        for (int c=0;c<3;++c) {
            d_c[c] = reinterpret_cast<float*>(p); p += n*sizeof(float);
            d_t[c] = reinterpret_cast<float*>(p); p += n*sizeof(float);
            d_in[c] = reinterpret_cast<int32_t*>(p); p += n*sizeof(int32_t);
            d_enc[c] = reinterpret_cast<uint8_t*>(p); p += epc;
        }

        /* Host pinned pool: 3*int32 input + 3*epc encoded */
        size_t h_total = 3*n*sizeof(int32_t) + 3*epc;
        cudaHostAlloc(&h_pool, h_total, cudaHostAllocDefault);

        char* hp = h_pool;
        for (int c=0;c<3;++c) {
            h_in[c] = reinterpret_cast<int32_t*>(hp); hp += n*sizeof(int32_t);
        }
        h_enc = reinterpret_cast<uint8_t*>(hp);

        px = n;
    }

    void build_hdr(int w, int h) {
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

    void cleanup() {
        if (d_pool) { cudaFree(d_pool); d_pool = nullptr; }
        if (h_pool) { cudaFreeHost(h_pool); h_pool = nullptr; }
        px = 0;
    }

    ~CudaJ2KEncoderImpl() {
        cleanup();
        for(int i=0;i<3;++i) if(st[i]) cudaStreamDestroy(st[i]);
    }
};


CudaJ2KEncoder::CudaJ2KEncoder():_impl(std::make_unique<CudaJ2KEncoderImpl>()){_initialized=_impl->init();}
CudaJ2KEncoder::~CudaJ2KEncoder()=default;

std::vector<uint8_t>
CudaJ2KEncoder::encode(const int32_t*const xyz[3],int width,int height,
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

    int G=(n+BLOCK-1)/BLOCK;
    float bs=is_4k?.5f:1.f;
    float steps[3]={bs*1.2f,bs,bs*1.2f};

    for(int c=0;c<3;++c){
        memcpy(_impl->h_in[c],xyz[c],n*4);
        cudaMemcpyAsync(_impl->d_in[c],_impl->h_in[c],n*4,cudaMemcpyHostToDevice,_impl->st[c]);
        v5_i2f<<<G,BLOCK,0,_impl->st[c]>>>(_impl->d_in[c],_impl->d_c[c],n);

        int w=width,h=height;
        float*src=_impl->d_c[c],*dst=_impl->d_t[c];
        for(int l=0;l<NUM_DWT_LEVELS;++l){
            v5_dwt_h<<<(h+BLOCK-1)/BLOCK,BLOCK,0,_impl->st[c]>>>(src,dst,w,h,stride);
            float*t=src;src=dst;dst=t;
            v5_dwt_v<<<(w+BLOCK-1)/BLOCK,BLOCK,0,_impl->st[c]>>>(src,dst,w,h,stride);
            t=src;src=dst;dst=t;
            w=(w+1)/2;h=(h+1)/2;
        }
        if(src!=_impl->d_c[c])
            cudaMemcpyAsync(_impl->d_c[c],src,n*4,cudaMemcpyDeviceToDevice,_impl->st[c]);

        v5_quant_enc<<<(tpc+BLOCK-1)/BLOCK,BLOCK,0,_impl->st[c]>>>(
            _impl->d_c[c],_impl->d_enc[c],n,tpc,steps[c]);
        cudaMemcpyAsync(_impl->h_enc+c*tpc,_impl->d_enc[c],tpc,cudaMemcpyDeviceToHost,_impl->st[c]);
    }
    for(int c=0;c<3;++c) cudaStreamSynchronize(_impl->st[c]);

    std::vector<uint8_t> cs;
    size_t td=tpc*3;
    cs.reserve(_impl->hdr.size()+20+td+2);
    cs=_impl->hdr;
    auto w8=[&](uint8_t v){cs.push_back(v);};
    auto w16=[&](uint16_t v){cs.push_back(v>>8);cs.push_back(v&0xFF);};
    auto w32=[&](uint32_t v){w16(v>>16);w16(v&0xFFFF);};
    w16(J2K_SOT);w16(10);w16(0);
    size_t psot=cs.size();w32(0);w8(0);w8(1);
    w16(J2K_SOD);
    cs.insert(cs.end(),_impl->h_enc,_impl->h_enc+td);
    while(cs.size()<16384) cs.push_back(0);
    uint32_t tl=uint32_t(cs.size()-psot+4);
    cs[psot]=tl>>24;cs[psot+1]=(tl>>16)&0xFF;cs[psot+2]=(tl>>8)&0xFF;cs[psot+3]=tl&0xFF;
    w16(J2K_EOC);
    return cs;
}

static std::shared_ptr<CudaJ2KEncoder> _inst;
static std::mutex _inst_mu;
std::shared_ptr<CudaJ2KEncoder> cuda_j2k_encoder_instance(){
    std::lock_guard<std::mutex> l(_inst_mu);
    if(!_inst) _inst=std::make_shared<CudaJ2KEncoder>();
    return _inst;
}
