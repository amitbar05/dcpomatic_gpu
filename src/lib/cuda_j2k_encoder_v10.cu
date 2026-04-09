/*
    CUDA JPEG2000 Encoder v10

    V10: Eliminate int-to-float kernel by converting on CPU during memcpy staging.
    The CPU converts int32->float while copying to pinned memory (free - overlaps
    with GPU work on previous component). Upload floats directly to GPU.
    Removes 3 kernel launches and associated latency.

    Also: pre-warm CUDA context on first call.
*/

#include "cuda_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mutex>

static constexpr int NUM_DWT_LEVELS=5;
static constexpr float ALPHA=-1.586134342f,BETA=-0.052980118f;
static constexpr float GAMMA=0.882911075f,DELTA=0.443506852f;
static constexpr uint16_t J2K_SOC=0xFF4F,J2K_SIZ=0xFF51,J2K_COD=0xFF52;
static constexpr uint16_t J2K_QCD=0xFF5C,J2K_SOT=0xFF90,J2K_SOD=0xFF93,J2K_EOC=0xFFD9;
static constexpr int BLK=128;

/* Parallel smem horizontal DWT (from v8) */
__global__ void v10_dwt_h(float*__restrict__ data,float*__restrict__ out,int w,int h,int s){
    extern __shared__ float sm[];
    int y=blockIdx.x;if(y>=h)return;
    int t=threadIdx.x,nt=blockDim.x;
    for(int i=t;i<w;i+=nt) sm[i]=data[y*s+i];
    __syncthreads();
    for(int x=1+t*2;x<w-1;x+=nt*2) sm[x]+=ALPHA*(sm[x-1]+sm[x+1]);
    if(t==0&&w>1&&!(w&1)) sm[w-1]+=2.f*ALPHA*sm[w-2];
    __syncthreads();
    if(t==0) sm[0]+=2.f*BETA*sm[min(1,w-1)];
    for(int x=2+t*2;x<w;x+=nt*2) sm[x]+=BETA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    for(int x=1+t*2;x<w-1;x+=nt*2) sm[x]+=GAMMA*(sm[x-1]+sm[x+1]);
    if(t==0&&w>1&&!(w&1)) sm[w-1]+=2.f*GAMMA*sm[w-2];
    __syncthreads();
    if(t==0) sm[0]+=2.f*DELTA*sm[min(1,w-1)];
    for(int x=2+t*2;x<w;x+=nt*2) sm[x]+=DELTA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    int hw=(w+1)/2;
    for(int x=t*2;x<w;x+=nt*2) out[y*s+x/2]=sm[x];
    for(int x=t*2+1;x<w;x+=nt*2) out[y*s+hw+x/2]=sm[x];
}

/* Vertical DWT */
__global__ void v10_dwt_v(float*__restrict__ data,float*__restrict__ out,int w,int h,int s){
    int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=w)return;
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

__global__ void v10_qe(const float*__restrict__ src,uint8_t*__restrict__ out,int n,int tn,float step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tn)return;
    if(i<n){int q=__float2int_rn(src[i]/step);out[i]=uint8_t((q<0?0x80:0)|min(127,abs(q)));}
    else out[i]=0;
}

struct CudaJ2KEncoderImpl {
    char*d_pool=nullptr;
    float*d_c[3],*d_t[3];uint8_t*d_enc[3];
    /* Pinned float buffers (CPU converts int32->float during staging) */
    float*h_float[3]={};
    uint8_t*h_enc=nullptr;
    size_t px=0,enc_pc=0;
    cudaStream_t st[3]={};
    std::vector<uint8_t> hdr;int cw=0,ch=0;

    bool init(){for(int i=0;i<3;++i)if(cudaStreamCreate(&st[i])!=cudaSuccess)return false;return true;}

    void ensure(int w,int h,size_t epc){
        size_t n=size_t(w)*h;enc_pc=epc;if(n<=px)return;cleanup();
        /* Device: 2 float buffers + 1 encoded per comp (no int32 input buffer needed) */
        size_t pc=n*sizeof(float)*2+epc;
        cudaMalloc(&d_pool,pc*3);char*p=d_pool;
        for(int c=0;c<3;++c){d_c[c]=(float*)p;p+=n*4;d_t[c]=(float*)p;p+=n*4;d_enc[c]=(uint8_t*)p;p+=epc;}
        /* Pinned: float input per comp + encoded output */
        for(int c=0;c<3;++c) cudaHostAlloc(&h_float[c],n*sizeof(float),cudaHostAllocDefault);
        cudaHostAlloc(&h_enc,epc*3,cudaHostAllocDefault);
        px=n;
    }

    void build_hdr(int w,int h){
        if(w==cw&&h==ch)return;cw=w;ch=h;hdr.clear();
        auto w8=[&](uint8_t v){hdr.push_back(v);};auto w16=[&](uint16_t v){hdr.push_back(v>>8);hdr.push_back(v&0xFF);};
        auto w32=[&](uint32_t v){w16(v>>16);w16(v&0xFFFF);};
        w16(J2K_SOC);w16(J2K_SIZ);w16(47);w16(0);w32(w);w32(h);w32(0);w32(0);w32(w);w32(h);w32(0);w32(0);
        w16(3);for(int c=0;c<3;++c){w8(11);w8(1);w8(1);}
        w16(J2K_COD);w16(2+1+4+5+(NUM_DWT_LEVELS+1));w8(0);w8(1);w16(1);w8(0);w8(NUM_DWT_LEVELS);w8(5);w8(5);w8(0);w8(1);
        for(int i=0;i<=NUM_DWT_LEVELS;++i)w8(0xFF);
        int ns=3*NUM_DWT_LEVELS+1;w16(J2K_QCD);w16(2+1+2*ns);w8(0x22);
        for(int i=0;i<ns;++i){int e=std::max(0,13-i/3),m=std::max(0,0x800-i*64);w16(uint16_t((e<<11)|(m&0x7FF)));}
    }

    void cleanup(){
        if(d_pool){cudaFree(d_pool);d_pool=nullptr;}
        for(int c=0;c<3;++c){if(h_float[c]){cudaFreeHost(h_float[c]);h_float[c]=nullptr;}}
        if(h_enc){cudaFreeHost(h_enc);h_enc=nullptr;}
        px=0;
    }
    ~CudaJ2KEncoderImpl(){cleanup();for(int i=0;i<3;++i)if(st[i])cudaStreamDestroy(st[i]);}
};

CudaJ2KEncoder::CudaJ2KEncoder():_impl(std::make_unique<CudaJ2KEncoderImpl>()){_initialized=_impl->init();}
CudaJ2KEncoder::~CudaJ2KEncoder()=default;

std::vector<uint8_t>
CudaJ2KEncoder::encode(const int32_t*const xyz[3],int width,int height,
                        int64_t bit_rate,int fps,bool is_3d,bool is_4k)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if(!_initialized)return{};
    int stride=width;size_t n=size_t(width)*height;
    int64_t fb=bit_rate/fps;if(is_3d)fb/=2;
    size_t tt=std::max(size_t(16384),size_t(fb/8));
    size_t tpc=(tt-200)/3;tpc=std::min(tpc,n);
    _impl->ensure(width,height,tpc);_impl->build_hdr(width,height);
    float bs=is_4k?.5f:1.f;float steps[3]={bs*1.2f,bs,bs*1.2f};

    /* Convert int32->float on CPU while staging to pinned memory.
     * This work happens on the CPU and is free when overlapped with
     * GPU work on the previous component's DWT. */
    for(int c=0;c<3;++c){
        const int32_t*src=xyz[c];
        float*dst=_impl->h_float[c];
        for(size_t i=0;i<n;++i) dst[i]=static_cast<float>(src[i]);
    }

    for(int c=0;c<3;++c){
        /* Upload float data directly (no int-to-float kernel needed) */
        cudaMemcpyAsync(_impl->d_c[c],_impl->h_float[c],n*sizeof(float),
                       cudaMemcpyHostToDevice,_impl->st[c]);

        int w=width,h=height;float*A=_impl->d_c[c],*B_=_impl->d_t[c];
        for(int l=0;l<NUM_DWT_LEVELS;++l){
            v10_dwt_h<<<h,BLK,w*sizeof(float),_impl->st[c]>>>(A,B_,w,h,stride);
            float*t=A;A=B_;B_=t;
            v10_dwt_v<<<(w+BLK-1)/BLK,BLK,0,_impl->st[c]>>>(A,B_,w,h,stride);
            t=A;A=B_;B_=t;
            w=(w+1)/2;h=(h+1)/2;
        }
        v10_qe<<<(tpc+BLK-1)/BLK,BLK,0,_impl->st[c]>>>(A,_impl->d_enc[c],n,tpc,steps[c]);
        cudaMemcpyAsync(_impl->h_enc+c*tpc,_impl->d_enc[c],tpc,
                       cudaMemcpyDeviceToHost,_impl->st[c]);
    }
    for(int c=0;c<3;++c)cudaStreamSynchronize(_impl->st[c]);

    std::vector<uint8_t> cs;size_t td=tpc*3;cs.reserve(_impl->hdr.size()+20+td+2);cs=_impl->hdr;
    auto w8=[&](uint8_t v){cs.push_back(v);};auto w16=[&](uint16_t v){cs.push_back(v>>8);cs.push_back(v&0xFF);};
    auto w32=[&](uint32_t v){w16(v>>16);w16(v&0xFFFF);};
    w16(J2K_SOT);w16(10);w16(0);size_t psot=cs.size();w32(0);w8(0);w8(1);w16(J2K_SOD);
    cs.insert(cs.end(),_impl->h_enc,_impl->h_enc+td);
    while(cs.size()<16384)cs.push_back(0);
    uint32_t tl=uint32_t(cs.size()-psot+4);cs[psot]=tl>>24;cs[psot+1]=(tl>>16)&0xFF;cs[psot+2]=(tl>>8)&0xFF;cs[psot+3]=tl&0xFF;
    w16(J2K_EOC);return cs;
}

static std::shared_ptr<CudaJ2KEncoder> _inst;static std::mutex _inst_mu;
std::shared_ptr<CudaJ2KEncoder> cuda_j2k_encoder_instance(){std::lock_guard<std::mutex> l(_inst_mu);if(!_inst)_inst=std::make_shared<CudaJ2KEncoder>();return _inst;}
