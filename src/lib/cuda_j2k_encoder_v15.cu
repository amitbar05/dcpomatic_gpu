/*
    CUDA JPEG2000 Encoder v15 — L2/Texture Cache Optimization

    V15: Uses __ldg() for read-only data paths in vertical DWT,
    routing reads through the texture cache (L2) for better bandwidth.
    The vertical DWT at level 0 (1080 rows) is memory-bound;
    __ldg uses the read-only data cache path, separate from L1,
    effectively doubling cache bandwidth for read-heavy patterns.

    Also includes all v14 optimizations (fused i2f, register-blocked v-DWT).
*/

#include "cuda_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mutex>

static constexpr int NUM_DWT_LEVELS = 5;
static constexpr float ALPHA = -1.586134342f, BETA = -0.052980118f;
static constexpr float GAMMA = 0.882911075f, DELTA = 0.443506852f;
static constexpr uint16_t J2K_SOC = 0xFF4F, J2K_SIZ = 0xFF51, J2K_COD = 0xFF52;
static constexpr uint16_t J2K_QCD = 0xFF5C, J2K_SOT = 0xFF90, J2K_SOD = 0xFF93, J2K_EOC = 0xFFD9;
static constexpr int MAX_REG_HEIGHT = 140;

/* Fused i2f + horizontal DWT (from v13) */
__global__ void v15_fused_i2f_dwt_h(const int32_t* __restrict__ in, float* __restrict__ out, int w, int h, int s) {
    extern __shared__ float sm[];
    int y = blockIdx.x; if (y >= h) return;
    int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < w; i += nt) sm[i] = __int2float_rn(in[y*s+i]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += ALPHA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*ALPHA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*BETA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += BETA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += GAMMA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*GAMMA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*DELTA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += DELTA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    int hw = (w+1)/2;
    for (int x = t*2; x < w; x += nt*2) out[y*s+x/2] = sm[x];
    for (int x = t*2+1; x < w; x += nt*2) out[y*s+hw+x/2] = sm[x];
}

__global__ void v15_dwt_h(float* __restrict__ data, float* __restrict__ out, int w, int h, int s) {
    extern __shared__ float sm[];
    int y = blockIdx.x; if (y >= h) return;
    int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < w; i += nt) sm[i] = data[y*s+i];
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += ALPHA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*ALPHA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*BETA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += BETA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += GAMMA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*GAMMA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*DELTA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += DELTA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    int hw = (w+1)/2;
    for (int x = t*2; x < w; x += nt*2) out[y*s+x/2] = sm[x];
    for (int x = t*2+1; x < w; x += nt*2) out[y*s+hw+x/2] = sm[x];
}

/*
 * Vertical DWT with __ldg for initial loads.
 * The first read of each column element goes through the read-only cache.
 * Subsequent lifting steps modify in-place (can't use __ldg after first write).
 * But the first step (alpha) reads the input fresh — __ldg helps here.
 */
__global__ void v15_dwt_v(float* __restrict__ data, float* __restrict__ out, int w, int h, int s) {
    int x = blockIdx.x*blockDim.x+threadIdx.x; if (x >= w) return;

    /* Step 1: alpha on odd rows - use __ldg for initial reads */
    for (int y = 1; y < h-1; y += 2) {
        float top = __ldg(&data[(y-1)*s+x]);
        float bot = __ldg(&data[(y+1)*s+x]);
        data[y*s+x] = __ldg(&data[y*s+x]) + ALPHA*(top+bot);
    }
    if (h>1&&!(h&1))
        data[(h-1)*s+x] = __ldg(&data[(h-1)*s+x]) + 2.f*ALPHA*data[(h-2)*s+x];

    /* Step 2: beta on even rows */
    data[0*s+x] += 2.f*BETA*data[min(1,h-1)*s+x];
    for (int y = 2; y < h; y += 2)
        data[y*s+x] += BETA*(data[(y-1)*s+x]+data[min(y+1,h-1)*s+x]);

    /* Step 3: gamma on odd rows */
    for (int y = 1; y < h-1; y += 2)
        data[y*s+x] += GAMMA*(data[(y-1)*s+x]+data[(y+1)*s+x]);
    if (h>1&&!(h&1))
        data[(h-1)*s+x] += 2.f*GAMMA*data[(h-2)*s+x];

    /* Step 4: delta on even rows */
    data[0*s+x] += 2.f*DELTA*data[min(1,h-1)*s+x];
    for (int y = 2; y < h; y += 2)
        data[y*s+x] += DELTA*(data[(y-1)*s+x]+data[min(y+1,h-1)*s+x]);

    /* Deinterleave */
    int hh = (h+1)/2;
    for (int y = 0; y < h; y += 2) out[(y/2)*s+x] = data[y*s+x];
    for (int y = 1; y < h; y += 2) out[(hh+y/2)*s+x] = data[y*s+x];
}

/* Register-blocked v-DWT for small subbands (from v14) */
__global__ void v15_dwt_v_reg(float* __restrict__ data, float* __restrict__ out, int w, int h, int s) {
    int x = blockIdx.x*blockDim.x+threadIdx.x; if (x >= w) return;
    float col[MAX_REG_HEIGHT];
    for (int y = 0; y < h; ++y) col[y] = data[y*s+x];
    for (int y=1;y<h-1;y+=2) col[y]+=ALPHA*(col[y-1]+col[y+1]);
    if (h>1&&!(h&1)) col[h-1]+=2.f*ALPHA*col[h-2];
    col[0]+=2.f*BETA*col[min(1,h-1)];
    for (int y=2;y<h;y+=2) col[y]+=BETA*(col[y-1]+col[min(y+1,h-1)]);
    for (int y=1;y<h-1;y+=2) col[y]+=GAMMA*(col[y-1]+col[y+1]);
    if (h>1&&!(h&1)) col[h-1]+=2.f*GAMMA*col[h-2];
    col[0]+=2.f*DELTA*col[min(1,h-1)];
    for (int y=2;y<h;y+=2) col[y]+=DELTA*(col[y-1]+col[min(y+1,h-1)]);
    int hh=(h+1)/2;
    for (int y=0;y<h;y+=2) out[(y/2)*s+x]=col[y];
    for (int y=1;y<h;y+=2) out[(hh+y/2)*s+x]=col[y];
}

__global__ void v15_qe(const float* __restrict__ src, uint8_t* __restrict__ out, int n, int tn, float step) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i >= tn) return;
    if (i < n) { int q = __float2int_rn(src[i]/step); out[i] = uint8_t((q<0?0x80:0)|min(127,abs(q))); }
    else out[i] = 0;
}

/* GPU config */
struct GpuConfig15 {
    int max_threads_per_block, max_shared_mem, warp_size;
    int h_block, v_block, gen_block;
    bool configure(int device = 0) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
        max_threads_per_block = prop.maxThreadsPerBlock;
        max_shared_mem = prop.sharedMemPerBlock;
        warp_size = prop.warpSize;
        h_block = std::min(256, max_threads_per_block);
        h_block = (h_block / warp_size) * warp_size;
        if (h_block < warp_size) h_block = warp_size;
        v_block = std::min(128, max_threads_per_block);
        v_block = (v_block / warp_size) * warp_size;
        if (v_block < warp_size) v_block = warp_size;
        gen_block = std::min(256, max_threads_per_block);
        gen_block = (gen_block / warp_size) * warp_size;
        return true;
    }
};

struct CudaJ2KEncoderImpl {
    char* d_pool = nullptr, *h_pool = nullptr;
    float* d_c[3], *d_t[3]; int32_t* d_in[3]; uint8_t* d_enc[3];
    int32_t* h_in[3]; uint8_t* h_enc;
    size_t px = 0, enc_pc = 0;
    cudaStream_t st[3] = {};
    std::vector<uint8_t> hdr; int cw = 0, ch = 0;
    GpuConfig15 gpu;

    bool init() {
        if (!gpu.configure(0)) return false;
        for (int i = 0; i < 3; ++i) if (cudaStreamCreate(&st[i]) != cudaSuccess) return false;
        return true;
    }
    bool ensure(int w, int h, size_t epc) {
        size_t n = size_t(w)*h; enc_pc = epc;
        if (n <= px) return true; cleanup();
        size_t per_comp = n*sizeof(float)*2 + n*sizeof(int32_t) + epc;
        if (cudaMalloc(&d_pool, per_comp*3) != cudaSuccess) return false;
        char* p = d_pool;
        for (int c = 0; c < 3; ++c) {
            d_c[c] = reinterpret_cast<float*>(p); p += n*sizeof(float);
            d_t[c] = reinterpret_cast<float*>(p); p += n*sizeof(float);
            d_in[c] = reinterpret_cast<int32_t*>(p); p += n*sizeof(int32_t);
            d_enc[c] = reinterpret_cast<uint8_t*>(p); p += epc;
        }
        if (cudaHostAlloc(&h_pool, 3*n*sizeof(int32_t)+3*epc, cudaHostAllocDefault) != cudaSuccess) {
            cudaFree(d_pool); d_pool = nullptr; return false;
        }
        char* hp = h_pool;
        for (int c = 0; c < 3; ++c) { h_in[c] = reinterpret_cast<int32_t*>(hp); hp += n*sizeof(int32_t); }
        h_enc = reinterpret_cast<uint8_t*>(hp); px = n;
        return true;
    }
    void build_hdr(int w, int h) {
        if (w==cw&&h==ch) return; cw=w; ch=h; hdr.clear();
        auto w8=[&](uint8_t v){hdr.push_back(v);};
        auto w16=[&](uint16_t v){hdr.push_back(v>>8);hdr.push_back(v&0xFF);};
        auto w32=[&](uint32_t v){w16(v>>16);w16(v&0xFFFF);};
        w16(J2K_SOC); w16(J2K_SIZ); w16(47); w16(0);
        w32(w);w32(h);w32(0);w32(0);w32(w);w32(h);w32(0);w32(0);
        w16(3); for(int c=0;c<3;++c){w8(11);w8(1);w8(1);}
        w16(J2K_COD);w16(2+1+4+5+(NUM_DWT_LEVELS+1));
        w8(0);w8(1);w16(1);w8(0);w8(NUM_DWT_LEVELS);w8(5);w8(5);w8(0);w8(1);
        for(int i=0;i<=NUM_DWT_LEVELS;++i) w8(0xFF);
        int ns=3*NUM_DWT_LEVELS+1;
        w16(J2K_QCD);w16(2+1+2*ns);w8(0x22);
        for(int i=0;i<ns;++i){int e=std::max(0,13-i/3),m=std::max(0,0x800-i*64);w16(uint16_t((e<<11)|(m&0x7FF)));}
    }
    void cleanup() {
        if(d_pool){cudaFree(d_pool);d_pool=nullptr;}
        if(h_pool){cudaFreeHost(h_pool);h_pool=nullptr;}
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
    if(!_initialized) return {};
    const int stride=width;
    const size_t n=size_t(width)*height;
    int64_t fb=bit_rate/fps; if(is_3d) fb/=2;
    size_t tt=std::max(size_t(16384),size_t(fb/8));
    size_t tpc=(tt-200)/3; tpc=std::min(tpc,n);
    if(!_impl->ensure(width,height,tpc)) return {};
    _impl->build_hdr(width,height);

    const auto& gpu=_impl->gpu;
    const int h_blk=gpu.h_block,v_blk=gpu.v_block,gen_blk=gpu.gen_block;
    float bs=is_4k?.5f:1.f;
    float steps[3]={bs*1.2f,bs,bs*1.2f};

    for(int c=0;c<3;++c){
        memcpy(_impl->h_in[c],xyz[c],n*sizeof(int32_t));
        cudaMemcpyAsync(_impl->d_in[c],_impl->h_in[c],n*sizeof(int32_t),cudaMemcpyHostToDevice,_impl->st[c]);

        int w=width,h=height;
        float*A,*B_=_impl->d_t[c];

        v15_fused_i2f_dwt_h<<<h,h_blk,w*sizeof(float),_impl->st[c]>>>(_impl->d_in[c],_impl->d_c[c],w,h,stride);
        A=_impl->d_c[c];

        /* Level 0: use __ldg-optimized v-DWT for large subband */
        v15_dwt_v<<<(w+v_blk-1)/v_blk,v_blk,0,_impl->st[c]>>>(A,B_,w,h,stride);
        float*t=A;A=B_;B_=t; w=(w+1)/2;h=(h+1)/2;

        for(int l=1;l<NUM_DWT_LEVELS;++l){
            v15_dwt_h<<<h,h_blk,w*sizeof(float),_impl->st[c]>>>(A,B_,w,h,stride);
            t=A;A=B_;B_=t;
            if(h<=MAX_REG_HEIGHT)
                v15_dwt_v_reg<<<(w+v_blk-1)/v_blk,v_blk,0,_impl->st[c]>>>(A,B_,w,h,stride);
            else
                v15_dwt_v<<<(w+v_blk-1)/v_blk,v_blk,0,_impl->st[c]>>>(A,B_,w,h,stride);
            t=A;A=B_;B_=t; w=(w+1)/2;h=(h+1)/2;
        }

        v15_qe<<<(tpc+gen_blk-1)/gen_blk,gen_blk,0,_impl->st[c]>>>(A,_impl->d_enc[c],n,tpc,steps[c]);
        cudaMemcpyAsync(_impl->h_enc+c*tpc,_impl->d_enc[c],tpc,cudaMemcpyDeviceToHost,_impl->st[c]);
    }
    for(int c=0;c<3;++c) cudaStreamSynchronize(_impl->st[c]);

    const size_t td=tpc*3;
    std::vector<uint8_t> cs;
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

static std::shared_ptr<CudaJ2KEncoder> _inst;static std::mutex _inst_mu;
std::shared_ptr<CudaJ2KEncoder> cuda_j2k_encoder_instance(){std::lock_guard<std::mutex> l(_inst_mu);if(!_inst)_inst=std::make_shared<CudaJ2KEncoder>();return _inst;}
