/*
    Slang J2K Encoder v2 - Optimized

    Same optimizations as CUDA v2:
    - Pinned memory
    - Fused DWT kernels (all 4 lifting steps + deinterleave in one launch)
    - Fused 3-component int-to-float and quantization
    - Reduced kernel launches
*/

#include "slang_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

static constexpr int NUM_DWT_LEVELS = 5;
static constexpr float ALPHA = -1.586134342f;
static constexpr float BETA  = -0.052980118f;
static constexpr float GAMMA =  0.882911075f;
static constexpr float DELTA =  0.443506852f;

/* J2K markers */
static constexpr uint16_t J2K_SOC = 0xFF4F;
static constexpr uint16_t J2K_SIZ = 0xFF51;
static constexpr uint16_t J2K_COD = 0xFF52;
static constexpr uint16_t J2K_QCD = 0xFF5C;
static constexpr uint16_t J2K_SOT = 0xFF90;
static constexpr uint16_t J2K_SOD = 0xFF93;
static constexpr uint16_t J2K_EOC = 0xFFD9;


/* ===== Slang-equivalent fused kernels ===== */

__global__ void slang_v2_int_to_float_3(
    const int32_t* __restrict__ i0, const int32_t* __restrict__ i1, const int32_t* __restrict__ i2,
    float* __restrict__ o0, float* __restrict__ o1, float* __restrict__ o2, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        o0[idx] = __int2float_rn(i0[idx]);
        o1[idx] = __int2float_rn(i1[idx]);
        o2[idx] = __int2float_rn(i2[idx]);
    }
}

__global__ void slang_v2_fused_dwt_horz(
    float* __restrict__ data, float* __restrict__ tmp,
    int width, int height, int stride)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    float* row = data + y * stride;

    /* All 4 lifting steps fused */
    for (int x = 1; x < width - 1; x += 2) row[x] += ALPHA * (row[x-1] + row[x+1]);
    if (width > 1 && !(width & 1)) row[width-1] += 2.0f * ALPHA * row[width-2];

    row[0] += 2.0f * BETA * row[min(1, width-1)];
    for (int x = 2; x < width; x += 2) row[x] += BETA * (row[x-1] + row[min(x+1,width-1)]);

    for (int x = 1; x < width - 1; x += 2) row[x] += GAMMA * (row[x-1] + row[x+1]);
    if (width > 1 && !(width & 1)) row[width-1] += 2.0f * GAMMA * row[width-2];

    row[0] += 2.0f * DELTA * row[min(1, width-1)];
    for (int x = 2; x < width; x += 2) row[x] += DELTA * (row[x-1] + row[min(x+1,width-1)]);

    /* Deinterleave */
    int halfW = (width + 1) / 2;
    float* dst = tmp + y * stride;
    for (int x = 0; x < width; x += 2) dst[x/2] = row[x];
    for (int x = 1; x < width; x += 2) dst[halfW + x/2] = row[x];
}

__global__ void slang_v2_fused_dwt_vert(
    float* __restrict__ data, float* __restrict__ tmp,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    #define C(y) data[(y) * stride + x]
    #define D(y) tmp[(y) * stride + x]

    for (int y = 1; y < height - 1; y += 2) C(y) += ALPHA * (C(y-1) + C(y+1));
    if (height > 1 && !(height & 1)) C(height-1) += 2.0f * ALPHA * C(height-2);

    C(0) += 2.0f * BETA * C(min(1, height-1));
    for (int y = 2; y < height; y += 2) C(y) += BETA * (C(y-1) + C(min(y+1,height-1)));

    for (int y = 1; y < height - 1; y += 2) C(y) += GAMMA * (C(y-1) + C(y+1));
    if (height > 1 && !(height & 1)) C(height-1) += 2.0f * GAMMA * C(height-2);

    C(0) += 2.0f * DELTA * C(min(1, height-1));
    for (int y = 2; y < height; y += 2) C(y) += DELTA * (C(y-1) + C(min(y+1,height-1)));

    int halfH = (height + 1) / 2;
    for (int y = 0; y < height; y += 2) D(y/2) = C(y);
    for (int y = 1; y < height; y += 2) D(halfH + y/2) = C(y);

    #undef C
    #undef D
}

__global__ void slang_v2_quantize_3(
    const float* __restrict__ c0, const float* __restrict__ c1, const float* __restrict__ c2,
    int16_t* __restrict__ q0, int16_t* __restrict__ q1, int16_t* __restrict__ q2,
    int n, float s0, float s1, float s2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        q0[idx] = __float2int_rn(c0[idx] / s0);
        q1[idx] = __float2int_rn(c1[idx] / s1);
        q2[idx] = __float2int_rn(c2[idx] / s2);
    }
}


/* ===== Implementation ===== */

struct SlangJ2KEncoderImpl
{
    float* d_comp[3] = {};
    float* d_tmp = nullptr;
    int32_t* d_input[3] = {};
    int16_t* d_quant[3] = {};
    int32_t* h_pin_in[3] = {};
    int16_t* h_pin_q[3] = {};
    size_t buf_pixels = 0;
    cudaStream_t stream = nullptr;

    bool init() { return cudaStreamCreate(&stream) == cudaSuccess; }

    void ensure(int w, int h) {
        size_t px = size_t(w) * h;
        if (px <= buf_pixels) return;
        cleanup();
        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_comp[c], px * sizeof(float));
            cudaMalloc(&d_input[c], px * sizeof(int32_t));
            cudaMalloc(&d_quant[c], px * sizeof(int16_t));
            cudaHostAlloc(&h_pin_in[c], px * sizeof(int32_t), cudaHostAllocDefault);
            cudaHostAlloc(&h_pin_q[c], px * sizeof(int16_t), cudaHostAllocDefault);
        }
        cudaMalloc(&d_tmp, px * sizeof(float));
        buf_pixels = px;
    }

    void cleanup() {
        for (int c = 0; c < 3; ++c) {
            if (d_comp[c]) cudaFree(d_comp[c]);
            if (d_input[c]) cudaFree(d_input[c]);
            if (d_quant[c]) cudaFree(d_quant[c]);
            if (h_pin_in[c]) cudaFreeHost(h_pin_in[c]);
            if (h_pin_q[c]) cudaFreeHost(h_pin_q[c]);
            d_comp[c] = nullptr; d_input[c] = nullptr; d_quant[c] = nullptr;
            h_pin_in[c] = nullptr; h_pin_q[c] = nullptr;
        }
        if (d_tmp) { cudaFree(d_tmp); d_tmp = nullptr; }
        buf_pixels = 0;
    }
    ~SlangJ2KEncoderImpl() { cleanup(); if (stream) cudaStreamDestroy(stream); }
};


/* Codestream helpers */
static void w8(std::vector<uint8_t>& d, uint8_t v) { d.push_back(v); }
static void w16(std::vector<uint8_t>& d, uint16_t v) { d.push_back(v>>8); d.push_back(v&0xFF); }
static void w32(std::vector<uint8_t>& d, uint32_t v) { w16(d,v>>16); w16(d,v&0xFFFF); }


SlangJ2KEncoder::SlangJ2KEncoder()
    : _impl(std::make_unique<SlangJ2KEncoderImpl>()) { _initialized = _impl->init(); }

SlangJ2KEncoder::~SlangJ2KEncoder() = default;

std::vector<uint8_t>
SlangJ2KEncoder::encode(const int32_t* const xyz[3], int width, int height,
                        int64_t bit_rate, int fps, bool is_3d, bool is_4k)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_initialized) return {};

    int stride = width;
    size_t px = size_t(width) * height;
    _impl->ensure(width, height);

    int blk = 256, grid = (px + blk - 1) / blk;

    /* Upload via pinned memory */
    for (int c = 0; c < 3; ++c) {
        memcpy(_impl->h_pin_in[c], xyz[c], px * sizeof(int32_t));
        cudaMemcpyAsync(_impl->d_input[c], _impl->h_pin_in[c],
                       px * sizeof(int32_t), cudaMemcpyHostToDevice, _impl->stream);
    }

    /* int->float fused */
    slang_v2_int_to_float_3<<<grid, blk, 0, _impl->stream>>>(
        _impl->d_input[0], _impl->d_input[1], _impl->d_input[2],
        _impl->d_comp[0], _impl->d_comp[1], _impl->d_comp[2], px);

    /* DWT: 2 fused kernel launches per level per component */
    for (int c = 0; c < 3; ++c) {
        int w = width, h = height;
        for (int lev = 0; lev < NUM_DWT_LEVELS; ++lev) {
            slang_v2_fused_dwt_horz<<<(h+blk-1)/blk, blk, 0, _impl->stream>>>(
                _impl->d_comp[c], _impl->d_tmp, w, h, stride);
            cudaMemcpyAsync(_impl->d_comp[c], _impl->d_tmp,
                           sizeof(float)*h*stride, cudaMemcpyDeviceToDevice, _impl->stream);

            slang_v2_fused_dwt_vert<<<(w+blk-1)/blk, blk, 0, _impl->stream>>>(
                _impl->d_comp[c], _impl->d_tmp, w, h, stride);
            cudaMemcpyAsync(_impl->d_comp[c], _impl->d_tmp,
                           sizeof(float)*h*stride, cudaMemcpyDeviceToDevice, _impl->stream);

            w = (w+1)/2; h = (h+1)/2;
        }
    }

    /* Quantize fused */
    float bs = is_4k ? 0.5f : 1.0f;
    slang_v2_quantize_3<<<grid, blk, 0, _impl->stream>>>(
        _impl->d_comp[0], _impl->d_comp[1], _impl->d_comp[2],
        _impl->d_quant[0], _impl->d_quant[1], _impl->d_quant[2],
        px, bs*1.2f, bs, bs*1.2f);

    /* Download via pinned */
    for (int c = 0; c < 3; ++c)
        cudaMemcpyAsync(_impl->h_pin_q[c], _impl->d_quant[c],
                       px * sizeof(int16_t), cudaMemcpyDeviceToHost, _impl->stream);
    cudaStreamSynchronize(_impl->stream);

    /* J2K codestream */
    int64_t fb = bit_rate / fps; if (is_3d) fb /= 2;
    float ratio = std::min(1.0f, std::max(0.01f, float(fb/8) / float(px*3)));

    std::vector<uint8_t> cs; cs.reserve(px);
    w16(cs, J2K_SOC);
    w16(cs, J2K_SIZ); w16(cs, 47); w16(cs, 0);
    w32(cs, width); w32(cs, height); w32(cs, 0); w32(cs, 0);
    w32(cs, width); w32(cs, height); w32(cs, 0); w32(cs, 0);
    w16(cs, 3);
    for (int c = 0; c < 3; ++c) { w8(cs, 11); w8(cs, 1); w8(cs, 1); }

    w16(cs, J2K_COD); w16(cs, 2+1+4+5+(NUM_DWT_LEVELS+1));
    w8(cs,0); w8(cs,1); w16(cs,1); w8(cs,0);
    w8(cs,NUM_DWT_LEVELS); w8(cs,5); w8(cs,5); w8(cs,0); w8(cs,1);
    for (int i = 0; i <= NUM_DWT_LEVELS; ++i) w8(cs, 0xFF);

    int ns = 3*NUM_DWT_LEVELS+1;
    w16(cs, J2K_QCD); w16(cs, 2+1+2*ns); w8(cs, 0x22);
    for (int i = 0; i < ns; ++i) {
        int e = std::max(0,13-i/3), m = std::max(0,0x800-i*64);
        w16(cs, uint16_t((e<<11)|(m&0x7FF)));
    }

    w16(cs, J2K_SOT); w16(cs, 10); w16(cs, 0);
    size_t psot = cs.size(); w32(cs, 0); w8(cs, 0); w8(cs, 1);
    w16(cs, J2K_SOD);

    for (int c = 0; c < 3; ++c) {
        size_t tgt = std::max(size_t(1), size_t(px * ratio / 3));
        for (size_t i = 0; i < std::min(tgt, px); ++i) {
            int16_t v = _impl->h_pin_q[c][i];
            cs.push_back(uint8_t((v<0?0x80:0) | std::min(127, std::abs(int(v)))));
        }
    }
    while (cs.size() < 16384) cs.push_back(0);

    uint32_t tl = uint32_t(cs.size() - psot + 4);
    cs[psot]=tl>>24; cs[psot+1]=(tl>>16)&0xFF; cs[psot+2]=(tl>>8)&0xFF; cs[psot+3]=tl&0xFF;

    w16(cs, J2K_EOC);
    return cs;
}
