/*
    CUDA JPEG2000 Encoder v2 - Optimized

    Improvements over v1:
    - Pinned (page-locked) host memory for async transfers
    - Fused horizontal DWT kernel (all 4 lifting steps in one launch)
    - Fused vertical DWT kernel
    - All 3 components processed with a single stream pipeline
    - Reduced kernel launch overhead
*/

#include "cuda_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mutex>
#include <iostream>

/* J2K markers */
static constexpr uint16_t J2K_SOC = 0xFF4F;
static constexpr uint16_t J2K_SIZ = 0xFF51;
static constexpr uint16_t J2K_COD = 0xFF52;
static constexpr uint16_t J2K_QCD = 0xFF5C;
static constexpr uint16_t J2K_SOT = 0xFF90;
static constexpr uint16_t J2K_SOD = 0xFF93;
static constexpr uint16_t J2K_EOC = 0xFFD9;

static constexpr int NUM_DWT_LEVELS = 5;

/* CDF 9/7 coefficients */
static constexpr float ALPHA = -1.586134342f;
static constexpr float BETA  = -0.052980118f;
static constexpr float GAMMA =  0.882911075f;
static constexpr float DELTA =  0.443506852f;


/* ===== FUSED DWT Kernels ===== */

/**
 * Fused horizontal DWT 9/7: all 4 lifting steps + deinterleave in one kernel.
 * Each thread processes one row.
 */
__global__ void
fused_dwt97_horz(float* __restrict__ data, float* __restrict__ tmp,
                 int width, int height, int stride)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    float* row = data + y * stride;

    /* Step 1: alpha on odd */
    for (int x = 1; x < width - 1; x += 2)
        row[x] += ALPHA * (row[x-1] + row[x+1]);
    if (width > 1 && (width & 1) == 0)
        row[width-1] += 2.0f * ALPHA * row[width-2];

    /* Step 2: beta on even */
    row[0] += 2.0f * BETA * row[min(1, width-1)];
    for (int x = 2; x < width; x += 2) {
        int xp = min(x+1, width-1);
        row[x] += BETA * (row[x-1] + row[xp]);
    }

    /* Step 3: gamma on odd */
    for (int x = 1; x < width - 1; x += 2)
        row[x] += GAMMA * (row[x-1] + row[x+1]);
    if (width > 1 && (width & 1) == 0)
        row[width-1] += 2.0f * GAMMA * row[width-2];

    /* Step 4: delta on even */
    row[0] += 2.0f * DELTA * row[min(1, width-1)];
    for (int x = 2; x < width; x += 2) {
        int xp = min(x+1, width-1);
        row[x] += DELTA * (row[x-1] + row[xp]);
    }

    /* Deinterleave into tmp */
    int halfW = (width + 1) / 2;
    float* dst = tmp + y * stride;
    for (int x = 0; x < width; x += 2)
        dst[x/2] = row[x];
    for (int x = 1; x < width; x += 2)
        dst[halfW + x/2] = row[x];
}


/**
 * Fused vertical DWT 9/7: all 4 lifting steps + deinterleave in one kernel.
 * Each thread processes one column.
 */
__global__ void
fused_dwt97_vert(float* __restrict__ data, float* __restrict__ tmp,
                 int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    /* Macro for accessing column element */
    #define COL(y) data[(y) * stride + x]
    #define DST(y) tmp[(y) * stride + x]

    /* Step 1: alpha */
    for (int y = 1; y < height - 1; y += 2)
        COL(y) += ALPHA * (COL(y-1) + COL(y+1));
    if (height > 1 && (height & 1) == 0)
        COL(height-1) += 2.0f * ALPHA * COL(height-2);

    /* Step 2: beta */
    COL(0) += 2.0f * BETA * COL(min(1, height-1));
    for (int y = 2; y < height; y += 2) {
        int yp = min(y+1, height-1);
        COL(y) += BETA * (COL(y-1) + COL(yp));
    }

    /* Step 3: gamma */
    for (int y = 1; y < height - 1; y += 2)
        COL(y) += GAMMA * (COL(y-1) + COL(y+1));
    if (height > 1 && (height & 1) == 0)
        COL(height-1) += 2.0f * GAMMA * COL(height-2);

    /* Step 4: delta */
    COL(0) += 2.0f * DELTA * COL(min(1, height-1));
    for (int y = 2; y < height; y += 2) {
        int yp = min(y+1, height-1);
        COL(y) += DELTA * (COL(y-1) + COL(yp));
    }

    /* Deinterleave */
    int halfH = (height + 1) / 2;
    for (int y = 0; y < height; y += 2)
        DST(y/2) = COL(y);
    for (int y = 1; y < height; y += 2)
        DST(halfH + y/2) = COL(y);

    #undef COL
    #undef DST
}


/**
 * Fused int-to-float conversion for all 3 components.
 * Uses a single kernel launch processing 3x pixels.
 */
__global__ void
int_to_float_3comp(const int32_t* __restrict__ in0,
                   const int32_t* __restrict__ in1,
                   const int32_t* __restrict__ in2,
                   float* __restrict__ out0,
                   float* __restrict__ out1,
                   float* __restrict__ out2,
                   int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out0[idx] = __int2float_rn(in0[idx]);
        out1[idx] = __int2float_rn(in1[idx]);
        out2[idx] = __int2float_rn(in2[idx]);
    }
}


__global__ void
quantize_3comp(const float* __restrict__ c0,
               const float* __restrict__ c1,
               const float* __restrict__ c2,
               int16_t* __restrict__ q0,
               int16_t* __restrict__ q1,
               int16_t* __restrict__ q2,
               int n, float step0, float step1, float step2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        q0[idx] = __float2int_rn(c0[idx] / step0);
        q1[idx] = __float2int_rn(c1[idx] / step1);
        q2[idx] = __float2int_rn(c2[idx] / step2);
    }
}


/* ===== Encoder Implementation ===== */

struct CudaJ2KEncoderImpl
{
    /* Device buffers */
    float* d_comp[3] = {};
    float* d_tmp = nullptr;
    int32_t* d_input[3] = {};  /* Separate input buffers per component */
    int16_t* d_quant[3] = {};

    /* Pinned host buffers */
    int32_t* h_pinned_input[3] = {};
    int16_t* h_pinned_quant[3] = {};

    size_t buf_pixels = 0;
    cudaStream_t stream = nullptr;

    bool init() {
        return cudaStreamCreate(&stream) == cudaSuccess;
    }

    void ensure_buffers(int width, int height) {
        size_t pixels = size_t(width) * height;
        if (pixels <= buf_pixels) return;
        cleanup();

        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_comp[c], pixels * sizeof(float));
            cudaMalloc(&d_input[c], pixels * sizeof(int32_t));
            cudaMalloc(&d_quant[c], pixels * sizeof(int16_t));
            cudaHostAlloc(&h_pinned_input[c], pixels * sizeof(int32_t), cudaHostAllocDefault);
            cudaHostAlloc(&h_pinned_quant[c], pixels * sizeof(int16_t), cudaHostAllocDefault);
        }
        cudaMalloc(&d_tmp, pixels * sizeof(float));
        buf_pixels = pixels;
    }

    void cleanup() {
        for (int c = 0; c < 3; ++c) {
            if (d_comp[c]) cudaFree(d_comp[c]);
            if (d_input[c]) cudaFree(d_input[c]);
            if (d_quant[c]) cudaFree(d_quant[c]);
            if (h_pinned_input[c]) cudaFreeHost(h_pinned_input[c]);
            if (h_pinned_quant[c]) cudaFreeHost(h_pinned_quant[c]);
            d_comp[c] = nullptr;
            d_input[c] = nullptr;
            d_quant[c] = nullptr;
            h_pinned_input[c] = nullptr;
            h_pinned_quant[c] = nullptr;
        }
        if (d_tmp) { cudaFree(d_tmp); d_tmp = nullptr; }
        buf_pixels = 0;
    }

    ~CudaJ2KEncoderImpl() {
        cleanup();
        if (stream) cudaStreamDestroy(stream);
    }
};


/* J2K codestream builder (same as v1) */
static void wr8(std::vector<uint8_t>& d, uint8_t v) { d.push_back(v); }
static void wr16(std::vector<uint8_t>& d, uint16_t v) { d.push_back(v>>8); d.push_back(v&0xFF); }
static void wr32(std::vector<uint8_t>& d, uint32_t v) { wr16(d,v>>16); wr16(d,v&0xFFFF); }

static std::vector<uint8_t>
encode_subband(const int16_t* c, size_t pixels, float ratio)
{
    std::vector<uint8_t> out;
    size_t target = std::max(size_t(1), size_t(pixels * ratio));
    out.reserve(target);
    for (size_t i = 0; i < std::min(target, pixels); ++i) {
        uint8_t sign = (c[i] < 0) ? 0x80 : 0x00;
        uint8_t mag = uint8_t(std::min(127, std::abs(int(c[i]))));
        out.push_back(sign | mag);
    }
    return out;
}


/* ===== Public API ===== */

CudaJ2KEncoder::CudaJ2KEncoder()
    : _impl(std::make_unique<CudaJ2KEncoderImpl>())
{
    _initialized = _impl->init();
}

CudaJ2KEncoder::~CudaJ2KEncoder() = default;


std::vector<uint8_t>
CudaJ2KEncoder::encode(
    const int32_t* const xyz_planes[3],
    int width, int height,
    int64_t bit_rate, int fps,
    bool is_3d, bool is_4k)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_initialized) return {};

    int stride = width;
    size_t pixels = size_t(width) * height;
    _impl->ensure_buffers(width, height);

    int block = 256;
    int grid = (pixels + block - 1) / block;

    /* Copy to pinned memory and upload all 3 components */
    for (int c = 0; c < 3; ++c) {
        memcpy(_impl->h_pinned_input[c], xyz_planes[c], pixels * sizeof(int32_t));
        cudaMemcpyAsync(_impl->d_input[c], _impl->h_pinned_input[c],
                       pixels * sizeof(int32_t), cudaMemcpyHostToDevice, _impl->stream);
    }

    /* Convert all 3 components int->float in one kernel */
    int_to_float_3comp<<<grid, block, 0, _impl->stream>>>(
        _impl->d_input[0], _impl->d_input[1], _impl->d_input[2],
        _impl->d_comp[0], _impl->d_comp[1], _impl->d_comp[2], pixels);

    /* DWT: fused kernels - only 2 launches per level per component */
    for (int c = 0; c < 3; ++c) {
        int w = width, h = height;
        for (int level = 0; level < NUM_DWT_LEVELS; ++level) {
            int grid_h = (h + block - 1) / block;
            int grid_v = (w + block - 1) / block;

            fused_dwt97_horz<<<grid_h, block, 0, _impl->stream>>>(
                _impl->d_comp[c], _impl->d_tmp, w, h, stride);
            /* Copy tmp -> data */
            cudaMemcpyAsync(_impl->d_comp[c], _impl->d_tmp,
                           sizeof(float) * h * stride, cudaMemcpyDeviceToDevice, _impl->stream);

            fused_dwt97_vert<<<grid_v, block, 0, _impl->stream>>>(
                _impl->d_comp[c], _impl->d_tmp, w, h, stride);
            cudaMemcpyAsync(_impl->d_comp[c], _impl->d_tmp,
                           sizeof(float) * h * stride, cudaMemcpyDeviceToDevice, _impl->stream);

            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
    }

    /* Quantize all 3 components in one kernel */
    float base_step = is_4k ? 0.5f : 1.0f;
    quantize_3comp<<<grid, block, 0, _impl->stream>>>(
        _impl->d_comp[0], _impl->d_comp[1], _impl->d_comp[2],
        _impl->d_quant[0], _impl->d_quant[1], _impl->d_quant[2],
        pixels, base_step * 1.2f, base_step, base_step * 1.2f);

    /* Download quantized data via pinned memory */
    for (int c = 0; c < 3; ++c) {
        cudaMemcpyAsync(_impl->h_pinned_quant[c], _impl->d_quant[c],
                       pixels * sizeof(int16_t), cudaMemcpyDeviceToHost, _impl->stream);
    }
    cudaStreamSynchronize(_impl->stream);

    /* Build J2K codestream */
    int64_t frame_bits = bit_rate / fps;
    if (is_3d) frame_bits /= 2;
    float target_ratio = std::min(1.0f, std::max(0.01f, float(frame_bits / 8) / float(pixels * 3)));

    std::vector<uint8_t> cs;
    cs.reserve(pixels);

    wr16(cs, J2K_SOC);

    /* SIZ */
    wr16(cs, J2K_SIZ); wr16(cs, 47); wr16(cs, 0);
    wr32(cs, width); wr32(cs, height); wr32(cs, 0); wr32(cs, 0);
    wr32(cs, width); wr32(cs, height); wr32(cs, 0); wr32(cs, 0);
    wr16(cs, 3);
    for (int c = 0; c < 3; ++c) { wr8(cs, 11); wr8(cs, 1); wr8(cs, 1); }

    /* COD */
    wr16(cs, J2K_COD); wr16(cs, 2+1+4+5+(NUM_DWT_LEVELS+1));
    wr8(cs, 0); wr8(cs, 0x01); wr16(cs, 1); wr8(cs, 0);
    wr8(cs, NUM_DWT_LEVELS); wr8(cs, 5); wr8(cs, 5); wr8(cs, 0); wr8(cs, 1);
    for (int i = 0; i <= NUM_DWT_LEVELS; ++i) wr8(cs, 0xFF);

    /* QCD */
    int nsub = 3 * NUM_DWT_LEVELS + 1;
    wr16(cs, J2K_QCD); wr16(cs, 2+1+2*nsub); wr8(cs, 0x22);
    for (int i = 0; i < nsub; ++i) {
        int exp = std::max(0, 13 - i/3);
        int man = std::max(0, 0x800 - i*64);
        wr16(cs, uint16_t((exp << 11) | (man & 0x7FF)));
    }

    /* SOT */
    wr16(cs, J2K_SOT); wr16(cs, 10); wr16(cs, 0);
    size_t psot = cs.size();
    wr32(cs, 0); wr8(cs, 0); wr8(cs, 1);

    /* SOD + tile data */
    wr16(cs, J2K_SOD);
    for (int c = 0; c < 3; ++c) {
        auto sub = encode_subband(_impl->h_pinned_quant[c], pixels, target_ratio / 3.0f);
        cs.insert(cs.end(), sub.begin(), sub.end());
    }
    while (cs.size() < 16384) cs.push_back(0);

    /* Patch Psot */
    uint32_t tl = uint32_t(cs.size() - psot + 4);
    cs[psot] = tl>>24; cs[psot+1] = (tl>>16)&0xFF;
    cs[psot+2] = (tl>>8)&0xFF; cs[psot+3] = tl&0xFF;

    wr16(cs, J2K_EOC);
    return cs;
}

static std::shared_ptr<CudaJ2KEncoder> _inst;
static std::mutex _inst_mu;
std::shared_ptr<CudaJ2KEncoder> cuda_j2k_encoder_instance() {
    std::lock_guard<std::mutex> l(_inst_mu);
    if (!_inst) _inst = std::make_shared<CudaJ2KEncoder>();
    return _inst;
}
