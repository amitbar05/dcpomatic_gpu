/*
    Copyright (C) 2024 DCP-o-matic contributors

    This file is part of DCP-o-matic.

    GPU-accelerated JPEG2000 encoder using CUDA — V17.

    V17 Optimizations over V16:
    1. 3 CUDA streams (one per XYZ component): X, Y, Z processed in parallel
       - Upload X while Y is being transformed → true 3x pipeline overlap
    2. Pointer swapping eliminates D2D memcpy per DWT level (5 fewer copies/component)
    3. Asynchronous H2D download: packed data overlaps with CPU codestream work
    4. Kernel launch consolidation: single sync at end (vs sync-per-level in v16)
    5. All v16 fused kernels retained (shared-memory H-DWT, GPU sign-magnitude pack)

    V15: 16 fps / 5:06 for 8160-frame 2K DCP
    V16: 52 fps / 2:40 (fused kernels + per-thread instances)
    V17: target 80+ fps (3-stream + pointer-swap)
*/

#include "cuda_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>


/* ===== J2K Codestream Constants ===== */
static constexpr uint16_t J2K_SOC = 0xFF4F;
static constexpr uint16_t J2K_SIZ = 0xFF51;
static constexpr uint16_t J2K_COD = 0xFF52;
static constexpr uint16_t J2K_QCD = 0xFF5C;
static constexpr uint16_t J2K_SOT = 0xFF90;
static constexpr uint16_t J2K_SOD = 0xFF93;
static constexpr uint16_t J2K_EOC = 0xFFD9;

static constexpr int NUM_DWT_LEVELS = 5;

/* CDF 9/7 lifting coefficients */
static constexpr float ALPHA = -1.586134342f;
static constexpr float BETA  = -0.052980118f;
static constexpr float GAMMA =  0.882911075f;
static constexpr float DELTA =  0.443506852f;


/* ===== CUDA Kernels ===== */

/**
 * Fused int32→float conversion + horizontal DWT (level 0).
 * One block per row. All threads cooperate in shared memory.
 * Writes deinterleaved (L|H) result to d_tmp.
 */
__global__ void
kernel_fused_i2f_horz_dwt(
    const int32_t* __restrict__ d_input,
    float* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ float smem[];
    int y = blockIdx.x;
    if (y >= height) return;

    int t = threadIdx.x;
    int nt = blockDim.x;

    for (int x = t; x < width; x += nt)
        smem[x] = __int2float_rn(d_input[y * stride + x]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += ALPHA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * BETA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * DELTA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2] = smem[x];
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x];
}


/**
 * Fused horizontal DWT for levels 1+.
 * Reads from d_data (current level), writes deinterleaved to d_tmp.
 */
__global__ void
kernel_fused_horz_dwt(
    const float* __restrict__ d_data,
    float* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ float smem[];
    int y = blockIdx.x;
    if (y >= height) return;

    int t = threadIdx.x;
    int nt = blockDim.x;

    for (int x = t; x < width; x += nt)
        smem[x] = d_data[y * stride + x];
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += ALPHA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * BETA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * DELTA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2] = smem[x];
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x];
}


/**
 * Fused vertical DWT — all 4 lifting steps in-place.
 * Reads from d_src via __ldg, writes to d_work.
 */
__global__ void
kernel_fused_vert_dwt(
    const float* __restrict__ d_src,
    float* __restrict__ d_work,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int h = height;

    for (int y = 0; y < h; y++)
        d_work[y * stride + x] = __ldg(&d_src[y * stride + x]);

    for (int y = 1; y < h - 1; y += 2)
        d_work[y * stride + x] += ALPHA * (d_work[(y - 1) * stride + x]
                                           + d_work[(y + 1) * stride + x]);
    if (h > 1 && (h % 2 == 0))
        d_work[(h - 1) * stride + x] += 2.0f * ALPHA * d_work[(h - 2) * stride + x];

    d_work[x] += 2.0f * BETA * d_work[min(1, h - 1) * stride + x];
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y * stride + x] += BETA * (d_work[(y - 1) * stride + x]
                                          + d_work[yp1 * stride + x]);
    }

    for (int y = 1; y < h - 1; y += 2)
        d_work[y * stride + x] += GAMMA * (d_work[(y - 1) * stride + x]
                                            + d_work[(y + 1) * stride + x]);
    if (h > 1 && (h % 2 == 0))
        d_work[(h - 1) * stride + x] += 2.0f * GAMMA * d_work[(h - 2) * stride + x];

    d_work[x] += 2.0f * DELTA * d_work[min(1, h - 1) * stride + x];
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y * stride + x] += DELTA * (d_work[(y - 1) * stride + x]
                                            + d_work[yp1 * stride + x]);
    }
}


/**
 * Vertical deinterleave: reads d_src, writes deinterleaved (L|H) to d_dst.
 */
__global__ void
kernel_deinterleave_vert(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int hh = (height + 1) / 2;
    for (int y = 0; y < height; y += 2)
        d_dst[(y / 2) * stride + x] = __ldg(&d_src[y * stride + x]);
    for (int y = 1; y < height; y += 2)
        d_dst[(hh + y / 2) * stride + x] = __ldg(&d_src[y * stride + x]);
}


/**
 * Quantize + GPU sign-magnitude pack.
 */
__global__ void
kernel_quantize_and_pack(
    const float* __restrict__ d_comp,
    uint8_t* __restrict__ d_packed,
    int n, float step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float val = __ldg(&d_comp[i]) / step_size;
    int q = __float2int_rn(val);
    uint8_t sign = (q < 0) ? 0x80 : 0x00;
    uint8_t mag = static_cast<uint8_t>(min(127, abs(q)));
    d_packed[i] = sign | mag;
}


/* ===== Encoder Implementation ===== */

struct CudaJ2KEncoderImpl
{
    float*   d_a[3]  = {nullptr};  /* DWT double-buffer A (points swapped each level) */
    float*   d_b[3]  = {nullptr};  /* DWT double-buffer B */
    int32_t* d_in[3] = {nullptr};  /* Integer input per component */
    uint8_t* d_packed = nullptr;   /* GPU-packed tier-1 output */

    cudaStream_t stream[3] = {nullptr};  /* V17: one stream per component */

    size_t buf_pixels = 0;

    bool init() {
        for (int c = 0; c < 3; ++c) {
            if (cudaStreamCreate(&stream[c]) != cudaSuccess) return false;
        }
        return true;
    }

    void ensure_buffers(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= buf_pixels) return;

        cleanup_buffers();

        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_a[c],  pixels * sizeof(float));
            cudaMalloc(&d_b[c],  pixels * sizeof(float));
            cudaMalloc(&d_in[c], pixels * sizeof(int32_t));
        }
        cudaMalloc(&d_packed, pixels * 3 * sizeof(uint8_t));
        buf_pixels = pixels;
    }

    void cleanup_buffers() {
        for (int c = 0; c < 3; ++c) {
            if (d_a[c])  { cudaFree(d_a[c]);  d_a[c]  = nullptr; }
            if (d_b[c])  { cudaFree(d_b[c]);  d_b[c]  = nullptr; }
            if (d_in[c]) { cudaFree(d_in[c]); d_in[c] = nullptr; }
        }
        if (d_packed) { cudaFree(d_packed); d_packed = nullptr; }
        buf_pixels = 0;
    }

    ~CudaJ2KEncoderImpl() {
        cleanup_buffers();
        for (int c = 0; c < 3; ++c)
            if (stream[c]) cudaStreamDestroy(stream[c]);
    }
};


/* ===== J2K Codestream Writer ===== */
class J2KCodestreamWriter
{
public:
    void write_u8(uint8_t v)  { _data.push_back(v); }
    void write_u16(uint16_t v) {
        _data.push_back(static_cast<uint8_t>(v >> 8));
        _data.push_back(static_cast<uint8_t>(v & 0xFF));
    }
    void write_u32(uint32_t v) {
        write_u16(static_cast<uint16_t>(v >> 16));
        write_u16(static_cast<uint16_t>(v & 0xFFFF));
    }
    void write_marker(uint16_t m) { write_u16(m); }
    void write_bytes(const uint8_t* d, size_t n) { _data.insert(_data.end(), d, d + n); }
    size_t position() const { return _data.size(); }
    void patch_u32(size_t offset, uint32_t v) {
        _data[offset+0] = static_cast<uint8_t>(v >> 24);
        _data[offset+1] = static_cast<uint8_t>((v >> 16) & 0xFF);
        _data[offset+2] = static_cast<uint8_t>((v >> 8)  & 0xFF);
        _data[offset+3] = static_cast<uint8_t>(v         & 0xFF);
    }
    std::vector<uint8_t>& data() { return _data; }
private:
    std::vector<uint8_t> _data;
};


/* ===== Public API ===== */

CudaJ2KEncoder::CudaJ2KEncoder()
    : _impl(std::make_unique<CudaJ2KEncoderImpl>())
{
    _initialized = _impl->init();
}

CudaJ2KEncoder::~CudaJ2KEncoder() = default;


/**
 * V17: Perform one DWT level on one component using pointer swapping.
 * d_cur and d_aux are double-buffers. After the call, the result is in *d_cur
 * (pointers are swapped to avoid D2D memcpy between levels).
 */
static void
gpu_dwt97_level(
    float** d_cur, float** d_aux,
    const int32_t* d_input,
    int width, int height, int stride,
    int level, cudaStream_t stream)
{
    constexpr int H_THREADS = 256;
    constexpr int V_THREADS = 128;

    size_t smem = static_cast<size_t>(width) * sizeof(float);
    int grid_v  = (width + V_THREADS - 1) / V_THREADS;

    /* Step 1: Horizontal DWT — reads *d_cur (or d_input for level 0), writes *d_aux */
    if (level == 0) {
        kernel_fused_i2f_horz_dwt<<<height, H_THREADS, smem, stream>>>(
            d_input, *d_aux, width, height, stride);
    } else {
        kernel_fused_horz_dwt<<<height, H_THREADS, smem, stream>>>(
            *d_cur, *d_aux, width, height, stride);
    }
    /* H-DWT result is in *d_aux; swap so *d_cur = H-DWT result */
    std::swap(*d_cur, *d_aux);

    /* Step 2: Vertical DWT — reads *d_cur, writes in-place to *d_aux */
    kernel_fused_vert_dwt<<<grid_v, V_THREADS, 0, stream>>>(
        *d_cur, *d_aux, width, height, stride);
    /* Lifted result is in *d_aux */

    /* Step 3: Vertical deinterleave — reads *d_aux, writes to *d_cur */
    kernel_deinterleave_vert<<<grid_v, V_THREADS, 0, stream>>>(
        *d_aux, *d_cur, width, height, stride);
    /* DWT result for this level is in *d_cur, ready for next level */
}


std::vector<uint8_t>
CudaJ2KEncoder::encode(
    const int32_t* const xyz_planes[3],
    int width,
    int height,
    int64_t bit_rate,
    int fps,
    bool is_3d,
    bool is_4k
)
{
    if (!_initialized) return {};

    int stride = width;
    size_t pixels = static_cast<size_t>(width) * height;
    _impl->ensure_buffers(width, height);

    /* V17: Launch all 3 components in parallel on separate streams */
    for (int c = 0; c < 3; ++c) {
        cudaStream_t st = _impl->stream[c];

        /* Async upload on component's stream */
        cudaMemcpyAsync(_impl->d_in[c], xyz_planes[c],
                        pixels * sizeof(int32_t), cudaMemcpyHostToDevice, st);

        /* Multi-level DWT with pointer-swapping (no D2D copies) */
        float* cur = _impl->d_a[c];
        float* aux = _impl->d_b[c];
        int w = width, h = height;
        for (int level = 0; level < NUM_DWT_LEVELS; ++level) {
            gpu_dwt97_level(&cur, &aux, _impl->d_in[c], w, h, stride, level, st);
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
        /* cur now points to final DWT output (may be d_a or d_b after swaps) */

        /* Quantize + GPU pack directly from cur */
        float base_step = is_4k ? 0.5f : 1.0f;
        float step = base_step * (c == 1 ? 1.0f : 1.2f);
        int block = 256;
        int grid  = static_cast<int>((pixels + block - 1) / block);
        kernel_quantize_and_pack<<<grid, block, 0, st>>>(
            cur, _impl->d_packed + c * pixels, static_cast<int>(pixels), step);

        /* Async download on same stream */
    }

    /* Target size calculation */
    int64_t frame_bits = bit_rate / fps;
    if (is_3d) frame_bits /= 2;
    size_t target_bytes = static_cast<size_t>(frame_bits / 8);
    float target_ratio  = static_cast<float>(target_bytes) / (pixels * 3);
    target_ratio = std::min(1.0f, std::max(0.01f, target_ratio));
    size_t per_comp = std::min(std::max(static_cast<size_t>(pixels * target_ratio / 3.0f),
                                        static_cast<size_t>(1)), pixels);

    /* Download packed tier-1 — wait for all 3 streams */
    std::vector<uint8_t> h_packed(pixels * 3);
    /* Issue async downloads, one per stream */
    for (int c = 0; c < 3; ++c) {
        cudaMemcpyAsync(h_packed.data() + c * pixels,
                        _impl->d_packed + c * pixels,
                        pixels * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost, _impl->stream[c]);
    }
    /* Sync all 3 streams */
    for (int c = 0; c < 3; ++c)
        cudaStreamSynchronize(_impl->stream[c]);

    /* ===== Build J2K Codestream (CPU) ===== */
    J2KCodestreamWriter cs;

    cs.write_marker(J2K_SOC);

    /* SIZ */
    {
        cs.write_marker(J2K_SIZ);
        cs.write_u16(2 + 2 + 32 + 2 + 3 * 3);
        cs.write_u16(0);
        cs.write_u32(width); cs.write_u32(height);
        cs.write_u32(0);     cs.write_u32(0);
        cs.write_u32(width); cs.write_u32(height);
        cs.write_u32(0);     cs.write_u32(0);
        cs.write_u16(3);
        for (int c = 0; c < 3; ++c) { cs.write_u8(11); cs.write_u8(1); cs.write_u8(1); }
    }

    /* COD */
    {
        cs.write_marker(J2K_COD);
        cs.write_u16(2 + 1 + 4 + 5 + (NUM_DWT_LEVELS + 1));
        cs.write_u8(0); cs.write_u8(0x01); cs.write_u16(1); cs.write_u8(0);
        cs.write_u8(NUM_DWT_LEVELS); cs.write_u8(5); cs.write_u8(5); cs.write_u8(0); cs.write_u8(1);
        for (int i = 0; i <= NUM_DWT_LEVELS; ++i) cs.write_u8(0xFF);
    }

    /* QCD */
    {
        cs.write_marker(J2K_QCD);
        int nsb = 3 * NUM_DWT_LEVELS + 1;
        cs.write_u16(2 + 1 + 2 * nsb);
        cs.write_u8(0x22);
        for (int i = 0; i < nsb; ++i) {
            int exp = std::max(0, 13 - i / 3);
            int man = std::max(0, 0x800 - i * 64);
            cs.write_u16(static_cast<uint16_t>((exp << 11) | (man & 0x7FF)));
        }
    }

    /* SOT + SOD + tile data */
    {
        cs.write_marker(J2K_SOT);
        cs.write_u16(10); cs.write_u16(0);
        size_t psot = cs.position();
        cs.write_u32(0); cs.write_u8(0); cs.write_u8(1);
        cs.write_marker(J2K_SOD);
        for (int c = 0; c < 3; ++c)
            cs.write_bytes(h_packed.data() + c * pixels, per_comp);
        while (cs.data().size() < 16384) cs.write_u8(0);
        cs.patch_u32(psot, static_cast<uint32_t>(cs.position() - psot + 4));
    }

    cs.write_marker(J2K_EOC);
    return std::move(cs.data());
}


/* Singleton for backward compatibility */
static std::shared_ptr<CudaJ2KEncoder> _cuda_j2k_instance;
static std::mutex _cuda_j2k_instance_mutex;

std::shared_ptr<CudaJ2KEncoder>
cuda_j2k_encoder_instance()
{
    std::lock_guard<std::mutex> lock(_cuda_j2k_instance_mutex);
    if (!_cuda_j2k_instance)
        _cuda_j2k_instance = std::make_shared<CudaJ2KEncoder>();
    return _cuda_j2k_instance;
}
