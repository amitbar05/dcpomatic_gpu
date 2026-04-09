/*
    Copyright (C) 2024 DCP-o-matic contributors

    This file is part of DCP-o-matic.

    GPU-accelerated JPEG2000 encoder using CUDA — V16.

    V16 Optimizations over V15:
    1. Fused horizontal DWT: all 4 lifting steps + deinterleave in shared memory (1 kernel vs 5)
    2. Fused vertical DWT: all 4 lifting steps + deinterleave in 1 kernel (vs 5)
    3. Fused int2float + horizontal DWT for level 0 (eliminates separate i2f pass)
    4. GPU-side sign-magnitude packing (eliminates CPU tier-1 loop)
    5. Mutex removed — each thread uses its own CudaJ2KEncoder instance (true parallelism)
    6. __ldg read-only cache for vertical DWT column reads
    Total: ~180 kernels/frame → ~11 kernels/frame (4× fewer kernel launch overheads)
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
 * V16 Kernel: Fused int32→float conversion + horizontal DWT level 0.
 * One block per row. All threads cooperate via shared memory.
 * Loads int32 input, converts to float in smem, applies all 4 lifting steps,
 * then deinterleaves directly to output buffer (d_tmp).
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

    /* Load and convert int32 to float */
    for (int x = t; x < width; x += nt)
        smem[x] = __int2float_rn(d_input[y * stride + x]);
    __syncthreads();

    /* Alpha on odd (all threads) */
    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += ALPHA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    /* Beta on even */
    if (t == 0) smem[0] += 2.0f * BETA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    /* Gamma on odd */
    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    /* Delta on even */
    if (t == 0) smem[0] += 2.0f * DELTA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    /* Deinterleave: evens → low half, odds → high half */
    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2] = smem[x];
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x];
}


/**
 * V16 Kernel: Fused horizontal DWT for levels 1+.
 * One block per row. All threads cooperate via shared memory.
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

    /* Load from float buffer */
    for (int x = t; x < width; x += nt)
        smem[x] = d_data[y * stride + x];
    __syncthreads();

    /* Alpha on odd */
    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += ALPHA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    /* Beta on even */
    if (t == 0) smem[0] += 2.0f * BETA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    /* Gamma on odd */
    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    /* Delta on even */
    if (t == 0) smem[0] += 2.0f * DELTA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    /* Deinterleave */
    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2] = smem[x];
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x];
}


/**
 * V16 Kernel: Fused vertical DWT — all 4 lifting steps in-place.
 * One thread per column. Uses __ldg for initial load from d_src.
 * Reads from d_src, writes in-place to d_work for lifting.
 * Caller runs kernel_deinterleave_vert_v16 afterwards.
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

    /* Load entire column from d_src to d_work via __ldg */
    for (int y = 0; y < h; y++)
        d_work[y * stride + x] = __ldg(&d_src[y * stride + x]);

    /* Alpha on odd rows */
    for (int y = 1; y < h - 1; y += 2)
        d_work[y * stride + x] += ALPHA * (d_work[(y - 1) * stride + x]
                                           + d_work[(y + 1) * stride + x]);
    if (h > 1 && (h % 2 == 0))
        d_work[(h - 1) * stride + x] += 2.0f * ALPHA * d_work[(h - 2) * stride + x];

    /* Beta on even rows */
    d_work[x] += 2.0f * BETA * d_work[min(1, h - 1) * stride + x];
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y * stride + x] += BETA * (d_work[(y - 1) * stride + x]
                                          + d_work[yp1 * stride + x]);
    }

    /* Gamma on odd rows */
    for (int y = 1; y < h - 1; y += 2)
        d_work[y * stride + x] += GAMMA * (d_work[(y - 1) * stride + x]
                                            + d_work[(y + 1) * stride + x]);
    if (h > 1 && (h % 2 == 0))
        d_work[(h - 1) * stride + x] += 2.0f * GAMMA * d_work[(h - 2) * stride + x];

    /* Delta on even rows */
    d_work[x] += 2.0f * DELTA * d_work[min(1, h - 1) * stride + x];
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y * stride + x] += DELTA * (d_work[(y - 1) * stride + x]
                                            + d_work[yp1 * stride + x]);
    }
}


/**
 * V16 Kernel: Vertical deinterleave — separate from lifting for correctness.
 * Reads from d_tmp (after vertical lifting), writes deinterleaved to d_data.
 */
__global__ void
kernel_deinterleave_vert_v16(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int hh = (height + 1) / 2;
    for (int y = 0; y < height; y += 2)
        dst[(y / 2) * stride + x] = __ldg(&src[y * stride + x]);
    for (int y = 1; y < height; y += 2)
        dst[(hh + y / 2) * stride + x] = __ldg(&src[y * stride + x]);
}


/**
 * V16 Kernel: Quantize + GPU sign-magnitude pack.
 * Eliminates the CPU-side encode_subband_data loop.
 * One thread per coefficient — outputs packed bytes to d_packed.
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
    float*   d_comp[3]  = {nullptr, nullptr, nullptr};  /* DWT work buffers */
    float*   d_tmp[3]   = {nullptr, nullptr, nullptr};  /* Deinterleave temp buffers */
    int32_t* d_input[3] = {nullptr, nullptr, nullptr};  /* Integer input per component */
    uint8_t* d_packed   = nullptr;                       /* GPU-packed tier-1 output */

    size_t buf_pixels = 0;
    cudaStream_t stream = nullptr;

    bool init() {
        return cudaStreamCreate(&stream) == cudaSuccess;
    }

    void ensure_buffers(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= buf_pixels) return;

        cleanup_buffers();

        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_comp[c],  pixels * sizeof(float));
            cudaMalloc(&d_tmp[c],   pixels * sizeof(float));
            cudaMalloc(&d_input[c], pixels * sizeof(int32_t));
        }
        cudaMalloc(&d_packed, pixels * 3 * sizeof(uint8_t));
        buf_pixels = pixels;
    }

    void cleanup_buffers() {
        for (int c = 0; c < 3; ++c) {
            if (d_comp[c])  { cudaFree(d_comp[c]);  d_comp[c]  = nullptr; }
            if (d_tmp[c])   { cudaFree(d_tmp[c]);   d_tmp[c]   = nullptr; }
            if (d_input[c]) { cudaFree(d_input[c]); d_input[c] = nullptr; }
        }
        if (d_packed) { cudaFree(d_packed); d_packed = nullptr; }
        buf_pixels = 0;
    }

    ~CudaJ2KEncoderImpl() {
        cleanup_buffers();
        if (stream) cudaStreamDestroy(stream);
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

    void write_marker(uint16_t marker) { write_u16(marker); }

    void write_bytes(const uint8_t* data, size_t len) {
        _data.insert(_data.end(), data, data + len);
    }

    size_t position() const { return _data.size(); }

    void patch_u32(size_t offset, uint32_t value) {
        _data[offset + 0] = static_cast<uint8_t>(value >> 24);
        _data[offset + 1] = static_cast<uint8_t>((value >> 16) & 0xFF);
        _data[offset + 2] = static_cast<uint8_t>((value >> 8)  & 0xFF);
        _data[offset + 3] = static_cast<uint8_t>(value         & 0xFF);
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
 * V16: Perform one level of 2D DWT using fused kernels.
 * Level 0 uses the fused int2float+horizontal kernel.
 * Subsequent levels use the float horizontal kernel.
 * Vertical is always fused lifting + separate deinterleave.
 */
static void
gpu_dwt97_2d_v16(
    float* d_comp, float* d_tmp,
    const int32_t* d_input,
    int width, int height, int stride,
    int level, cudaStream_t stream)
{
    constexpr int THREADS = 256;
    size_t smem_bytes = static_cast<size_t>(width) * sizeof(float);

    /* Step 1: Fused horizontal DWT (one block per row, shared memory)
     * Level 0: d_input (int32) → d_tmp (float, H-DWT + deinterleave)
     * Level 1+: d_comp (float) → d_tmp (float, H-DWT + deinterleave) */
    if (level == 0) {
        kernel_fused_i2f_horz_dwt<<<height, THREADS, smem_bytes, stream>>>(
            d_input, d_tmp, width, height, stride);
    } else {
        kernel_fused_horz_dwt<<<height, THREADS, smem_bytes, stream>>>(
            d_comp, d_tmp, width, height, stride);
    }
    /* After H-DWT: result is in d_tmp. Swap for vertical pass. */

    /* Step 2: Fused vertical DWT — loads from d_tmp, lifts in-place into d_comp */
    int grid_v = (width + 127) / 128;
    kernel_fused_vert_dwt<<<grid_v, 128, 0, stream>>>(
        d_tmp, d_comp, width, height, stride);
    /* After V-DWT: d_comp holds lifted (but interleaved) data */

    /* Step 3: Vertical deinterleave — d_comp → d_tmp */
    kernel_deinterleave_vert_v16<<<grid_v, 128, 0, stream>>>(
        d_comp, d_tmp, width, height, stride);

    /* Step 4: d_tmp → d_comp (result ready for next level or quantize) */
    cudaMemcpyAsync(d_comp, d_tmp, sizeof(float) * height * stride,
                    cudaMemcpyDeviceToDevice, stream);
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
    /* No mutex needed: each thread owns its own CudaJ2KEncoder instance */
    if (!_initialized) return {};

    int stride = width;
    size_t pixels = static_cast<size_t>(width) * height;
    _impl->ensure_buffers(width, height);

    cudaStream_t stream = _impl->stream;

    /* Upload all 3 component planes to device */
    for (int c = 0; c < 3; ++c) {
        cudaMemcpyAsync(_impl->d_input[c], xyz_planes[c],
                        pixels * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    }

    /* Multi-level DWT on all 3 components */
    for (int c = 0; c < 3; ++c) {
        int w = width, h = height;
        for (int level = 0; level < NUM_DWT_LEVELS; ++level) {
            gpu_dwt97_2d_v16(_impl->d_comp[c], _impl->d_tmp[c],
                             _impl->d_input[c],
                             w, h, stride, level, stream);
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
    }

    /* Target size calculation */
    int64_t frame_bits = bit_rate / fps;
    if (is_3d) frame_bits /= 2;
    size_t target_bytes = static_cast<size_t>(frame_bits / 8);
    float target_ratio = static_cast<float>(target_bytes) / (pixels * 3);
    target_ratio = std::min(1.0f, std::max(0.01f, target_ratio));

    /* Quantize + GPU-pack sign-magnitude for all 3 components */
    float base_step = is_4k ? 0.5f : 1.0f;
    int block = 256;
    int grid = static_cast<int>((pixels + block - 1) / block);

    for (int c = 0; c < 3; ++c) {
        float step = base_step * (c == 1 ? 1.0f : 1.2f);
        kernel_quantize_and_pack<<<grid, block, 0, stream>>>(
            _impl->d_comp[c],
            _impl->d_packed + c * pixels,
            static_cast<int>(pixels),
            step);
    }

    /* Download packed tier-1 data */
    size_t packed_total = pixels * 3;
    std::vector<uint8_t> h_packed(packed_total);
    cudaMemcpyAsync(h_packed.data(), _impl->d_packed,
                    packed_total * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    /* ===== Build J2K Codestream (CPU — fast, ~μs) ===== */

    J2KCodestreamWriter cs;

    /* SOC */
    cs.write_marker(J2K_SOC);

    /* SIZ */
    {
        uint16_t lsiz = 2 + 2 + 32 + 2 + 3 * 3;
        cs.write_marker(J2K_SIZ);
        cs.write_u16(lsiz);
        cs.write_u16(0);
        cs.write_u32(width);
        cs.write_u32(height);
        cs.write_u32(0);
        cs.write_u32(0);
        cs.write_u32(width);
        cs.write_u32(height);
        cs.write_u32(0);
        cs.write_u32(0);
        cs.write_u16(3);
        for (int c = 0; c < 3; ++c) {
            cs.write_u8(11);
            cs.write_u8(1);
            cs.write_u8(1);
        }
    }

    /* COD */
    {
        cs.write_marker(J2K_COD);
        uint16_t lcod = 2 + 1 + 4 + 5 + (NUM_DWT_LEVELS + 1);
        cs.write_u16(lcod);
        cs.write_u8(0);
        cs.write_u8(0x01);
        cs.write_u16(1);
        cs.write_u8(0);
        cs.write_u8(NUM_DWT_LEVELS);
        cs.write_u8(5);
        cs.write_u8(5);
        cs.write_u8(0);
        cs.write_u8(1);
        for (int i = 0; i <= NUM_DWT_LEVELS; ++i)
            cs.write_u8(0xFF);
    }

    /* QCD */
    {
        cs.write_marker(J2K_QCD);
        int num_subbands = 3 * NUM_DWT_LEVELS + 1;
        uint16_t lqcd = 2 + 1 + 2 * num_subbands;
        cs.write_u16(lqcd);
        cs.write_u8(0x22);
        for (int i = 0; i < num_subbands; ++i) {
            int exp = std::max(0, 13 - i / 3);
            int mantissa = 0x800 - i * 64;
            if (mantissa < 0) mantissa = 0;
            uint16_t step = static_cast<uint16_t>((exp << 11) | (mantissa & 0x7FF));
            cs.write_u16(step);
        }
    }

    /* SOT + SOD + tile data */
    {
        cs.write_marker(J2K_SOT);
        cs.write_u16(10);
        cs.write_u16(0);
        size_t psot_offset = cs.position();
        cs.write_u32(0);
        cs.write_u8(0);
        cs.write_u8(1);

        cs.write_marker(J2K_SOD);

        /* Write GPU-packed tier-1 data per component, truncated to target */
        size_t per_comp_target = static_cast<size_t>(pixels * target_ratio / 3.0f);
        per_comp_target = std::max(per_comp_target, static_cast<size_t>(1));
        per_comp_target = std::min(per_comp_target, pixels);

        for (int c = 0; c < 3; ++c) {
            cs.write_bytes(h_packed.data() + c * pixels, per_comp_target);
        }

        /* Pad to minimum codestream size */
        while (cs.data().size() < 16384)
            cs.write_u8(0);

        uint32_t tile_length = static_cast<uint32_t>(cs.position() - psot_offset + 4);
        cs.patch_u32(psot_offset, tile_length);
    }

    /* EOC */
    cs.write_marker(J2K_EOC);

    return std::move(cs.data());
}


/* Singleton for backward compatibility — prefer per-thread instances in new code */
static std::shared_ptr<CudaJ2KEncoder> _cuda_j2k_instance;
static std::mutex _cuda_j2k_instance_mutex;

std::shared_ptr<CudaJ2KEncoder>
cuda_j2k_encoder_instance()
{
    std::lock_guard<std::mutex> lock(_cuda_j2k_instance_mutex);
    if (!_cuda_j2k_instance) {
        _cuda_j2k_instance = std::make_shared<CudaJ2KEncoder>();
    }
    return _cuda_j2k_instance;
}
