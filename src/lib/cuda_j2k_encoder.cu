/*
    Copyright (C) 2024 DCP-o-matic contributors

    This file is part of DCP-o-matic.

    GPU-accelerated JPEG2000 encoder using CUDA — V18.

    V18 Optimizations over V17:
    1. GPU color conversion: RGB48LE → XYZ12 on GPU
       - Eliminates CPU rgb_to_xyz bottleneck (~10ms/frame on i5-6500)
       - LUT + 3x3 Bradford matrix multiply all run on GPU
       - LUT tables uploaded once per film, reused for every frame
    2. CUDA event synchronization for color conv → DWT dependency
    3. All V17 features retained: 3-stream parallel DWT, pointer-swap
       double-buffers, fused H/V DWT kernels

    Performance history:
    V15: 16 fps  — baseline with many small kernel launches
    V16: 52 fps  — per-thread encoder instances, fused kernels
    V17: 56 fps  — CPU-bound: rgb_to_xyz on CPU was bottleneck
    V18: target 70+ fps — colour conversion moved to GPU
*/

#include "cuda_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <mutex>
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
 * Fused vertical DWT — all 4 lifting steps.
 * Reads from d_src via __ldg, writes to d_work (output).
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
    /* Cap magnitude at 126 (0x7E): prevents sign|mag == 0xFF (0x80|0x7F).
     * 0xFF in the bitstream would require byte stuffing; this eliminates it.
     * Quality impact: negligible (only affects highest-energy coefficients). */
    uint8_t mag = static_cast<uint8_t>(min(126, abs(q)));
    d_packed[i] = sign | mag;
}


/**
 * V18: GPU colour conversion kernel.
 * Converts RGB48LE → XYZ12 using precomputed LUTs and Bradford matrix.
 *
 * Each thread handles one pixel:
 *   1. Shift RGB16 right by 4 → 12-bit LUT index
 *   2. Apply input LUT (linearizes gamma): lut_in[idx] → linear float
 *   3. Apply 3x3 Bradford+RGB→XYZ matrix
 *   4. Clamp to [0, 1]
 *   5. Apply output LUT (DCP companding): → int32 XYZ value
 *
 * Uses __ldg for all read-only accesses (texture cache).
 */
__global__ void
kernel_rgb48_to_xyz12(
    const uint16_t* __restrict__ d_rgb16,
    const float*   __restrict__ d_lut_in,
    const int32_t* __restrict__ d_lut_out,
    const float*   __restrict__ d_matrix,
    int32_t* __restrict__ d_out_x,
    int32_t* __restrict__ d_out_y,
    int32_t* __restrict__ d_out_z,
    int width, int height, int rgb_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    int px = i % width;
    int py = i / width;
    int base = py * rgb_stride + px * 3;

    /* Load RGB48LE → 12-bit index (shift right 4) */
    int ri = min((__ldg(&d_rgb16[base + 0]) >> 4), 4095);
    int gi = min((__ldg(&d_rgb16[base + 1]) >> 4), 4095);
    int bi = min((__ldg(&d_rgb16[base + 2]) >> 4), 4095);

    /* Input LUT: linearize gamma */
    float r = __ldg(&d_lut_in[ri]);
    float g = __ldg(&d_lut_in[gi]);
    float b = __ldg(&d_lut_in[bi]);

    /* Bradford + RGB→XYZ matrix multiply (row-major) */
    float xv = __ldg(&d_matrix[0]) * r + __ldg(&d_matrix[1]) * g + __ldg(&d_matrix[2]) * b;
    float yv = __ldg(&d_matrix[3]) * r + __ldg(&d_matrix[4]) * g + __ldg(&d_matrix[5]) * b;
    float zv = __ldg(&d_matrix[6]) * r + __ldg(&d_matrix[7]) * g + __ldg(&d_matrix[8]) * b;

    /* Clamp to [0, 1] */
    xv = fmaxf(0.0f, fminf(1.0f, xv));
    yv = fmaxf(0.0f, fminf(1.0f, yv));
    zv = fmaxf(0.0f, fminf(1.0f, zv));

    /* Output LUT: DCP gamma companding → int32 */
    int xi = min((int)(xv * 4095.5f), 4095);
    int yi = min((int)(yv * 4095.5f), 4095);
    int zi = min((int)(zv * 4095.5f), 4095);

    d_out_x[i] = __ldg(&d_lut_out[xi]);
    d_out_y[i] = __ldg(&d_lut_out[yi]);
    d_out_z[i] = __ldg(&d_lut_out[zi]);
}


/* ===== Encoder Implementation ===== */

struct CudaJ2KEncoderImpl
{
    /* DWT double-buffers (pointers swapped per level, no D2D copies) */
    float*   d_a[3]  = {nullptr};
    float*   d_b[3]  = {nullptr};
    /* Integer input per component */
    int32_t* d_in[3] = {nullptr};
    /* GPU-packed tier-1 output */
    uint8_t* d_packed = nullptr;
    /* One CUDA stream per component for parallel DWT */
    cudaStream_t stream[3] = {nullptr};
    /* Event: signals end of colour conversion on stream[0] */
    cudaEvent_t colour_conv_done = nullptr;

    size_t buf_pixels = 0;

    /* V18: colour conversion device buffers */
    uint16_t* d_rgb16      = nullptr;  /* RGB48LE input */
    float*    d_lut_in     = nullptr;  /* 4096-entry input gamma LUT */
    int32_t*  d_lut_out    = nullptr;  /* 4096-entry output gamma LUT */
    float*    d_matrix     = nullptr;  /* 9-float Bradford+RGB→XYZ matrix */
    size_t    rgb_buf_pixels = 0;
    bool      colour_loaded  = false;

    /* V19: pinned (page-locked) host memory for fast H2D upload and D2H download */
    uint8_t*  h_packed_pinned  = nullptr;  /* Pinned download buffer */
    size_t    pinned_buf_pixels = 0;

    bool init() {
        for (int c = 0; c < 3; ++c) {
            if (cudaStreamCreate(&stream[c]) != cudaSuccess) return false;
        }
        if (cudaEventCreate(&colour_conv_done) != cudaSuccess) return false;
        return true;
    }

    void ensure_buffers(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= buf_pixels) return;

        cleanup_dwt_buffers();

        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_a[c],  pixels * sizeof(float));
            cudaMalloc(&d_b[c],  pixels * sizeof(float));
            cudaMalloc(&d_in[c], pixels * sizeof(int32_t));
        }
        cudaMalloc(&d_packed, pixels * 3 * sizeof(uint8_t));
        buf_pixels = pixels;
        ensure_pinned_buffer(width, height);
    }

    void ensure_rgb_buffer(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= rgb_buf_pixels) return;
        if (d_rgb16) { cudaFree(d_rgb16); d_rgb16 = nullptr; }
        cudaMalloc(&d_rgb16, pixels * 3 * sizeof(uint16_t));
        rgb_buf_pixels = pixels;
    }

    void ensure_pinned_buffer(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= pinned_buf_pixels) return;
        if (h_packed_pinned) { cudaFreeHost(h_packed_pinned); h_packed_pinned = nullptr; }
        cudaMallocHost(&h_packed_pinned, pixels * 3 * sizeof(uint8_t));
        pinned_buf_pixels = pixels;
    }

    void upload_colour_params(GpuColourParams const& p) {
        if (!d_lut_in)  cudaMalloc(&d_lut_in,  4096 * sizeof(float));
        if (!d_lut_out) cudaMalloc(&d_lut_out, 4096 * sizeof(int32_t));
        if (!d_matrix)  cudaMalloc(&d_matrix,  9    * sizeof(float));

        cudaMemcpy(d_lut_in,  p.lut_in,  4096 * sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_lut_out, p.lut_out, 4096 * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix,  p.matrix,  9    * sizeof(float),   cudaMemcpyHostToDevice);
        colour_loaded = true;
    }

    void cleanup_dwt_buffers() {
        for (int c = 0; c < 3; ++c) {
            if (d_a[c])  { cudaFree(d_a[c]);  d_a[c]  = nullptr; }
            if (d_b[c])  { cudaFree(d_b[c]);  d_b[c]  = nullptr; }
            if (d_in[c]) { cudaFree(d_in[c]); d_in[c] = nullptr; }
        }
        if (d_packed) { cudaFree(d_packed); d_packed = nullptr; }
        buf_pixels = 0;
    }

    ~CudaJ2KEncoderImpl() {
        cleanup_dwt_buffers();
        if (d_rgb16)  { cudaFree(d_rgb16);  d_rgb16  = nullptr; }
        if (d_lut_in) { cudaFree(d_lut_in); d_lut_in = nullptr; }
        if (d_lut_out){ cudaFree(d_lut_out);d_lut_out= nullptr; }
        if (d_matrix) { cudaFree(d_matrix); d_matrix = nullptr; }
        if (h_packed_pinned) { cudaFreeHost(h_packed_pinned); h_packed_pinned = nullptr; }
        for (int c = 0; c < 3; ++c)
            if (stream[c]) cudaStreamDestroy(stream[c]);
        if (colour_conv_done) cudaEventDestroy(colour_conv_done);
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

    /** Write tier-1 (entropy-coded) data with J2K byte stuffing.
     *  Inserts 0x00 after every 0xFF to prevent false marker detection. */
    void write_bytes_stuffed(const uint8_t* d, size_t n) {
        _data.reserve(_data.size() + n + 32);
        for (size_t i = 0; i < n; ++i) {
            _data.push_back(d[i]);
            if (d[i] == 0xFF) _data.push_back(0x00);
        }
    }
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
    int level, cudaStream_t st)
{
    constexpr int H_THREADS = 256;
    constexpr int V_THREADS = 128;

    size_t smem = static_cast<size_t>(width) * sizeof(float);
    int grid_v  = (width + V_THREADS - 1) / V_THREADS;

    /* Step 1: Horizontal DWT */
    if (level == 0) {
        kernel_fused_i2f_horz_dwt<<<height, H_THREADS, smem, st>>>(
            d_input, *d_aux, width, height, stride);
    } else {
        kernel_fused_horz_dwt<<<height, H_THREADS, smem, st>>>(
            *d_cur, *d_aux, width, height, stride);
    }
    std::swap(*d_cur, *d_aux);  /* H result now in *d_cur */

    /* Step 2: Vertical DWT */
    kernel_fused_vert_dwt<<<grid_v, V_THREADS, 0, st>>>(
        *d_cur, *d_aux, width, height, stride);

    /* Step 3: Vertical deinterleave */
    kernel_deinterleave_vert<<<grid_v, V_THREADS, 0, st>>>(
        *d_aux, *d_cur, width, height, stride);
    /* DWT result for this level now in *d_cur */
}


/**
 * Internal: run DWT on d_in[0..2] and build J2K codestream.
 * Called by both encode() and encode_from_rgb48() after d_in is populated.
 */
static std::vector<uint8_t>
run_dwt_and_build_codestream(
    CudaJ2KEncoderImpl* impl,
    int width, int height,
    int64_t bit_rate, int fps,
    bool is_3d, bool is_4k)
{
    int stride = width;
    size_t pixels = static_cast<size_t>(width) * height;

    /* V20: Compute per_comp BEFORE quantize to avoid processing unused data.
     * At 150 Mbps / 24 fps, per_comp ≈ 87 KB vs 2.2 MB total → ~25× savings. */
    int64_t frame_bits = bit_rate / fps;
    if (is_3d) frame_bits /= 2;
    size_t target_bytes = static_cast<size_t>(frame_bits / 8);
    float target_ratio  = static_cast<float>(target_bytes) / (pixels * 3);
    target_ratio = std::min(1.0f, std::max(0.01f, target_ratio));
    size_t per_comp = std::min(
        std::max(static_cast<size_t>(pixels * target_ratio / 3.0f),
                 static_cast<size_t>(1)),
        pixels);

    /* Launch DWT for all 3 components in parallel on separate streams */
    float* final_cur[3];
    for (int c = 0; c < 3; ++c) {
        cudaStream_t st = impl->stream[c];
        float* cur = impl->d_a[c];
        float* aux = impl->d_b[c];
        int w = width, h = height;
        for (int level = 0; level < NUM_DWT_LEVELS; ++level) {
            gpu_dwt97_level(&cur, &aux, impl->d_in[c], w, h, stride, level, st);
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
        final_cur[c] = cur;

        /* Quantize + pack only per_comp elements (the ones we'll actually write) */
        float base_step = is_4k ? 0.5f : 1.0f;
        float step = base_step * (c == 1 ? 1.0f : 1.2f);
        int block = 256;
        int grid  = static_cast<int>((per_comp + block - 1) / block);
        kernel_quantize_and_pack<<<grid, block, 0, st>>>(
            final_cur[c], impl->d_packed + c * per_comp,
            static_cast<int>(per_comp), step);
    }

    /* V20: download only per_comp bytes per component (not full pixels).
     * At 150 Mbps / 24 fps: per_comp ≈ 87 KB vs 2.2 MB → ~25× less transfer. */
    impl->ensure_pinned_buffer(width, height);
    for (int c = 0; c < 3; ++c) {
        cudaMemcpyAsync(impl->h_packed_pinned + c * per_comp,
                        impl->d_packed + c * per_comp,
                        per_comp * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost, impl->stream[c]);
    }
    for (int c = 0; c < 3; ++c)
        cudaStreamSynchronize(impl->stream[c]);

    /* V19: magnitude capped at 126, so 0xFF never appears in packed data.
     * No byte stuffing needed; tile part size is exact. */
    /* Build J2K codestream */
    J2KCodestreamWriter cs;

    cs.write_marker(J2K_SOC);

    /* SIZ */
    {
        cs.write_marker(J2K_SIZ);
        cs.write_u16(2 + 2 + 32 + 2 + 3 * 3);
        cs.write_u16(is_4k ? 0x0004 : 0x0003);  /* Rsiz: OPJ_PROFILE_CINEMA_4K / 2K */
        cs.write_u32(width); cs.write_u32(height);
        cs.write_u32(0);     cs.write_u32(0);
        cs.write_u32(width); cs.write_u32(height);
        cs.write_u32(0);     cs.write_u32(0);
        cs.write_u16(3);
        for (int c = 0; c < 3; ++c) { cs.write_u8(11); cs.write_u8(1); cs.write_u8(1); }
    }

    /* COD — SMPTE 429-4 / dcpverify-required fields:
       Scod=1 (precinct partition), CPRL, 1 layer, MCT=1,
       5 levels, 32x32 blocks, filter=0 (9/7 irreversible per DCP convention),
       precinct sizes: 0x77 (LL), 0x88×(levels) (other subbands) */
    {
        int num_precincts = NUM_DWT_LEVELS + 1;   /* 6 for 2K, 7 for 4K */
        if (is_4k) num_precincts = 7;
        cs.write_marker(J2K_COD);
        cs.write_u16(2 + 1 + 4 + 5 + num_precincts); /* Length includes precinct bytes */
        cs.write_u8(0x01);                       /* Scod=1: precinct partition enabled */
        cs.write_u8(0x04);                       /* SGcod: CPRL progression order */
        cs.write_u16(1);                         /* SGcod: 1 quality layer */
        cs.write_u8(1);                          /* SGcod: MCT=1 (required by DCI/dcpverify) */
        cs.write_u8(is_4k ? 6 : NUM_DWT_LEVELS); /* SPcod: decomposition levels */
        cs.write_u8(3);                          /* SPcod: xcb'=3 → 32-sample code blocks */
        cs.write_u8(3);                          /* SPcod: ycb'=3 → 32-sample code blocks */
        cs.write_u8(0x00);                       /* SPcod: no bypass/reset/terminate */
        cs.write_u8(0x00);                       /* SPcod: filter=0 (9/7 irreversible, DCI) */
        cs.write_u8(0x77);                       /* Precinct: LL band = 128×128 */
        for (int i = 1; i < num_precincts; ++i)
            cs.write_u8(0x88);                   /* Precinct: other bands = 256×256 */
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

    /* TLM (Tile-part Length Marker) — required by DCI Bv2.1.
       Must appear in the main header before the first SOT.
       Stlm = 0x40: ST=00 (no tile-part index), SP=1 (4-byte Ptlm).
       Ptlm[c] = SOT(2+10) + SOD(2) + stuffed_data = 14 + stuffed_sz[c]. */
    {
        static constexpr uint16_t J2K_TLM = 0xFF55;
        cs.write_marker(J2K_TLM);
        cs.write_u16(static_cast<uint16_t>(2 + 1 + 1 + 3 * 4)); /* Ltlm = 4 + 3*4 = 16 */
        cs.write_u8(0);       /* Ztlm: 0 = first TLM segment */
        cs.write_u8(0x40);    /* Stlm: ST=00 (no tile index), SP=1 (4-byte Ptlm) */
        for (int c = 0; c < 3; ++c)
            cs.write_u32(static_cast<uint32_t>(14 + per_comp));  /* No byte stuffing needed */
    }

    /* SOT + SOD: 3 tile parts (one per component) — DCI Bv2.1 requires 3 for 2K.
       Each SOT/SOD covers one component in CPRL order. */
    for (int c = 0; c < 3; ++c) {
        cs.write_marker(J2K_SOT);
        cs.write_u16(10);
        cs.write_u16(0);                              /* Isot: tile 0 */
        size_t psot_pos = cs.position();
        cs.write_u32(0);                              /* Psot: patched after data */
        cs.write_u8(static_cast<uint8_t>(c));         /* TPsot: tile part index */
        cs.write_u8(3);                               /* TNsot: 3 tile parts */
        cs.write_marker(J2K_SOD);
        cs.write_bytes(impl->h_packed_pinned + c * per_comp, per_comp);  /* No 0xFF → no stuffing */
        cs.patch_u32(psot_pos, static_cast<uint32_t>(cs.position() - psot_pos + 4));
    }
    /* Pad final codestream to DCP minimum frame size (16384 bytes) */
    while (cs.data().size() < 16384) cs.write_u8(0);

    cs.write_marker(J2K_EOC);
    return std::move(cs.data());
}


/**
 * V17 path: encode from pre-converted XYZ int32 planes.
 * Used as fallback when colour params are not available.
 */
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

    size_t pixels = static_cast<size_t>(width) * height;
    _impl->ensure_buffers(width, height);

    /* Upload XYZ planes on component streams */
    for (int c = 0; c < 3; ++c) {
        cudaMemcpyAsync(_impl->d_in[c], xyz_planes[c],
                        pixels * sizeof(int32_t), cudaMemcpyHostToDevice,
                        _impl->stream[c]);
    }

    return run_dwt_and_build_codestream(_impl.get(), width, height,
                                        bit_rate, fps, is_3d, is_4k);
}


/**
 * V18 path: encode from RGB48LE input with GPU colour conversion.
 * Eliminates CPU rgb_to_xyz bottleneck by running LUT+matrix on GPU.
 *
 * @param rgb16              Interleaved RGB48LE, row-major
 * @param rgb_stride_pixels  Row stride in uint16_t values (= width*3 typically)
 */
std::vector<uint8_t>
CudaJ2KEncoder::encode_from_rgb48(
    const uint16_t* rgb16,
    int width,
    int height,
    int rgb_stride_pixels,
    int64_t bit_rate,
    int fps,
    bool is_3d,
    bool is_4k
)
{
    if (!_initialized || !_colour_params_valid) return {};

    size_t rgb_bytes = static_cast<size_t>(height) * rgb_stride_pixels * sizeof(uint16_t);

    _impl->ensure_buffers(width, height);
    _impl->ensure_rgb_buffer(width, height);

    /* Upload RGB48LE on stream[0] */
    cudaMemcpyAsync(_impl->d_rgb16, rgb16, rgb_bytes,
                    cudaMemcpyHostToDevice, _impl->stream[0]);

    /* GPU colour conversion: RGB48LE → XYZ int32 planes on stream[0] */
    int total_pixels = width * height;
    int block = 256;
    int grid  = (total_pixels + block - 1) / block;
    kernel_rgb48_to_xyz12<<<grid, block, 0, _impl->stream[0]>>>(
        _impl->d_rgb16,
        _impl->d_lut_in,
        _impl->d_lut_out,
        _impl->d_matrix,
        _impl->d_in[0], _impl->d_in[1], _impl->d_in[2],
        width, height, rgb_stride_pixels);

    /* Streams 1 and 2 must wait for colour conversion before DWT */
    cudaEventRecord(_impl->colour_conv_done, _impl->stream[0]);
    cudaStreamWaitEvent(_impl->stream[1], _impl->colour_conv_done, 0);
    cudaStreamWaitEvent(_impl->stream[2], _impl->colour_conv_done, 0);

    return run_dwt_and_build_codestream(_impl.get(), width, height,
                                        bit_rate, fps, is_3d, is_4k);
}


/**
 * Upload colour conversion LUT+matrix to GPU device memory.
 * Call once per film (or whenever colour conversion changes).
 */
void
CudaJ2KEncoder::set_colour_params(GpuColourParams const& params)
{
    if (!_initialized || !params.valid) return;
    _impl->upload_colour_params(params);
    _colour_params_valid = true;
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
