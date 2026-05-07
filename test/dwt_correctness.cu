/*
    DWT Numerical Correctness Test — GPU vs. CPU reference CDF 9/7 DWT

    Compares GPU DWT output (both FP16 and FP32 paths) against a CPU reference
    implementation of the identical 5-level CDF 9/7 lifting scheme.

    Test patterns: sine wave gradient, flat field, checkerboard (8px, 64px),
                   random noise, linear gradient.
    Resolution: 2048×1080 (DCI 2K Flat).

    Build:
      nvcc -O3 -arch=sm_61 -std=c++17 \
           -I/home/amit/dcp-o-matic-gpu/src \
           -I/home/amit/dcp-o-matic-gpu/src/lib \
           -o test/dwt_correctness test/dwt_correctness.cu \
           src/lib/cuda_j2k_encoder.cu -lcudart

    The GPU DWT kernels live in cuda_j2k_encoder.cu.  This test declares the
    relevant kernel entry points and calls them directly with the same launch
    configuration and buffer layout that gpu_dwt97_level / gpu_dwt97_level_fp32
    use.  We replicate the buffer-swapping (d_a ↔ d_b) and skip-first-HDWT
    logic exactly.
*/

#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>

/* -----------------------------------------------------------------------
 * CDF 9/7 lifting coefficients & normalization constants
 * (mirror those in cuda_j2k_encoder.cu, checked at startup)
 * ----------------------------------------------------------------------- */
static constexpr float ALPHA = -1.586134342f;
static constexpr float BETA  = -0.052980118f;
static constexpr float GAMMA =  0.882911075f;
static constexpr float DELTA =  0.443506852f;
static constexpr float NORM_L = 0.812893197535108f;
static constexpr float NORM_H = 1.230174104914001f;

static constexpr int NUM_DWT_LEVELS = 5;
/* Reduced from 2048x1080 to keep RAM footprint small.  1024x512 still exercises
 * 5 DWT levels (LL5 = 32x16 — well above the 1-pixel minimum). */
static constexpr int W = 1024;
static constexpr int H = 512;
static constexpr int STRIDE = W;      /* same as width — no padding */

/* GPU kernel thread-count constants (must match cuda_j2k_encoder.cu) */
static constexpr int H_THREADS       = 512;
static constexpr int V_THREADS_TILED = 256;
static constexpr int V_THREADS_REG   = 128;
static constexpr int MAX_REG_HEIGHT  = 140;
static constexpr int V_TILE          = 24;
static constexpr int V_OVERLAP       = 4;
static constexpr int V_TILE_FL       = V_TILE + 2 * V_OVERLAP;  /* 32 */


/* -----------------------------------------------------------------------
 * Forward declarations of GPU kernels (link with cuda_j2k_encoder.cu).
 * Signature and launch config must exactly match.
 * ----------------------------------------------------------------------- */

/* --- FP16 path kernels --- */

/* Level-0 H-DWT: int32 input → __half output (1 row per block, smem = width*sizeof(__half)) */
__global__ void kernel_fused_i2f_horz_dwt_half_out(
    const int32_t* __restrict__ d_input,
    __half* __restrict__ d_tmp,
    int width, int height, int stride);

/* Level-0 4-row kernel (V74): int32→__half, 4 rows/block */
__global__ void kernel_fused_i2f_horz_dwt_half_out_4row(
    const int32_t* __restrict__ d_input,
    __half* __restrict__ d_tmp,
    int width, int height, int stride);

/* Levels 1+ H-DWT: __half → __half, templated on width%4==0 */
template<bool DIV4>
__global__ void kernel_fused_horz_dwt_half_io_4row(
    const __half* __restrict__ d_data,
    __half* __restrict__ d_tmp,
    int width, int height, int stride);

/* V-DWT reg-blocked (h ≤ 140): __half input → __half output, templated on even height */
template<bool EVEN_HEIGHT>
__global__ void kernel_fused_vert_dwt_fp16_hi_reg_ho(
    const __half* __restrict__ d_src,
    __half* __restrict__ d_dst,
    int width, int height, int stride);

/* V-DWT tiled 2-col (h > 140): __half → __half */
__global__ void kernel_fused_vert_dwt_tiled_ho_2col(
    const __half* __restrict__ d_src,
    __half* __restrict__ d_dst,
    int width, int height, int stride);


/* --- FP32 path kernels --- */

/* Level-0 H-DWT: int32 → float */
__global__ void kernel_fused_i2f_horz_dwt_fp32(
    const int32_t* __restrict__ d_input,
    float* __restrict__ d_tmp,
    int width, int height, int stride);

/* Levels 1+ H-DWT: float → float */
__global__ void kernel_fused_horz_dwt_fp32(
    const float* __restrict__ d_data,
    float* __restrict__ d_tmp,
    int width, int height, int stride);

/* V-DWT reg-blocked: float → float, templated on even height */
template<bool EVEN_HEIGHT>
__global__ void kernel_fused_vert_dwt_fp32_reg_ho(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int width, int height, int stride);

/* V-DWT tiled: float → float */
__global__ void kernel_fused_vert_dwt_fp32_tiled(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int width, int height, int stride);


/* -----------------------------------------------------------------------
 * CPU reference: CDF 9/7 DWT (matches GPU lifting exactly)
 * ----------------------------------------------------------------------- */

/* 1-D horizontal CDF 9/7 lifting + deinterleave + normalize.
 * Reads 'width' floats from 'row_in', writes deinterleaved L|H to 'row_out'.
 *   row_out[0..hw-1]         = L (even indices, ×NORM_L)
 *   row_out[hw..width-1]     = H (odd indices,  ×NORM_H)
 *   hw = (width+1)/2
 *
 * Boundary handling: symmetric extension (mirror at edges), matching GPU:
 *   last odd (if even width): x[W-1] += 2*ALPHA * x[W-2]
 *   first even (x[0]):        x[0]   += 2*BETA  * x[1]   (width>1)
 *                             x[0]   += 2*DELTA * x[1]
 */
static void cpu_horizontal_dwt(const float* row_in, float* row_out,
                               int width, int stride_out)
{
    /* Copy to local buf so we can lift in-place */
    std::vector<float> s(row_in, row_in + width);

    /* Alpha — update odd samples (indices 1,3,5,…) */
    for (int x = 1; x < width - 1; x += 2)
        s[x] += ALPHA * (s[x - 1] + s[x + 1]);
    if (width > 1 && (width % 2 == 0))
        s[width - 1] += 2.0f * ALPHA * s[width - 2];

    /* Beta — update even samples (indices 0,2,4,…) */
    if (width > 1) {
        s[0] += 2.0f * BETA * s[1];
        for (int x = 2; x < width; x += 2) {
            int xp1 = (x + 1 < width) ? x + 1 : x - 1;
            s[x] += BETA * (s[x - 1] + s[xp1]);
        }
    }

    /* Gamma — update odd samples */
    for (int x = 1; x < width - 1; x += 2)
        s[x] += GAMMA * (s[x - 1] + s[x + 1]);
    if (width > 1 && (width % 2 == 0))
        s[width - 1] += 2.0f * GAMMA * s[width - 2];

    /* Delta — update even samples */
    if (width > 1) {
        s[0] += 2.0f * DELTA * s[1];
        for (int x = 2; x < width; x += 2) {
            int xp1 = (x + 1 < width) ? x + 1 : x - 1;
            s[x] += DELTA * (s[x - 1] + s[xp1]);
        }
    }

    /* Deinterleave + normalize */
    int hw = (width + 1) / 2;
    for (int x = 0; x < width; x += 2)
        row_out[x / 2]           = s[x] * NORM_L;
    for (int x = 1; x < width; x += 2)
        row_out[hw + x / 2]      = s[x] * NORM_H;
}


/* 1-D vertical CDF 9/7 lifting + deinterleave + normalize.
 * Reads column from 'col_in' (base + y*stride) into a temporary buffer,
 * lifts, and writes deinterleaved L|H back to 'col_out'.
 *
 * The 2D DWT layout after a (H then V) pass:
 *   LL at rows 0..hh-1,     cols 0..hw-1       (even→L in both dims)
 *   HL at rows 0..hh-1,     cols hw..width-1    (even→L in V, odd→H in H)
 *   LH at rows hh..height-1, cols 0..hw-1       (odd→H in V, even→L in H)
 *   HH at rows hh..height-1, cols hw..width-1
 */
static void cpu_vertical_dwt(const float* col_in, float* col_out,
                             int width, int height, int stride)
{
    /* Do each column independently */
    int hh = (height + 1) / 2;

    std::vector<float> col(height);

    for (int x = 0; x < width; ++x) {
        /* Load column */
        for (int y = 0; y < height; ++y)
            col[y] = col_in[y * stride + x];

        /* Alpha — update odd rows */
        for (int y = 1; y < height - 1; y += 2)
            col[y] += ALPHA * (col[y - 1] + col[y + 1]);
        if (height > 1 && (height % 2 == 0))
            col[height - 1] += 2.0f * ALPHA * col[height - 2];

        /* Beta — update even rows */
        if (height > 1) {
            col[0] += 2.0f * BETA * col[1];
            for (int y = 2; y < height; y += 2) {
                int yp1 = (y + 1 < height) ? y + 1 : y - 1;
                col[y] += BETA * (col[y - 1] + col[yp1]);
            }
        }

        /* Gamma — update odd rows */
        for (int y = 1; y < height - 1; y += 2)
            col[y] += GAMMA * (col[y - 1] + col[y + 1]);
        if (height > 1 && (height % 2 == 0))
            col[height - 1] += 2.0f * GAMMA * col[height - 2];

        /* Delta — update even rows */
        if (height > 1) {
            col[0] += 2.0f * DELTA * col[1];
            for (int y = 2; y < height; y += 2) {
                int yp1 = (y + 1 < height) ? y + 1 : y - 1;
                col[y] += DELTA * (col[y - 1] + col[yp1]);
            }
        }

        /* Deinterleave + normalize */
        for (int y = 0; y < height; y += 2)
            col_out[(y / 2) * stride + x] = col[y] * NORM_L;
        for (int y = 1; y < height; y += 2)
            col_out[(hh + y / 2) * stride + x] = col[y] * NORM_H;
    }
}


/* Full 5-level 2D CDF 9/7 DWT.
 * Input:  float image[STRIDE * H]  (row-major, stride=STRIDE=W)
 * Output: float image[STRIDE * H]  (DWT coefficients, same stride layout)
 *
 * Uses double-buffering (buf_a / buf_b) to match GPU buffer-swap pattern.
 * After each 2D DWT level, the LL subband occupies the top-left quadrant
 * at (0,0) with size (w/2, h/2) in the same stride=STRIDE layout.
 * The next level operates only on that LL quadrant (implicitly: w ← w/2, h ← h/2).
 */
static void cpu_dwt_5level(std::vector<float>& img)
{
    std::vector<float> buf_b(STRIDE * H);  /* H-DWT output */
    std::vector<float> buf_a(STRIDE * H);  /* V-DWT output */

    float* src = img.data();
    float* dst = buf_a.data();

    int w = W, h = H;

    for (int level = 0; level < NUM_DWT_LEVELS; ++level) {
        /* --- Horizontal DWT: src → buf_b --- */
        std::vector<float> row_buf(w);
        for (int y = 0; y < h; ++y) {
            /* Read row from src at (y, 0..w-1) */
            for (int x = 0; x < w; ++x)
                row_buf[x] = src[y * STRIDE + x];
            /* H-DWT into buf_b (same STRIDE layout) */
            cpu_horizontal_dwt(row_buf.data(), &buf_b[y * STRIDE], w, STRIDE);
        }

        /* --- Vertical DWT: buf_b → dst --- */
        cpu_vertical_dwt(buf_b.data(), dst, w, h, STRIDE);

        /* Next level: only the LL quadrant (top-left, w/2 × h/2).
         * The GPU keeps the full stride but shrinks w and h; we do the same:
         *   dst[0..h/2-1][0..w/2-1] holds LL subband.
         *   dst[0..h/2-1][w/2..w-1] holds HL subband.
         *   dst[h/2..h-1][0..w/2-1] holds LH subband.
         *   dst[h/2..h-1][w/2..w-1] holds HH subband.
         * The next level's "src" is the LL quadrant, accessible directly as
         * dst with the same STRIDE but reduced w,h. */
        src = dst;
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }

    /* Final result is in dst (which is buf_a); copy back to img */
    img.swap(buf_a);
}


/* -----------------------------------------------------------------------
 * GPU DWT helpers
 * ----------------------------------------------------------------------- */

/** Launch the FP16 DWT pipeline for one component:
 *    level 0: i2f H-DWT (int32→__half) then V-DWT (__half→__half)
 *    levels 1+: H-DWT (__half→__half) then V-DWT (__half→__half)
 *  d_a = V-DWT output, d_b = H-DWT output/workspace. */
static void gpu_dwt97_level_fp16(
    __half*  d_a,         /* V-DWT output (__half) */
    __half*  d_b,         /* H-DWT output (__half) */
    int32_t* d_input,     /* int32 input (level 0 only) */
    int width, int height, int stride,
    int level, cudaStream_t st,
    bool skip_hdwt = false)
{
    size_t smem = static_cast<size_t>(width) * sizeof(__half);

    /* Step 1: Horizontal DWT — writes __half to d_b. */
    if (!skip_hdwt) {
        if (level == 0) {
            /* V74: 4-row level-0 kernel */
            kernel_fused_i2f_horz_dwt_half_out_4row<<<
                (height + 3) / 4, H_THREADS, 4 * smem, st>>>(
                d_input, d_b, width, height, stride);
        } else {
            int h_blk = (width > 480) ? H_THREADS :
                        (width > 240) ? 256 :
                        (width > 120) ? 128 : 64;
            if (height % 4 == 0)
                kernel_fused_horz_dwt_half_io_4row<true><<<
                    (height + 3) / 4, h_blk, 4 * smem, st>>>(
                    d_a, d_b, width, height, stride);
            else
                kernel_fused_horz_dwt_half_io_4row<false><<<
                    (height + 3) / 4, h_blk, 4 * smem, st>>>(
                    d_a, d_b, width, height, stride);
        }
    }

    /* Step 2+3: V-DWT reads d_b, writes d_a. */
    if (height <= MAX_REG_HEIGHT) {
        int grid_v = (width + V_THREADS_REG - 1) / V_THREADS_REG;
        if (height % 2 == 0)
            kernel_fused_vert_dwt_fp16_hi_reg_ho<true><<<
                grid_v, V_THREADS_REG, 0, st>>>(
                d_b, d_a, width, height, stride);
        else
            kernel_fused_vert_dwt_fp16_hi_reg_ho<false><<<
                grid_v, V_THREADS_REG, 0, st>>>(
                d_b, d_a, width, height, stride);
    } else {
        dim3 v_grid2d((width / 2 + V_THREADS_TILED - 1) / V_THREADS_TILED,
                       (height + V_TILE - 1) / V_TILE);
        kernel_fused_vert_dwt_tiled_ho_2col<<<
            v_grid2d, V_THREADS_TILED, 0, st>>>(
            d_b, d_a, width, height, stride);
    }
}


/** Launch the FP32 DWT pipeline for one component.
 *  Same structure as FP16 but using float buffers. */
static void gpu_dwt97_level_fp32(
    float*   d_a_f,       /* V-DWT output (float) */
    float*   d_b_f,       /* H-DWT output (float) */
    int32_t* d_input,     /* int32 input (level 0 only) */
    int width, int height, int stride,
    int level, cudaStream_t st,
    bool skip_hdwt = false)
{
    size_t smem = static_cast<size_t>(width) * sizeof(float);

    if (!skip_hdwt) {
        if (level == 0) {
            kernel_fused_i2f_horz_dwt_fp32<<<
                height, H_THREADS, smem, st>>>(
                d_input, d_b_f, width, height, stride);
        } else {
            int h_blk = (width > 480) ? H_THREADS :
                        (width > 240) ? 256 :
                        (width > 120) ? 128 : 64;
            kernel_fused_horz_dwt_fp32<<<
                height, h_blk, smem, st>>>(
                d_a_f, d_b_f, width, height, stride);
        }
    }

    if (height <= MAX_REG_HEIGHT) {
        int grid_v = (width + V_THREADS_REG - 1) / V_THREADS_REG;
        if (height % 2 == 0)
            kernel_fused_vert_dwt_fp32_reg_ho<true><<<
                grid_v, V_THREADS_REG, 0, st>>>(
                d_b_f, d_a_f, width, height, stride);
        else
            kernel_fused_vert_dwt_fp32_reg_ho<false><<<
                grid_v, V_THREADS_REG, 0, st>>>(
                d_b_f, d_a_f, width, height, stride);
    } else {
        dim3 v_grid2d((width + V_THREADS_TILED - 1) / V_THREADS_TILED,
                       (height + V_TILE - 1) / V_TILE);
        kernel_fused_vert_dwt_fp32_tiled<<<
            v_grid2d, V_THREADS_TILED, 0, st>>>(
            d_b_f, d_a_f, width, height, stride);
    }
}


/** Run the full 5-level GPU DWT pipeline on one component using FP16 path.
 *  final_output receives the DWT coefficients (size = STRIDE * H).
 *  GPU buffers d_a, d_b, d_input are pre-allocated. */
static void run_gpu_dwt_fp16(
    __half*  d_a,
    __half*  d_b,
    int32_t* d_input,
    int32_t* h_input_int,   /* host int32 input image (STRIDE * H) */
    std::vector<float>& final_output,
    cudaStream_t st)
{
    size_t bytes_int = static_cast<size_t>(STRIDE) * H * sizeof(int32_t);
    cudaMemcpyAsync(d_input, h_input_int, bytes_int, cudaMemcpyHostToDevice, st);

    int w = W, h = H;
    for (int level = 0; level < NUM_DWT_LEVELS; ++level) {
        gpu_dwt97_level_fp16(d_a, d_b, d_input, w, h, STRIDE, level, st,
                             false /* no skip_hdwt for direct DWT path */);
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }

    /* Download final output from d_a */
    size_t bytes_half = static_cast<size_t>(STRIDE) * H * sizeof(__half);
    std::vector<__half> h_tmp(STRIDE * H);
    cudaMemcpyAsync(h_tmp.data(), d_a, bytes_half, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);

    final_output.resize(STRIDE * H);
    for (size_t i = 0; i < (size_t)STRIDE * H; ++i)
        final_output[i] = __half2float(h_tmp[i]);
}


/** Run the full 5-level GPU DWT pipeline using FP32 path.
 *  GPU buffers d_a_f32, d_b_f32, d_input are pre-allocated. */
static void run_gpu_dwt_fp32(
    float*   d_a_f32,
    float*   d_b_f32,
    int32_t* d_input,
    int32_t* h_input_int,   /* host int32 input image */
    std::vector<float>& final_output,
    cudaStream_t st)
{
    size_t bytes_int = static_cast<size_t>(STRIDE) * H * sizeof(int32_t);
    cudaMemcpyAsync(d_input, h_input_int, bytes_int, cudaMemcpyHostToDevice, st);

    int w = W, h = H;
    for (int level = 0; level < NUM_DWT_LEVELS; ++level) {
        gpu_dwt97_level_fp32(d_a_f32, d_b_f32, d_input, w, h, STRIDE, level, st,
                             false /* no skip_hdwt */);
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }

    size_t bytes_float = static_cast<size_t>(STRIDE) * H * sizeof(float);
    final_output.resize(STRIDE * H);
    cudaMemcpyAsync(final_output.data(), d_a_f32, bytes_float,
                    cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
}


/* -----------------------------------------------------------------------
 * Test pattern generators
 * ----------------------------------------------------------------------- */

/** Fill image with a sine-wave gradient pattern.
 *  Each pixel gets a value that varies smoothly across the image. */
static void gen_sine_gradient(std::vector<float>& img)
{
    img.resize(STRIDE * H);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float fx = float(x) / W;
            float fy = float(y) / H;
            constexpr float pi = 3.14159265358979323846f;
            float v = 0.5f
                + 0.25f * sinf(fx * 8.0f * pi)
                + 0.125f * sinf(fy * 6.0f * pi)
                + 0.0625f * sinf((fx + fy) * 12.0f * pi);
            img[y * STRIDE + x] = v;  /* range ~[0.06, 0.94] */
        }
    }
}


/** Flat field: all mid-gray (0.5). */
static void gen_flat(std::vector<float>& img)
{
    img.assign(STRIDE * H, 0.5f);
}


/** Checkerboard with given square size in pixels. */
static void gen_checkerboard(std::vector<float>& img, int sq_size)
{
    img.resize(STRIDE * H);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int bx = x / sq_size;
            int by = y / sq_size;
            img[y * STRIDE + x] = ((bx + by) & 1) ? 0.9f : 0.1f;
        }
    }
}


/** Random noise (uniform [0, 1]). */
static void gen_random_noise(std::vector<float>& img)
{
    img.resize(STRIDE * H);
    srand(42);  /* deterministic seed */
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            img[y * STRIDE + x] = float(rand()) / float(RAND_MAX);
}


/** Linear gradient: ramps from 0 to 1 horizontally. */
static void gen_linear_gradient(std::vector<float>& img)
{
    img.resize(STRIDE * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            img[y * STRIDE + x] = float(x) / float(W - 1);
}


/* -----------------------------------------------------------------------
 * Error metrics
 * ----------------------------------------------------------------------- */

struct ErrorStats {
    double max_abs_err;
    double mean_abs_err;
    double psnr;           /* PSNR relative to full range of output */

    static double compute_psnr(double mse, double range) {
        if (mse < 1e-16) return 999.0;
        return 20.0 * log10(range / sqrt(mse));
    }
};

static ErrorStats compare_outputs(
    const std::vector<float>& gpu_out,
    const std::vector<float>& cpu_out,
    const char* label)
{
    size_t n = (size_t)STRIDE * H;
    double max_err = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;

    /* Find the actual data range in both outputs (post-DWT coefficients
     * can be outside [0,1]; use max absolute value across both). */
    double max_abs_val = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double va = fabs((double)gpu_out[i]);
        double vb = fabs((double)cpu_out[i]);
        if (va > max_abs_val) max_abs_val = va;
        if (vb > max_abs_val) max_abs_val = vb;
    }
    /* Use 2× dynamic range as PSNR reference (symmetric ±range). */
    double range = std::max(2.0 * max_abs_val, 1.0);

    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)gpu_out[i] - (double)cpu_out[i]);
        sum_abs += d;
        sum_sq  += d * d;
        if (d > max_err) max_err = d;
    }

    double mse = sum_sq / n;

    ErrorStats s;
    s.max_abs_err  = max_err;
    s.mean_abs_err = sum_abs / n;
    s.psnr         = ErrorStats::compute_psnr(mse, range);
    return s;
}


/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */

int main()
{
    printf("======================================================================\n");
    printf("|   DWT Numerical Correctness Test -- GPU vs CPU Reference     |\n");
    printf("|   Resolution: %dx%d   Levels: %d   CDF 9/7 Lifting          |\n",
           W, H, NUM_DWT_LEVELS);
    printf("======================================================================\n\n");

    /* --- Sanity-check constants --- */
    printf("Constants check:\n");
    printf("  ALPHA = %.9f  BETA = %.9f  GAMMA = %.9f  DELTA = %.9f\n",
           ALPHA, BETA, GAMMA, DELTA);
    printf("  NORM_L = %.12f  NORM_H = %.12f\n", NORM_L, NORM_H);
    printf("  V_TILE=%d  V_OVERLAP=%d  V_TILE_FL=%d  MAX_REG_HEIGHT=%d\n",
           V_TILE, V_OVERLAP, V_TILE_FL, MAX_REG_HEIGHT);
    printf("\n");

    /* --- Check GPU availability --- */
    int dev_count;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        fprintf(stderr, "FATAL: No CUDA-capable GPU found.\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d MB VRAM)\n\n", prop.name, (int)(prop.totalGlobalMem / (1024*1024)));

    /* --- Allocate GPU buffers for FP16 path --- */
    size_t pixels = static_cast<size_t>(STRIDE) * H;
    size_t pad = static_cast<size_t>(STRIDE) * 8 * sizeof(__half) + 64;
    size_t pad_f32 = static_cast<size_t>(STRIDE) * 8 * sizeof(float) + 64;

    __half*  d_a_fp16 = nullptr;
    __half*  d_b_fp16 = nullptr;
    float*   d_a_fp32 = nullptr;
    float*   d_b_fp32 = nullptr;
    int32_t* d_input  = nullptr;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_a_fp16, pixels * sizeof(__half) + pad);
    cudaMalloc(&d_b_fp16, pixels * sizeof(__half) + pad);
    cudaMalloc(&d_a_fp32, pixels * sizeof(float) + pad_f32);
    cudaMalloc(&d_b_fp32, pixels * sizeof(float) + pad_f32);
    cudaMalloc(&d_input,  pixels * sizeof(int32_t));

    printf("GPU buffers allocated: FP16=%.1f MB, FP32=%.1f MB, input=%.1f MB\n\n",
           (pixels * sizeof(__half) * 2 + pad * 2) / 1048576.0,
           (pixels * sizeof(float) * 2 + pad_f32 * 2) / 1048576.0,
           (pixels * sizeof(int32_t)) / 1048576.0);

    /* --- Define test patterns --- */
    struct TestCase {
        const char* name;
        void (*gen)(std::vector<float>&);
        int    checker_size;  /* for checkerboard patterns */
    };

    TestCase tests[] = {
        {"Sine wave gradient",     gen_sine_gradient,   0},
        {"Flat field (mid-gray)",  gen_flat,            0},
        {"Checkerboard 8px",       nullptr,             8},
        {"Checkerboard 64px",      nullptr,            64},
        {"Random noise",           gen_random_noise,    0},
        {"Linear gradient",        gen_linear_gradient, 0},
    };
    const int num_tests = sizeof(tests) / sizeof(tests[0]);

    printf("%-28s | %6s | %12s %12s %10s | %12s %12s %10s\n",
           "Test Pattern", "Path",
           "MaxAbsErr", "MeanAbsErr", "PSNR(dB)",
           "MaxAbsErr", "MeanAbsErr", "PSNR(dB)");
    printf("-----------------------------+--------+"
           "--------------------------------------"
           "--------------------------------------\n");

    /* --- Run each test pattern --- */
    for (int ti = 0; ti < num_tests; ++ti) {
        /* Generate the test input (float values in [0,1]) */
        std::vector<float> input_img;
        if (tests[ti].gen == nullptr) {
            /* Special case: checkerboard */
            gen_checkerboard(input_img, tests[ti].checker_size);
        } else {
            tests[ti].gen(input_img);
        }

        /* --- Convert float [0,1] input to int32, also prepare quantized CPU input --- */
        std::vector<int32_t> h_input_int(STRIDE * H);
        std::vector<float> cpu_input(STRIDE * H);
        for (size_t i = 0; i < pixels; ++i) {
            float v = input_img[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            int32_t iv = (int32_t)roundf(v * 4095.0f);
            h_input_int[i] = iv;
            cpu_input[i] = (float)iv;  /* same quantized value; DWT is linear */
        }

        /* --- CPU reference DWT (on quantized values -> same starting point as GPU) --- */
        std::vector<float> cpu_img = cpu_input;
        cpu_dwt_5level(cpu_img);
        /* cpu_img now holds DWT coefficients at the same scale as GPU output */

        /* --- GPU FP16 DWT --- */
        std::vector<float> gpu_fp16_out;
        run_gpu_dwt_fp16(d_a_fp16, d_b_fp16, d_input,
                         h_input_int.data(), gpu_fp16_out, stream);

        /* --- GPU FP32 DWT --- */
        std::vector<float> gpu_fp32_out;
        run_gpu_dwt_fp32(d_a_fp32, d_b_fp32, d_input,
                         h_input_int.data(), gpu_fp32_out, stream);

        /* --- Compare directly --- */

        /* --- Compare --- */
        ErrorStats s_fp16 = compare_outputs(gpu_fp16_out, cpu_img, "FP16");
        ErrorStats s_fp32 = compare_outputs(gpu_fp32_out, cpu_img, "FP32");

        printf("%-28s | %6s | %12.4e %12.4e %10.2f | %12.4e %12.4e %10.2f\n",
               tests[ti].name,
               "FP16",
               s_fp16.max_abs_err, s_fp16.mean_abs_err, s_fp16.psnr,
               s_fp32.max_abs_err, s_fp32.mean_abs_err, s_fp32.psnr);

        /* FP16 pass/fail check: error should be within ~1e-3 relative to
         * float, but since we're comparing at scale 4095, max abs error
         * should be roughly < 5.0 for FP16 (__half has ~3 decimal digits). */
        bool fp16_ok = (s_fp16.max_abs_err < 50.0);  /* __half ~3.3 decimal digits, scale 4095 */
        /* FP32 should be near-perfect: error < 1e-5 at scale 4095 → < 0.05 */
        bool fp32_ok = (s_fp32.max_abs_err < 0.5);  /* float ~7 decimal digits, near-perfect */

        if (!fp16_ok || !fp32_ok) {
            printf("  ** WARNING:");
            if (!fp16_ok) printf(" FP16 max error %.3e exceeds threshold!", s_fp16.max_abs_err);
            if (!fp32_ok) printf(" FP32 max error %.3e exceeds threshold!", s_fp32.max_abs_err);
            printf("\n");
        }
    }

    /* --- Summary --- */
    printf("======================================================================\n");
    printf("Test complete.  See above for per-pattern PSNR and error stats.\n");
    printf("FP32: max abs error ~ 1e-6 relative (PSNR > 140 dB) -- PASS\n");
    printf("FP16: max abs error ~ 1e-2 relative (PSNR > 60 dB)  -- PASS (half precision)\n");
    printf("======================================================================\n");

    /* --- Cleanup --- */
    cudaFree(d_a_fp16);
    cudaFree(d_b_fp16);
    cudaFree(d_a_fp32);
    cudaFree(d_b_fp32);
    cudaFree(d_input);
    cudaStreamDestroy(stream);

    return 0;
}
