/*
    3-Way DCP Export Benchmark: CPU (OpenJPEG) vs GPU (CUDA) vs GPU (Slang)

    Build:
      nvcc -std=c++17 -O2 \
        test/benchmark_cpu_cuda_slang.cu \
        src/lib/cuda_j2k_encoder.cu \
        src/lib/slang_j2k_encoder.cu \
        -I src/lib -I /usr/include/openjpeg-2.5 \
        -lopenjp2 -lcudart \
        -o test/benchmark_3way

    Usage:
      ffmpeg -i video.mp4 -vf "scale=2048:1080,format=rgb24" -frames:v 50 -f rawvideo /tmp/frames.rgb
      ./test/benchmark_3way /tmp/frames.rgb 2048 1080 50
*/

#include "cuda_j2k_encoder.h"
#include "slang_j2k_encoder.h"
#include <cuda_runtime.h>
#include <openjpeg.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>


using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;


/* ===== CPU OpenJPEG JPEG2000 Encoder ===== */

static std::vector<uint8_t>
cpu_openjpeg_encode(const int32_t* const xyz[3], int width, int height,
                    int64_t bit_rate, int fps)
{
    /* Create OpenJPEG image */
    opj_image_cmptparm_t cmptparm[3];
    memset(cmptparm, 0, sizeof(cmptparm));
    for (int i = 0; i < 3; ++i) {
        cmptparm[i].prec = 12;
        cmptparm[i].bpp = 12;
        cmptparm[i].sgnd = 0;
        cmptparm[i].dx = 1;
        cmptparm[i].dy = 1;
        cmptparm[i].w = width;
        cmptparm[i].h = height;
    }

    auto image = opj_image_create(3, cmptparm, OPJ_CLRSPC_SYCC);
    if (!image) return {};

    image->x0 = 0;
    image->y0 = 0;
    image->x1 = width;
    image->y1 = height;

    /* Copy XYZ data */
    size_t pixels = size_t(width) * height;
    for (int c = 0; c < 3; ++c) {
        memcpy(image->comps[c].data, xyz[c], pixels * sizeof(int32_t));
    }

    /* Set up encoder */
    auto encoder = opj_create_compress(OPJ_CODEC_J2K);
    if (!encoder) { opj_image_destroy(image); return {}; }

    opj_cparameters_t params;
    opj_set_default_encoder_parameters(&params);
    params.tcp_numlayers = 1;
    params.cp_disto_alloc = 1;
    params.tcp_rates[0] = 0;  /* Lossless for layer, control via max_cs_size */
    params.numresolution = 6;  /* 5 DWT levels + 1 */
    params.irreversible = 1;   /* CDF 9/7 */
    params.prog_order = OPJ_LRCP;

    /* Target size from bit rate */
    int64_t frame_bits = bit_rate / fps;
    params.max_cs_size = static_cast<int>(frame_bits / 8);

    if (!opj_setup_encoder(encoder, &params, image)) {
        opj_destroy_codec(encoder);
        opj_image_destroy(image);
        return {};
    }

    /* Encode to memory */
    opj_stream_t* stream = opj_stream_default_create(OPJ_FALSE);
    if (!stream) {
        opj_destroy_codec(encoder);
        opj_image_destroy(image);
        return {};
    }

    /* Custom memory stream */
    struct MemBuf {
        std::vector<uint8_t> data;
        size_t pos = 0;
    };
    auto membuf = new MemBuf();

    opj_stream_set_user_data(stream, membuf, [](void* p) { delete static_cast<MemBuf*>(p); });
    opj_stream_set_user_data_length(stream, 0);

    opj_stream_set_write_function(stream, [](void* buf, OPJ_SIZE_T nb, void* ud) -> OPJ_SIZE_T {
        auto mb = static_cast<MemBuf*>(ud);
        auto p = static_cast<const uint8_t*>(buf);
        mb->data.insert(mb->data.end(), p, p + nb);
        mb->pos += nb;
        return nb;
    });

    opj_stream_set_seek_function(stream, [](OPJ_OFF_T pos, void* ud) -> OPJ_BOOL {
        auto mb = static_cast<MemBuf*>(ud);
        if (size_t(pos) > mb->data.size()) mb->data.resize(size_t(pos));
        mb->pos = size_t(pos);
        return OPJ_TRUE;
    });

    opj_stream_set_skip_function(stream, [](OPJ_OFF_T nb, void* ud) -> OPJ_OFF_T {
        auto mb = static_cast<MemBuf*>(ud);
        mb->pos += size_t(nb);
        if (mb->pos > mb->data.size()) mb->data.resize(mb->pos);
        return nb;
    });

    bool ok = opj_start_compress(encoder, image, stream) &&
              opj_encode(encoder, stream) &&
              opj_end_compress(encoder, stream);

    std::vector<uint8_t> result;
    if (ok) {
        result = std::move(membuf->data);
    }

    opj_stream_destroy(stream);
    opj_destroy_codec(encoder);
    opj_image_destroy(image);

    return result;
}


/* ===== Helpers ===== */

static void rgb_to_xyz(const uint8_t* rgb, int width, int height,
                       std::vector<int32_t>& x, std::vector<int32_t>& y, std::vector<int32_t>& z)
{
    size_t n = size_t(width) * height;
    x.resize(n); y.resize(n); z.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float r = rgb[i*3] / 255.0f, g = rgb[i*3+1] / 255.0f, b = rgb[i*3+2] / 255.0f;
        x[i] = int32_t(std::min(4095.0f, (r*0.4124f + g*0.3576f + b*0.1805f) * 4095));
        y[i] = int32_t(std::min(4095.0f, (r*0.2126f + g*0.7152f + b*0.0722f) * 4095));
        z[i] = int32_t(std::min(4095.0f, (r*0.0193f + g*0.1192f + b*0.9505f) * 4095));
    }
}

static bool verify_j2k(const std::vector<uint8_t>& d) {
    if (d.size() < 4) return false;
    return d[0]==0xFF && d[1]==0x4F && d[d.size()-2]==0xFF && d[d.size()-1]==0xD9;
}


/* ===== Main ===== */

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <frames.rgb> <width> <height> <num_frames>" << std::endl;
        return 1;
    }

    const char* input = argv[1];
    int width = std::atoi(argv[2]), height = std::atoi(argv[3]), num_frames = std::atoi(argv[4]);
    size_t frame_bytes = size_t(width) * height * 3;
    int64_t bit_rate = 100000000;
    int fps = 24;

    std::cout << "===========================================================" << std::endl;
    std::cout << " DCP Export Benchmark: CPU (OpenJPEG) vs CUDA vs Slang" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "Resolution:  " << width << "x" << height << std::endl;
    std::cout << "Frames:      " << num_frames << std::endl;
    std::cout << "Bit rate:    " << (bit_rate / 1000000) << " Mbit/s" << std::endl;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        std::cout << "GPU:         " << prop.name << std::endl;

    /* Load frames */
    std::cout << "\nLoading frames..." << std::endl;
    std::ifstream fin(input, std::ios::binary);
    if (!fin) { std::cerr << "Cannot open " << input << std::endl; return 1; }

    std::vector<std::vector<uint8_t>> raw;
    for (int i = 0; i < num_frames; ++i) {
        std::vector<uint8_t> f(frame_bytes);
        fin.read(reinterpret_cast<char*>(f.data()), frame_bytes);
        if (fin.gcount() != std::streamsize(frame_bytes)) { num_frames = i; break; }
        raw.push_back(std::move(f));
    }

    /* Convert to XYZ */
    struct XYZ { std::vector<int32_t> x, y, z; };
    std::vector<XYZ> xyz(num_frames);
    for (int i = 0; i < num_frames; ++i)
        rgb_to_xyz(raw[i].data(), width, height, xyz[i].x, xyz[i].y, xyz[i].z);

    std::cout << "Loaded " << num_frames << " frames" << std::endl;

    /* =============================== */
    /* 1. CPU OpenJPEG                 */
    /* =============================== */
    std::cout << "\n--- [1] CPU OpenJPEG JPEG2000 ---" << std::endl;
    std::vector<double> cpu_times;
    size_t cpu_bytes = 0;
    int cpu_valid = 0;

    auto cpu_start = Clock::now();
    for (int i = 0; i < num_frames; ++i) {
        const int32_t* p[3] = { xyz[i].x.data(), xyz[i].y.data(), xyz[i].z.data() };
        auto t0 = Clock::now();
        auto enc = cpu_openjpeg_encode(p, width, height, bit_rate, fps);
        auto t1 = Clock::now();
        double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
        cpu_times.push_back(ms);
        cpu_bytes += enc.size();
        if (verify_j2k(enc)) ++cpu_valid;
        if (i == 0 || (i+1) % 10 == 0)
            std::cout << "  Frame " << (i+1) << "/" << num_frames << " - "
                      << std::fixed << std::setprecision(1) << ms << " ms ("
                      << enc.size() << " bytes)" << std::endl;
    }
    auto cpu_end = Clock::now();
    double cpu_total = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    double cpu_avg = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / cpu_times.size();

    /* =============================== */
    /* 2. CUDA J2K                     */
    /* =============================== */
    std::cout << "\n--- [2] CUDA J2K (DWT 9/7 on GPU) ---" << std::endl;
    CudaJ2KEncoder cuda_enc;
    if (!cuda_enc.is_initialized()) { std::cerr << "CUDA init failed" << std::endl; return 1; }

    /* Warm up */
    { const int32_t* p[3] = { xyz[0].x.data(), xyz[0].y.data(), xyz[0].z.data() };
      cuda_enc.encode(p, width, height, bit_rate, fps, false, false); }

    std::vector<double> cuda_times;
    size_t cuda_bytes = 0;
    int cuda_valid = 0;

    auto cuda_start = Clock::now();
    for (int i = 0; i < num_frames; ++i) {
        const int32_t* p[3] = { xyz[i].x.data(), xyz[i].y.data(), xyz[i].z.data() };
        auto t0 = Clock::now();
        auto enc = cuda_enc.encode(p, width, height, bit_rate, fps, false, false);
        auto t1 = Clock::now();
        double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
        cuda_times.push_back(ms);
        cuda_bytes += enc.size();
        if (verify_j2k(enc)) ++cuda_valid;
        if (i == 0 || (i+1) % 10 == 0)
            std::cout << "  Frame " << (i+1) << "/" << num_frames << " - "
                      << std::fixed << std::setprecision(1) << ms << " ms ("
                      << enc.size() << " bytes)" << std::endl;
    }
    auto cuda_end = Clock::now();
    double cuda_total = duration_cast<milliseconds>(cuda_end - cuda_start).count();
    double cuda_avg = std::accumulate(cuda_times.begin(), cuda_times.end(), 0.0) / cuda_times.size();

    /* =============================== */
    /* 3. Slang J2K                    */
    /* =============================== */
    std::cout << "\n--- [3] Slang J2K (DWT 9/7 on GPU, Slang-generated) ---" << std::endl;
    SlangJ2KEncoder slang_enc;
    if (!slang_enc.is_initialized()) { std::cerr << "Slang init failed" << std::endl; return 1; }

    /* Warm up */
    { const int32_t* p[3] = { xyz[0].x.data(), xyz[0].y.data(), xyz[0].z.data() };
      slang_enc.encode(p, width, height, bit_rate, fps, false, false); }

    std::vector<double> slang_times;
    size_t slang_bytes = 0;
    int slang_valid = 0;

    auto slang_start = Clock::now();
    for (int i = 0; i < num_frames; ++i) {
        const int32_t* p[3] = { xyz[i].x.data(), xyz[i].y.data(), xyz[i].z.data() };
        auto t0 = Clock::now();
        auto enc = slang_enc.encode(p, width, height, bit_rate, fps, false, false);
        auto t1 = Clock::now();
        double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
        slang_times.push_back(ms);
        slang_bytes += enc.size();
        if (verify_j2k(enc)) ++slang_valid;
        if (i == 0 || (i+1) % 10 == 0)
            std::cout << "  Frame " << (i+1) << "/" << num_frames << " - "
                      << std::fixed << std::setprecision(1) << ms << " ms ("
                      << enc.size() << " bytes)" << std::endl;
    }
    auto slang_end = Clock::now();
    double slang_total = duration_cast<milliseconds>(slang_end - slang_start).count();
    double slang_avg = std::accumulate(slang_times.begin(), slang_times.end(), 0.0) / slang_times.size();

    /* =============================== */
    /*  Results                        */
    /* =============================== */
    std::cout << "\n===========================================================" << std::endl;
    std::cout << "                       RESULTS" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "                  CPU(OpenJPEG)    CUDA J2K      Slang J2K" << std::endl;
    std::cout << "  Total (ms):     " << std::setw(10) << cpu_total
              << "    " << std::setw(10) << cuda_total
              << "    " << std::setw(10) << slang_total << std::endl;
    std::cout << "  Avg/frame(ms):  " << std::setw(10) << cpu_avg
              << "    " << std::setw(10) << cuda_avg
              << "    " << std::setw(10) << slang_avg << std::endl;
    std::cout << "  FPS:            " << std::setw(10) << (num_frames*1000.0/cpu_total)
              << "    " << std::setw(10) << (num_frames*1000.0/cuda_total)
              << "    " << std::setw(10) << (num_frames*1000.0/slang_total) << std::endl;
    std::cout << "  J2K valid:      " << std::setw(7) << cpu_valid << "/" << num_frames
              << "    " << std::setw(7) << cuda_valid << "/" << num_frames
              << "    " << std::setw(7) << slang_valid << "/" << num_frames << std::endl;
    std::cout << "  Output (MB):    " << std::setw(10) << std::setprecision(2) << (cpu_bytes/1048576.0)
              << "    " << std::setw(10) << (cuda_bytes/1048576.0)
              << "    " << std::setw(10) << (slang_bytes/1048576.0) << std::endl;

    std::cout << "\n  Speedup vs CPU:" << std::endl;
    std::cout << "    CUDA:  " << std::setprecision(2) << (cpu_total / cuda_total) << "x" << std::endl;
    std::cout << "    Slang: " << std::setprecision(2) << (cpu_total / slang_total) << "x" << std::endl;

    std::cout << "===========================================================" << std::endl;
    return 0;
}
