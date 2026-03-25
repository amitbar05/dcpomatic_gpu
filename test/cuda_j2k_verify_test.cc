/*
    CUDA J2K Encoder: Verification and Benchmark

    Tests that the GPU JPEG2000 encoder produces valid J2K codestreams
    and compares performance against CPU encoding simulation.

    Build:
      nvcc -c -std=c++17 -O2 -o /tmp/cuda_j2k_encoder.o src/lib/cuda_j2k_encoder.cu -I src/lib
      g++ -std=c++17 -O2 test/cuda_j2k_verify_test.cc /tmp/cuda_j2k_encoder.o \
          -I src -I src/lib -lcudart -o test/cuda_j2k_verify

    Usage:
      ffmpeg -i video.mp4 -vf "scale=2048:1080,format=rgb24" -frames:v 48 -f rawvideo /tmp/frames.rgb
      ./test/cuda_j2k_verify /tmp/frames.rgb 2048 1080 48
*/

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cassert>

#include "lib/cuda_j2k_encoder.h"


using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;


/** Verify J2K codestream has valid markers */
struct J2KVerifyResult {
    bool has_soc = false;     /* FF 4F - Start of Codestream */
    bool has_siz = false;     /* FF 51 - Image size */
    bool has_cod = false;     /* FF 52 - Coding style */
    bool has_qcd = false;     /* FF 5C - Quantization */
    bool has_sot = false;     /* FF 90 - Start of tile */
    bool has_sod = false;     /* FF 93 - Start of data */
    bool has_eoc = false;     /* FF D9 - End of codestream */
    int siz_width = 0;
    int siz_height = 0;
    int siz_components = 0;
    int siz_bit_depth = 0;
    size_t total_size = 0;
    bool valid = false;

    void print(const std::string& label) const {
        std::cout << "  [" << label << " J2K Verification]" << std::endl;
        std::cout << "    SOC (FF4F): " << (has_soc ? "OK" : "MISSING") << std::endl;
        std::cout << "    SIZ (FF51): " << (has_siz ? "OK" : "MISSING");
        if (has_siz) {
            std::cout << " (" << siz_width << "x" << siz_height
                      << ", " << siz_components << " components, "
                      << (siz_bit_depth + 1) << "-bit)";
        }
        std::cout << std::endl;
        std::cout << "    COD (FF52): " << (has_cod ? "OK" : "MISSING") << std::endl;
        std::cout << "    QCD (FF5C): " << (has_qcd ? "OK" : "MISSING") << std::endl;
        std::cout << "    SOT (FF90): " << (has_sot ? "OK" : "MISSING") << std::endl;
        std::cout << "    SOD (FF93): " << (has_sod ? "OK" : "MISSING") << std::endl;
        std::cout << "    EOC (FFD9): " << (has_eoc ? "OK" : "MISSING") << std::endl;
        std::cout << "    Size:       " << total_size << " bytes" << std::endl;
        std::cout << "    RESULT:     " << (valid ? "PASS" : "FAIL") << std::endl;
    }
};


static J2KVerifyResult
verify_j2k(const std::vector<uint8_t>& data)
{
    J2KVerifyResult r;
    r.total_size = data.size();

    if (data.size() < 4) return r;

    /* Check SOC */
    if (data[0] == 0xFF && data[1] == 0x4F) r.has_soc = true;

    /* Check EOC */
    if (data[data.size()-2] == 0xFF && data[data.size()-1] == 0xD9) r.has_eoc = true;

    /* Scan for marker segments */
    size_t pos = 2;  /* Skip SOC */
    while (pos + 1 < data.size()) {
        if (data[pos] != 0xFF) { ++pos; continue; }

        uint8_t marker_lo = data[pos + 1];

        /* Skip filler bytes (FF followed by 00 or another FF) */
        if (marker_lo == 0x00 || marker_lo == 0xFF) { pos += 2; continue; }

        if (marker_lo == 0x51) {  /* SIZ */
            r.has_siz = true;
            if (pos + 10 < data.size()) {
                size_t base = pos + 6;  /* marker(2) + Lsiz(2) + Rsiz(2) */
                if (base + 8 <= data.size()) {
                    r.siz_width = (data[base] << 24) | (data[base+1] << 16) |
                                  (data[base+2] << 8) | data[base+3];
                    r.siz_height = (data[base+4] << 24) | (data[base+5] << 16) |
                                   (data[base+6] << 8) | data[base+7];
                }
                if (pos + 40 <= data.size()) {
                    r.siz_components = (data[pos + 38] << 8) | data[pos + 39];
                }
                if (pos + 41 <= data.size()) {
                    r.siz_bit_depth = data[pos + 40];
                }
            }
        } else if (marker_lo == 0x52) {
            r.has_cod = true;
        } else if (marker_lo == 0x5C) {
            r.has_qcd = true;
        } else if (marker_lo == 0x90) {
            r.has_sot = true;
        } else if (marker_lo == 0x93) {
            r.has_sod = true;
            break;  /* Data follows SOD, stop scanning */
        }

        /* Markers with length fields (all markers 0x51-0xFF except SOC/SOD/EOC/SOT special) */
        bool has_length = (marker_lo >= 0x40 && marker_lo != 0x4F && marker_lo != 0x93 && marker_lo != 0xD9);
        /* SOT (0x90) also has a length */
        if (marker_lo == 0x90) has_length = true;

        if (has_length && pos + 3 < data.size()) {
            uint16_t len = (data[pos + 2] << 8) | data[pos + 3];
            pos += 2 + len;
        } else {
            pos += 2;
        }
    }

    r.valid = r.has_soc && r.has_siz && r.has_cod && r.has_qcd &&
              r.has_sot && r.has_sod && r.has_eoc;

    return r;
}


/** Convert 8-bit RGB to 12-bit XYZ-like planar (simulated color conversion) */
static void
rgb_to_xyz_planes(const uint8_t* rgb, int width, int height,
                  std::vector<int32_t>& x_plane,
                  std::vector<int32_t>& y_plane,
                  std::vector<int32_t>& z_plane)
{
    size_t pixels = static_cast<size_t>(width) * height;
    x_plane.resize(pixels);
    y_plane.resize(pixels);
    z_plane.resize(pixels);

    for (size_t i = 0; i < pixels; ++i) {
        float r = rgb[i * 3 + 0] / 255.0f;
        float g = rgb[i * 3 + 1] / 255.0f;
        float b = rgb[i * 3 + 2] / 255.0f;

        /* Simplified RGB to XYZ (scaled to 12-bit range 0-4095) */
        x_plane[i] = static_cast<int32_t>(std::min(4095.0f,
            (r * 0.4124f + g * 0.3576f + b * 0.1805f) * 4095.0f));
        y_plane[i] = static_cast<int32_t>(std::min(4095.0f,
            (r * 0.2126f + g * 0.7152f + b * 0.0722f) * 4095.0f));
        z_plane[i] = static_cast<int32_t>(std::min(4095.0f,
            (r * 0.0193f + g * 0.1192f + b * 0.9505f) * 4095.0f));
    }
}


/** CPU J2K encode simulation (for timing comparison) */
static std::vector<uint8_t>
cpu_j2k_simulate(const int32_t* const planes[3], int width, int height)
{
    size_t pixels = static_cast<size_t>(width) * height;
    std::vector<uint8_t> output;
    output.reserve(pixels);

    /* SOC */
    output.push_back(0xFF); output.push_back(0x4F);
    /* SIZ (minimal) */
    output.push_back(0xFF); output.push_back(0x51);

    /* Simulate DWT + quantization workload */
    for (int c = 0; c < 3; ++c) {
        for (size_t i = 0; i < pixels; i += 8) {
            float acc = 0;
            for (int j = 0; j < 8 && i + j < pixels; ++j) {
                acc += planes[c][i + j] * 0.125f;
            }
            output.push_back(static_cast<uint8_t>(static_cast<int>(acc) & 0xFF));
        }
    }

    /* Pad to min size */
    while (output.size() < 16384) output.push_back(0);

    /* EOC */
    output.push_back(0xFF); output.push_back(0xD9);
    return output;
}


int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <frames.rgb> <width> <height> <num_frames>" << std::endl;
        return 1;
    }

    const char* input = argv[1];
    int width = std::atoi(argv[2]);
    int height = std::atoi(argv[3]);
    int num_frames = std::atoi(argv[4]);
    size_t frame_bytes = static_cast<size_t>(width) * height * 3;

    std::cout << "================================================" << std::endl;
    std::cout << " CUDA J2K Encoder: DCP Export Verification" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Resolution:  " << width << "x" << height << std::endl;
    std::cout << "Frames:      " << num_frames << std::endl;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::cout << "GPU:         " << prop.name << std::endl;
    }

    /* Load frames */
    std::cout << "\nLoading frames..." << std::endl;
    std::ifstream fin(input, std::ios::binary);
    if (!fin) { std::cerr << "Cannot open " << input << std::endl; return 1; }

    std::vector<std::vector<uint8_t>> raw_frames;
    for (int i = 0; i < num_frames; ++i) {
        std::vector<uint8_t> f(frame_bytes);
        fin.read(reinterpret_cast<char*>(f.data()), frame_bytes);
        if (fin.gcount() != static_cast<std::streamsize>(frame_bytes)) { num_frames = i; break; }
        raw_frames.push_back(std::move(f));
    }
    std::cout << "Loaded " << num_frames << " frames" << std::endl;

    /* Convert to XYZ planes */
    std::cout << "Converting to XYZ..." << std::endl;
    struct XYZFrame {
        std::vector<int32_t> x, y, z;
    };
    std::vector<XYZFrame> xyz_frames(num_frames);
    for (int i = 0; i < num_frames; ++i) {
        rgb_to_xyz_planes(raw_frames[i].data(), width, height,
                         xyz_frames[i].x, xyz_frames[i].y, xyz_frames[i].z);
    }

    /* ===== GPU J2K Encoding ===== */
    std::cout << "\n--- GPU JPEG2000 Encoding (CUDA DWT + J2K) ---" << std::endl;

    CudaJ2KEncoder gpu_encoder;
    if (!gpu_encoder.is_initialized()) {
        std::cerr << "ERROR: Failed to initialize CUDA J2K encoder" << std::endl;
        return 1;
    }

    /* Warm up */
    {
        const int32_t* planes[3] = { xyz_frames[0].x.data(), xyz_frames[0].y.data(), xyz_frames[0].z.data() };
        gpu_encoder.encode(planes, width, height, 100000000, 24, false, false);
    }

    std::vector<double> gpu_times;
    size_t gpu_total_bytes = 0;
    int gpu_valid = 0;
    std::vector<uint8_t> first_j2k;

    auto gpu_start = Clock::now();
    for (int i = 0; i < num_frames; ++i) {
        const int32_t* planes[3] = { xyz_frames[i].x.data(), xyz_frames[i].y.data(), xyz_frames[i].z.data() };

        auto t0 = Clock::now();
        auto encoded = gpu_encoder.encode(planes, width, height, 100000000, 24, false, false);
        auto t1 = Clock::now();

        double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
        gpu_times.push_back(ms);
        gpu_total_bytes += encoded.size();

        auto vr = verify_j2k(encoded);
        if (vr.valid) ++gpu_valid;

        if (i == 0) first_j2k = encoded;

        if (i == 0 || (i + 1) % 10 == 0) {
            std::cout << "  Frame " << (i + 1) << "/" << num_frames
                      << " - " << std::fixed << std::setprecision(2) << ms << " ms"
                      << " (" << encoded.size() << " bytes)"
                      << " [J2K " << (vr.valid ? "VALID" : "INVALID") << "]"
                      << std::endl;
        }
    }
    auto gpu_end = Clock::now();
    double gpu_total_ms = duration_cast<milliseconds>(gpu_end - gpu_start).count();
    double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();

    /* Verify first frame in detail */
    if (!first_j2k.empty()) {
        auto vr = verify_j2k(first_j2k);
        vr.print("GPU Frame 1");

        /* Save sample J2K file */
        std::ofstream fout("test/gpu_frame_sample.j2c", std::ios::binary);
        fout.write(reinterpret_cast<const char*>(first_j2k.data()), first_j2k.size());
        std::cout << "    Saved to: test/gpu_frame_sample.j2c" << std::endl;
    }

    std::cout << "\n  GPU Results:" << std::endl;
    std::cout << "    Total time:     " << gpu_total_ms << " ms" << std::endl;
    std::cout << "    Avg/frame:      " << gpu_avg << " ms" << std::endl;
    std::cout << "    FPS:            " << (num_frames * 1000.0 / gpu_total_ms) << std::endl;
    std::cout << "    Valid J2K:      " << gpu_valid << "/" << num_frames << std::endl;
    std::cout << "    Total output:   " << (gpu_total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

    /* ===== CPU J2K Encoding (simulated) ===== */
    std::cout << "\n--- CPU JPEG2000 Encoding (simulated) ---" << std::endl;

    std::vector<double> cpu_times;
    size_t cpu_total_bytes = 0;

    auto cpu_start = Clock::now();
    for (int i = 0; i < num_frames; ++i) {
        const int32_t* planes[3] = { xyz_frames[i].x.data(), xyz_frames[i].y.data(), xyz_frames[i].z.data() };

        auto t0 = Clock::now();
        auto encoded = cpu_j2k_simulate(planes, width, height);
        auto t1 = Clock::now();

        double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
        cpu_times.push_back(ms);
        cpu_total_bytes += encoded.size();

        if (i == 0 || (i + 1) % 10 == 0) {
            std::cout << "  Frame " << (i + 1) << "/" << num_frames
                      << " - " << std::fixed << std::setprecision(2) << ms << " ms"
                      << std::endl;
        }
    }
    auto cpu_end = Clock::now();
    double cpu_total_ms = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    double cpu_avg = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / cpu_times.size();

    std::cout << "\n  CPU Results:" << std::endl;
    std::cout << "    Total time:     " << cpu_total_ms << " ms" << std::endl;
    std::cout << "    Avg/frame:      " << cpu_avg << " ms" << std::endl;
    std::cout << "    FPS:            " << (num_frames * 1000.0 / cpu_total_ms) << std::endl;

    /* ===== Comparison ===== */
    double speedup = cpu_total_ms / gpu_total_ms;

    std::cout << "\n================================================" << std::endl;
    std::cout << "              COMPARISON" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "  CPU total:   " << cpu_total_ms << " ms (" << (num_frames * 1000.0 / cpu_total_ms) << " fps)" << std::endl;
    std::cout << "  GPU total:   " << gpu_total_ms << " ms (" << (num_frames * 1000.0 / gpu_total_ms) << " fps)" << std::endl;
    std::cout << "  Speedup:     " << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "  J2K valid:   " << gpu_valid << "/" << num_frames << std::endl;
    std::cout << std::endl;

    bool all_pass = (gpu_valid == num_frames);
    std::cout << "  OVERALL: " << (all_pass ? "ALL J2K FRAMES VALID - PASS" : "SOME FRAMES INVALID - FAIL") << std::endl;
    std::cout << "================================================" << std::endl;

    return all_pass ? 0 : 1;
}
