/*
    DCP Export: CPU vs GPU Encoding Benchmark

    Compares:
    - CPU: Pixel processing + data copy (simulating encode overhead)
    - GPU: nvJPEG JPEG encoding on NVIDIA GPU

    Build:
      g++ -std=c++17 -O2 -DDCPOMATIC_NVJPEG \
        gpu_benchmark_test.cc \
        -I../src -I../src/lib \
        -lnvjpeg -lcudart \
        -o gpu_benchmark

    Usage:
      ./gpu_benchmark [width] [height] [num_frames]
      defaults: 2048 1080 100
*/

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

#include <nvjpeg.h>
#include <cuda_runtime.h>


using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;


/** Generate a fake RGB frame with a gradient + noise pattern */
static void
generate_frame(uint8_t* rgb, int width, int height, int seed)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = (y * width + x) * 3;
			rgb[idx + 0] = static_cast<uint8_t>(((x + seed * 37) * 13) & 0xFF);
			rgb[idx + 1] = static_cast<uint8_t>(((y + seed * 53) * 17) & 0xFF);
			rgb[idx + 2] = static_cast<uint8_t>(((x + y + seed * 71) * 11) & 0xFF);
		}
	}
}


/**
 * CPU "encode": simulates the work of JPEG encoding by performing
 * DCT-like operations (block-wise sum-of-products) and writing output.
 * This approximates the computational cost of CPU JPEG encoding.
 */
static std::vector<uint8_t>
cpu_encode_simulate(const uint8_t* rgb, int width, int height)
{
	/* Simulate block-based encoding like JPEG (8x8 blocks) */
	int const block_size = 8;
	int const blocks_x = (width + block_size - 1) / block_size;
	int const blocks_y = (height + block_size - 1) / block_size;

	/* Output buffer - approximate JPEG size at quality 95 */
	std::vector<uint8_t> output(static_cast<size_t>(width) * height * 3 / 4);

	size_t out_pos = 0;
	for (int by = 0; by < blocks_y; ++by) {
		for (int bx = 0; bx < blocks_x; ++bx) {
			/* Process each 8x8 block - simulate DCT + quantization */
			float block[64 * 3];
			for (int j = 0; j < block_size; ++j) {
				for (int i = 0; i < block_size; ++i) {
					int py = by * block_size + j;
					int px = bx * block_size + i;
					if (py < height && px < width) {
						int src = (py * width + px) * 3;
						int dst = (j * block_size + i) * 3;
						block[dst + 0] = rgb[src + 0] * 0.299f + rgb[src + 1] * 0.587f + rgb[src + 2] * 0.114f;
						block[dst + 1] = rgb[src + 0] * -0.169f + rgb[src + 1] * -0.331f + rgb[src + 2] * 0.500f;
						block[dst + 2] = rgb[src + 0] * 0.500f + rgb[src + 1] * -0.419f + rgb[src + 2] * -0.081f;
					}
				}
			}

			/* Simulate quantization and entropy coding */
			for (int k = 0; k < 64 * 3; ++k) {
				int q = static_cast<int>(block[k]) / 4;
				if (out_pos < output.size()) {
					output[out_pos++] = static_cast<uint8_t>(q & 0xFF);
				}
			}
		}
	}

	output.resize(out_pos);
	return output;
}


/** GPU encode using nvJPEG */
struct GpuEncoder {
	nvjpegHandle_t handle = nullptr;
	nvjpegEncoderState_t state = nullptr;
	nvjpegEncoderParams_t params = nullptr;
	cudaStream_t stream = nullptr;
	unsigned char* d_rgb = nullptr;
	size_t d_rgb_size = 0;

	bool init() {
		if (nvjpegCreateSimple(&handle) != NVJPEG_STATUS_SUCCESS) return false;
		if (cudaStreamCreate(&stream) != cudaSuccess) return false;
		if (nvjpegEncoderStateCreate(handle, &state, stream) != NVJPEG_STATUS_SUCCESS) return false;
		if (nvjpegEncoderParamsCreate(handle, &params, stream) != NVJPEG_STATUS_SUCCESS) return false;
		nvjpegEncoderParamsSetQuality(params, 95, stream);
		nvjpegEncoderParamsSetSamplingFactors(params, NVJPEG_CSS_444, stream);
		nvjpegEncoderParamsSetOptimizedHuffman(params, 1, stream);
		return true;
	}

	std::vector<uint8_t> encode(const uint8_t* rgb, int width, int height) {
		size_t required = static_cast<size_t>(height) * width * 3;
		if (required > d_rgb_size) {
			if (d_rgb) cudaFree(d_rgb);
			cudaMalloc(&d_rgb, required);
			d_rgb_size = required;
		}

		int stride = width * 3;
		cudaMemcpy2DAsync(d_rgb, stride, rgb, stride, width * 3, height, cudaMemcpyHostToDevice, stream);

		nvjpegImage_t nv_image;
		memset(&nv_image, 0, sizeof(nv_image));
		nv_image.channel[0] = d_rgb;
		nv_image.pitch[0] = stride;

		nvjpegEncodeImage(handle, state, params, &nv_image, NVJPEG_INPUT_RGBI, width, height, stream);

		size_t length = 0;
		nvjpegEncodeRetrieveBitstream(handle, state, nullptr, &length, stream);

		std::vector<uint8_t> output(length);
		nvjpegEncodeRetrieveBitstream(handle, state, output.data(), &length, stream);
		cudaStreamSynchronize(stream);

		return output;
	}

	~GpuEncoder() {
		if (d_rgb) cudaFree(d_rgb);
		if (params) nvjpegEncoderParamsDestroy(params);
		if (state) nvjpegEncoderStateDestroy(state);
		if (stream) cudaStreamDestroy(stream);
		if (handle) nvjpegDestroy(handle);
	}
};


int main(int argc, char* argv[])
{
	int width = 2048;
	int height = 1080;
	int num_frames = 100;

	if (argc > 1) width = std::atoi(argv[1]);
	if (argc > 2) height = std::atoi(argv[2]);
	if (argc > 3) num_frames = std::atoi(argv[3]);

	std::cout << "========================================" << std::endl;
	std::cout << "  DCP Export: CPU vs GPU Benchmark" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Resolution:  " << width << "x" << height << std::endl;
	std::cout << "Frames:      " << num_frames << std::endl;

	/* Show GPU info */
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
		std::cout << "GPU:         " << prop.name << std::endl;
	}
	std::cout << std::endl;

	/* Pre-generate frames */
	size_t frame_bytes = static_cast<size_t>(width) * height * 3;
	std::vector<std::vector<uint8_t>> frames(num_frames, std::vector<uint8_t>(frame_bytes));
	std::cout << "Generating " << num_frames << " fake frames (" << (frame_bytes * num_frames / 1024 / 1024) << " MB)..." << std::endl;
	for (int i = 0; i < num_frames; ++i) {
		generate_frame(frames[i].data(), width, height, i);
	}

	/* ===== CPU Benchmark ===== */
	std::cout << "\n--- CPU Encoding (simulated JPEG processing) ---" << std::endl;

	std::vector<double> cpu_times;
	size_t cpu_total_bytes = 0;

	auto cpu_start = Clock::now();
	for (int i = 0; i < num_frames; ++i) {
		auto t0 = Clock::now();
		auto encoded = cpu_encode_simulate(frames[i].data(), width, height);
		auto t1 = Clock::now();

		double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
		cpu_times.push_back(ms);
		cpu_total_bytes += encoded.size();

		if ((i + 1) % 25 == 0 || i == 0) {
			std::cout << "  Frame " << (i + 1) << "/" << num_frames
			          << " - " << ms << " ms" << std::endl;
		}
	}
	auto cpu_end = Clock::now();
	double cpu_total_ms = duration_cast<milliseconds>(cpu_end - cpu_start).count();
	double cpu_avg = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / cpu_times.size();

	std::cout << "\n  CPU Results:" << std::endl;
	std::cout << "    Total time:    " << cpu_total_ms << " ms" << std::endl;
	std::cout << "    Avg per frame: " << cpu_avg << " ms" << std::endl;
	std::cout << "    FPS:           " << (num_frames * 1000.0 / cpu_total_ms) << std::endl;
	std::cout << "    Total output:  " << (cpu_total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

	/* ===== GPU Benchmark (nvJPEG) ===== */
	std::cout << "\n--- GPU Encoding (nvJPEG) ---" << std::endl;

	GpuEncoder gpu;
	if (!gpu.init()) {
		std::cerr << "ERROR: Failed to initialize nvJPEG" << std::endl;
		return 1;
	}

	/* Warm up */
	std::cout << "  Warming up GPU..." << std::endl;
	for (int i = 0; i < 5; ++i) {
		gpu.encode(frames[0].data(), width, height);
	}

	std::vector<double> gpu_times;
	size_t gpu_total_bytes = 0;

	auto gpu_start = Clock::now();
	for (int i = 0; i < num_frames; ++i) {
		auto t0 = Clock::now();
		auto encoded = gpu.encode(frames[i].data(), width, height);
		auto t1 = Clock::now();

		double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
		gpu_times.push_back(ms);
		gpu_total_bytes += encoded.size();

		if ((i + 1) % 25 == 0 || i == 0) {
			std::cout << "  Frame " << (i + 1) << "/" << num_frames
			          << " - " << ms << " ms (" << encoded.size() << " bytes)" << std::endl;
		}
	}
	auto gpu_end = Clock::now();
	double gpu_total_ms = duration_cast<milliseconds>(gpu_end - gpu_start).count();
	double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();

	std::cout << "\n  GPU Results:" << std::endl;
	std::cout << "    Total time:    " << gpu_total_ms << " ms" << std::endl;
	std::cout << "    Avg per frame: " << gpu_avg << " ms" << std::endl;
	std::cout << "    FPS:           " << (num_frames * 1000.0 / gpu_total_ms) << std::endl;
	std::cout << "    Total output:  " << (gpu_total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

	/* ===== Comparison ===== */
	std::cout << "\n========================================" << std::endl;
	std::cout << "             COMPARISON" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "  CPU total:     " << cpu_total_ms << " ms  (" << (num_frames * 1000.0 / cpu_total_ms) << " fps)" << std::endl;
	std::cout << "  GPU total:     " << gpu_total_ms << " ms  (" << (num_frames * 1000.0 / gpu_total_ms) << " fps)" << std::endl;

	double speedup = cpu_total_ms / gpu_total_ms;
	std::cout << "  GPU speedup:   " << speedup << "x" << std::endl;

	if (speedup > 1.0) {
		std::cout << "  --> GPU is " << speedup << "x FASTER than CPU" << std::endl;
	} else {
		std::cout << "  --> CPU is " << (1.0 / speedup) << "x faster than GPU" << std::endl;
	}

	std::cout << "\n  Note: CPU uses simulated JPEG block processing." << std::endl;
	std::cout << "        GPU uses nvJPEG hardware-accelerated JPEG encoding." << std::endl;
	std::cout << "        In production DCP export, CPU uses OpenJPEG (JPEG2000)" << std::endl;
	std::cout << "        which is significantly slower than the simulation above." << std::endl;
	std::cout << "========================================" << std::endl;

	return 0;
}
