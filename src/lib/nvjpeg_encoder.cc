/*
    Copyright (C) 2024 DCP-o-matic contributors

    This file is part of DCP-o-matic.

    DCP-o-matic is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    DCP-o-matic is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DCP-o-matic.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "nvjpeg_encoder.h"
#include "dcpomatic_log.h"
#include "exceptions.h"
#include <dcp/array_data.h>
#include <cstring>
#include <algorithm>

#include "i18n.h"


using dcp::ArrayData;


#define NVJPEG_CHECK(call) \
	do { \
		nvjpegStatus_t status = (call); \
		if (status != NVJPEG_STATUS_SUCCESS) { \
			LOG_ERROR("nvJPEG error {} at {}:{}", static_cast<int>(status), __FILE__, __LINE__); \
			throw EncodeError(String::compose("nvJPEG error %1", static_cast<int>(status))); \
		} \
	} while (0)

#define CUDA_CHECK(call) \
	do { \
		cudaError_t err = (call); \
		if (err != cudaSuccess) { \
			LOG_ERROR("CUDA error {} ({}) at {}:{}", static_cast<int>(err), cudaGetErrorString(err), __FILE__, __LINE__); \
			throw EncodeError(String::compose("CUDA error: %1", cudaGetErrorString(err))); \
		} \
	} while (0)


NvjpegEncoder::NvjpegEncoder()
{
	initialize();
}


NvjpegEncoder::~NvjpegEncoder()
{
	cleanup();
}


void
NvjpegEncoder::initialize()
{
	auto status = nvjpegCreateSimple(&_handle);
	if (status != NVJPEG_STATUS_SUCCESS) {
		LOG_ERROR("Failed to create nvJPEG handle: {}", static_cast<int>(status));
		return;
	}

	auto err = cudaStreamCreate(&_stream);
	if (err != cudaSuccess) {
		LOG_ERROR("Failed to create CUDA stream: {}", cudaGetErrorString(err));
		nvjpegDestroy(_handle);
		_handle = nullptr;
		return;
	}

	status = nvjpegEncoderStateCreate(_handle, &_encoder_state, _stream);
	if (status != NVJPEG_STATUS_SUCCESS) {
		LOG_ERROR("Failed to create nvJPEG encoder state: {}", static_cast<int>(status));
		cudaStreamDestroy(_stream);
		nvjpegDestroy(_handle);
		_handle = nullptr;
		_stream = nullptr;
		return;
	}

	status = nvjpegEncoderParamsCreate(_handle, &_encoder_params, _stream);
	if (status != NVJPEG_STATUS_SUCCESS) {
		LOG_ERROR("Failed to create nvJPEG encoder params: {}", static_cast<int>(status));
		nvjpegEncoderStateDestroy(_encoder_state);
		cudaStreamDestroy(_stream);
		nvjpegDestroy(_handle);
		_handle = nullptr;
		_stream = nullptr;
		_encoder_state = nullptr;
		return;
	}

	/* Use 4:4:4 chroma subsampling for highest quality */
	nvjpegEncoderParamsSetSamplingFactors(_encoder_params, NVJPEG_CSS_444, _stream);
	nvjpegEncoderParamsSetOptimizedHuffman(_encoder_params, 1, _stream);

	_initialized = true;
	LOG_GENERAL_NC("nvJPEG GPU encoder initialized successfully");
}


void
NvjpegEncoder::cleanup()
{
	if (_d_rgb) {
		cudaFree(_d_rgb);
		_d_rgb = nullptr;
		_d_rgb_size = 0;
	}
	if (_encoder_params) {
		nvjpegEncoderParamsDestroy(_encoder_params);
		_encoder_params = nullptr;
	}
	if (_encoder_state) {
		nvjpegEncoderStateDestroy(_encoder_state);
		_encoder_state = nullptr;
	}
	if (_stream) {
		cudaStreamDestroy(_stream);
		_stream = nullptr;
	}
	if (_handle) {
		nvjpegDestroy(_handle);
		_handle = nullptr;
	}
	_initialized = false;
}


ArrayData
NvjpegEncoder::encode(
	const uint8_t* rgb_data,
	int width,
	int height,
	int stride,
	int quality
)
{
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw EncodeError(_("nvJPEG encoder is not initialized"));
	}

	NVJPEG_CHECK(nvjpegEncoderParamsSetQuality(_encoder_params, quality, _stream));

	/* Ensure device buffer is large enough */
	size_t required = static_cast<size_t>(height) * stride;
	if (required > _d_rgb_size) {
		if (_d_rgb) {
			cudaFree(_d_rgb);
		}
		CUDA_CHECK(cudaMalloc(&_d_rgb, required));
		_d_rgb_size = required;
	}

	/* Copy RGB data to device */
	CUDA_CHECK(cudaMemcpy2DAsync(
		_d_rgb, stride,
		rgb_data, stride,
		width * 3, height,
		cudaMemcpyHostToDevice, _stream
	));

	/* Set up nvjpegImage_t for interleaved RGB input */
	nvjpegImage_t nv_image;
	memset(&nv_image, 0, sizeof(nv_image));
	nv_image.channel[0] = _d_rgb;
	nv_image.pitch[0] = stride;

	/* Encode */
	NVJPEG_CHECK(nvjpegEncodeImage(
		_handle,
		_encoder_state,
		_encoder_params,
		&nv_image,
		NVJPEG_INPUT_RGBI,
		width,
		height,
		_stream
	));

	/* Retrieve compressed data size */
	size_t compressed_size = 0;
	NVJPEG_CHECK(nvjpegEncodeRetrieveBitstream(
		_handle,
		_encoder_state,
		nullptr,
		&compressed_size,
		_stream
	));

	/* Retrieve compressed data */
	ArrayData result(compressed_size);
	NVJPEG_CHECK(nvjpegEncodeRetrieveBitstream(
		_handle,
		_encoder_state,
		result.data(),
		&compressed_size,
		_stream
	));

	CUDA_CHECK(cudaStreamSynchronize(_stream));

	return result;
}


ArrayData
NvjpegEncoder::encode_from_16bit(
	const uint16_t* rgb16_data,
	int width,
	int height,
	int stride,
	int quality
)
{
	/* Convert 16-bit to 8-bit RGB (take upper 8 bits) */
	int const row_bytes_8bit = width * 3;
	std::vector<uint8_t> rgb8(static_cast<size_t>(height) * row_bytes_8bit);

	for (int y = 0; y < height; ++y) {
		auto src_row = reinterpret_cast<const uint16_t*>(
			reinterpret_cast<const uint8_t*>(rgb16_data) + static_cast<size_t>(y) * stride
		);
		auto dst_row = rgb8.data() + static_cast<size_t>(y) * row_bytes_8bit;
		for (int x = 0; x < width * 3; ++x) {
			dst_row[x] = static_cast<uint8_t>(src_row[x] >> 8);
		}
	}

	return encode(rgb8.data(), width, height, row_bytes_8bit, quality);
}


static std::shared_ptr<NvjpegEncoder> _nvjpeg_instance;
static std::mutex _nvjpeg_instance_mutex;

std::shared_ptr<NvjpegEncoder>
nvjpeg_encoder_instance()
{
	std::lock_guard<std::mutex> lock(_nvjpeg_instance_mutex);
	if (!_nvjpeg_instance) {
		_nvjpeg_instance = std::make_shared<NvjpegEncoder>();
	}
	return _nvjpeg_instance;
}
