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

#ifndef DCPOMATIC_NVJPEG_ENCODER_H
#define DCPOMATIC_NVJPEG_ENCODER_H

#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <dcp/array_data.h>
#include <dcp/types.h>
#include <memory>
#include <mutex>


/** @class NvjpegEncoder
 *  @brief GPU-accelerated JPEG encoder using NVIDIA nvJPEG.
 *
 *  Provides GPU-accelerated image compression for the DCP export pipeline.
 *  Each thread should have its own encoder state, but the handle can be shared.
 */
class NvjpegEncoder
{
public:
	NvjpegEncoder();
	~NvjpegEncoder();

	NvjpegEncoder(NvjpegEncoder const&) = delete;
	NvjpegEncoder& operator=(NvjpegEncoder const&) = delete;

	/** Encode an RGB image to JPEG on the GPU.
	 *  @param rgb_data Pointer to interleaved RGB pixel data (8-bit per channel).
	 *  @param width Image width in pixels.
	 *  @param height Image height in pixels.
	 *  @param stride Row stride in bytes.
	 *  @param quality JPEG quality (1-100).
	 *  @return Encoded JPEG data.
	 */
	dcp::ArrayData encode(
		const uint8_t* rgb_data,
		int width,
		int height,
		int stride,
		int quality = 95
	);

	/** Encode a 16-bit XYZ/RGB image by downconverting to 8-bit first.
	 *  @param rgb16_data Pointer to interleaved 16-bit RGB pixel data.
	 *  @param width Image width in pixels.
	 *  @param height Image height in pixels.
	 *  @param stride Row stride in bytes.
	 *  @param quality JPEG quality (1-100).
	 *  @return Encoded JPEG data.
	 */
	dcp::ArrayData encode_from_16bit(
		const uint16_t* rgb16_data,
		int width,
		int height,
		int stride,
		int quality = 95
	);

	bool is_initialized() const { return _initialized; }

private:
	void initialize();
	void cleanup();

	nvjpegHandle_t _handle = nullptr;
	nvjpegEncoderState_t _encoder_state = nullptr;
	nvjpegEncoderParams_t _encoder_params = nullptr;
	cudaStream_t _stream = nullptr;
	unsigned char* _d_rgb = nullptr;   ///< Device buffer for RGB image data
	size_t _d_rgb_size = 0;            ///< Current device buffer size
	bool _initialized = false;
	std::mutex _mutex;
};


/** @return A shared singleton NvjpegEncoder instance (thread-safe). */
std::shared_ptr<NvjpegEncoder> nvjpeg_encoder_instance();


#endif
