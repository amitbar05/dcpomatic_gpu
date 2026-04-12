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

#ifndef DCPOMATIC_CUDA_J2K_ENCODER_H
#define DCPOMATIC_CUDA_J2K_ENCODER_H

/** @file cuda_j2k_encoder.h
 *  @brief GPU-accelerated JPEG2000 encoder using CUDA.
 *
 *  Implements a JPEG2000 encoder that runs the Discrete Wavelet Transform
 *  (DWT 9/7) and quantization on the GPU via CUDA, then packages the
 *  result as a valid J2K codestream for DCP MXF picture assets.
 *
 *  The output is a standard JPEG2000 codestream with:
 *  - SOC, SIZ, COD, QCD marker segments
 *  - Single tile with SOT/SOD markers
 *  - CDF 9/7 irreversible wavelet transform (5 decomposition levels)
 *  - Scalar quantization
 *  - Raw codeblock data (simplified tier-1)
 *  - EOC marker
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <array>


struct CudaJ2KEncoderImpl;


/** GPU color conversion parameters extracted from libdcp ColourConversion.
 *  Cached and uploaded to GPU once per film (or when conversion changes). */
struct GpuColourParams
{
    float    lut_in[4096];   /**< Input gamma: 12-bit index → linear float [0,1] (host; V55: GPU d_lut_in is __half) */
    uint16_t lut_out[4096];  /**< V48: Output gamma: 12-bit index → DCP value [0,4095] (was int32_t; saves 8KB GPU LUT cache) */
    float    matrix[9];     /**< Combined RGB→XYZ 3×3 matrix (row-major) */
    bool     valid = false;
};


class CudaJ2KEncoder
{
public:
	CudaJ2KEncoder();
	~CudaJ2KEncoder();

	CudaJ2KEncoder(CudaJ2KEncoder const&) = delete;
	CudaJ2KEncoder& operator=(CudaJ2KEncoder const&) = delete;

	/** Encode 3-component 12-bit XYZ image to JPEG2000 codestream. */
	std::vector<uint8_t> encode(
		const int32_t* const xyz_planes[3],
		int width,
		int height,
		int64_t bit_rate,
		int fps,
		bool is_3d,
		bool is_4k
	);

	/** V18: Encode from RGB48LE input with GPU color conversion.
	 *  Eliminates CPU rgb_to_xyz bottleneck by running LUT+matrix on GPU. */
	std::vector<uint8_t> encode_from_rgb48(
		const uint16_t* rgb16,   /**< Interleaved RGB48LE, width*3 uint16 per row */
		int width,
		int height,
		int rgb_stride_pixels,   /**< Row stride in uint16_t units (usually width*3) */
		int64_t bit_rate,
		int fps,
		bool is_3d,
		bool is_4k
	);

	/** V127: GPU-accelerated RGB48→XYZ12 color conversion.
	 *  Returns 3 planes of int32 (12-bit XYZ values 0-4095), compatible with OpenJPEGImage.
	 *  Caller owns the returned buffer (3 * width * height int32_t values, planar). */
	bool gpu_rgb_to_xyz(
		const uint16_t* rgb16,
		int width,
		int height,
		int rgb_stride_pixels,
		int32_t* xyz_out   /**< Pre-allocated: 3 * width * height int32_t */
	);

	/** Upload colour conversion LUT+matrix to GPU constant memory. */
	void set_colour_params(GpuColourParams const& params);

	/** V41: Flush the pipeline — collect the last in-flight frame's codestream.
	 *  Must be called after the final encode_from_rgb48() to avoid losing the last frame. */
	std::vector<uint8_t> flush();

	bool is_initialized() const { return _initialized; }
	bool has_colour_params() const { return _colour_params_valid; }

private:
	std::unique_ptr<CudaJ2KEncoderImpl> _impl;
	bool _initialized = false;
	bool _colour_params_valid = false;
};


std::shared_ptr<CudaJ2KEncoder> cuda_j2k_encoder_instance();


#endif
