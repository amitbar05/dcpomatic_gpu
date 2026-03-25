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


struct CudaJ2KEncoderImpl;


class CudaJ2KEncoder
{
public:
	CudaJ2KEncoder();
	~CudaJ2KEncoder();

	CudaJ2KEncoder(CudaJ2KEncoder const&) = delete;
	CudaJ2KEncoder& operator=(CudaJ2KEncoder const&) = delete;

	/** Encode 3-component 12-bit XYZ image to JPEG2000 codestream.
	 *
	 *  @param xyz_planes  Array of 3 pointers to planar int32_t data (12-bit values 0-4095).
	 *                     Component order: X, Y, Z.  Size: width * height per plane.
	 *  @param width       Image width in pixels.
	 *  @param height      Image height in pixels.
	 *  @param bit_rate    Target bit rate in bits/second.
	 *  @param fps         Frame rate (used with bit_rate to compute per-frame budget).
	 *  @param is_3d       True for stereoscopic (halves the per-frame budget).
	 *  @param is_4k       True for 4K resolution.
	 *  @return            Valid JPEG2000 codestream data.
	 */
	std::vector<uint8_t> encode(
		const int32_t* const xyz_planes[3],
		int width,
		int height,
		int64_t bit_rate,
		int fps,
		bool is_3d,
		bool is_4k
	);

	bool is_initialized() const { return _initialized; }

private:
	std::unique_ptr<CudaJ2KEncoderImpl> _impl;
	bool _initialized = false;
	std::mutex _mutex;
};


std::shared_ptr<CudaJ2KEncoder> cuda_j2k_encoder_instance();


#endif
