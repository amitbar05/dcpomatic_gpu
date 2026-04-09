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


#include "nvjpeg_j2k_encoder_thread.h"
#include "cross.h"
#include "dcpomatic_log.h"
#include "dcp_video.h"
#include "j2k_encoder.h"
#include "player_video.h"
#include "image.h"
#include "util.h"
#include <dcp/openjpeg_image.h>
#include <dcp/rgb_xyz.h>

#include "i18n.h"


using std::make_shared;
using std::shared_ptr;


NvjpegJ2KEncoderThread::NvjpegJ2KEncoderThread(J2KEncoder& encoder, shared_ptr<CudaJ2KEncoder> cuda_j2k)
	: J2KSyncEncoderThread(encoder)
	, _cuda_j2k(cuda_j2k)
{

}


void
NvjpegJ2KEncoderThread::log_thread_start() const
{
	start_of_thread("CudaJ2KEncoder");
	LOG_TIMING("start-encoder-thread thread={} server=gpu-cuda-j2k", thread_id());
}


/**
 * Build GpuColourParams from a libdcp ColourConversion.
 * Called once per encoder thread on the first frame.
 *
 * - lut_in[4096]:  12-bit index → linear float (input gamma linearisation)
 * - lut_out[4096]: linear float [0,1] → DCP int32 (output gamma companding)
 * - matrix[9]:     Bradford + RGB→XYZ combined matrix (row-major)
 */
static GpuColourParams
build_gpu_colour_params(dcp::ColourConversion const& conv)
{
	GpuColourParams p;

	/* Input LUT: 12-bit index → linear float */
	auto const& lut_in_d = conv.in()->double_lut(0, 1, 12, false);
	for (int i = 0; i < 4096; ++i)
		p.lut_in[i] = static_cast<float>(lut_in_d[i]);

	/* Output LUT: linear [0,1] → DCP int32 via PiecewiseLUT2.
	   Sample the piecewise LUT at uniform float intervals. */
	auto lut_out = dcp::make_inverse_gamma_lut(conv.out());
	for (int i = 0; i < 4096; ++i)
		p.lut_out[i] = lut_out.lookup(i / 4095.0);

	/* Bradford + RGB→XYZ combined matrix (9 doubles → 9 floats) */
	double mat[9];
	dcp::combined_rgb_to_xyz(conv, mat);
	for (int i = 0; i < 9; ++i)
		p.matrix[i] = static_cast<float>(mat[i]);

	p.valid = true;
	return p;
}


shared_ptr<dcp::ArrayData>
NvjpegJ2KEncoderThread::encode(DCPVideo const& frame)
{
	try {
		/* On first frame (or if colour params not yet uploaded), extract
		   colour conversion params and upload LUT+matrix to GPU. */
		if (!_cuda_j2k->has_colour_params()) {
			auto const& colour_conv = frame.frame()->colour_conversion();
			if (colour_conv) {
				auto params = build_gpu_colour_params(colour_conv.get());
				_cuda_j2k->set_colour_params(params);
			}
		}

		std::vector<uint8_t> encoded;

		if (_cuda_j2k->has_colour_params()) {
			/* V18 path: get raw RGB48LE and do colour conversion on GPU */
			auto image = frame.frame()->image(
				[](AVPixelFormat) { return AV_PIX_FMT_RGB48LE; },
				VideoRange::FULL,
				false
			);
			auto size = image->size();
			/* stride()[0] is in bytes; divide by sizeof(uint16_t) to get uint16_t stride */
			int rgb_stride = image->stride()[0] / static_cast<int>(sizeof(uint16_t));

			encoded = _cuda_j2k->encode_from_rgb48(
				reinterpret_cast<const uint16_t*>(image->data()[0]),
				size.width,
				size.height,
				rgb_stride,
				frame.video_bit_rate(),
				frame.frames_per_second(),
				frame.eyes() == Eyes::LEFT || frame.eyes() == Eyes::RIGHT,
				frame.is_4k()
			);
		} else {
			/* Fallback: CPU colour conversion (V17 path) */
			auto xyz = DCPVideo::convert_to_xyz(frame.frame());
			auto size = xyz->size();
			const int32_t* planes[3] = {
				xyz->data(0),
				xyz->data(1),
				xyz->data(2)
			};
			encoded = _cuda_j2k->encode(
				planes,
				size.width,
				size.height,
				frame.video_bit_rate(),
				frame.frames_per_second(),
				frame.eyes() == Eyes::LEFT || frame.eyes() == Eyes::RIGHT,
				frame.is_4k()
			);
		}

		if (encoded.empty()) {
			LOG_ERROR(N_("CUDA J2K encode returned empty data for frame {}"), frame.index());
			return {};
		}

		auto result = make_shared<dcp::ArrayData>(encoded.size());
		memcpy(result->data(), encoded.data(), encoded.size());
		return result;

	} catch (std::exception& e) {
		LOG_ERROR(N_("CUDA J2K GPU encode failed ({})"), e.what());
	}

	return {};
}
