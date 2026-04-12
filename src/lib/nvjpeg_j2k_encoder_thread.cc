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
#include "config.h"
#include "cross.h"
#include "dcpomatic_log.h"
#include "dcp_video.h"
#include "j2k_encoder.h"
#include "player_video.h"
#include "image.h"
#include "util.h"
#include <dcp/j2k_transcode.h>
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


/* V127: Use parent J2KSyncEncoderThread::run() — hybrid GPU+OpenJPEG encoding is synchronous. */

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
		p.lut_out[i] = static_cast<uint16_t>(lut_out.lookup(i / 4095.0));  /* V48: was int32_t */

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
		/* On first frame, extract colour conversion params and upload to GPU. */
		if (!_cuda_j2k->has_colour_params()) {
			auto const& colour_conv = frame.frame()->colour_conversion();
			if (colour_conv) {
				auto params = build_gpu_colour_params(colour_conv.get());
				_cuda_j2k->set_colour_params(params);
			}
		}

		auto const comment = Config::instance()->dcp_j2k_comment();

		if (_cuda_j2k->has_colour_params()) {
			/* V127: Hybrid GPU+OpenJPEG path:
			 *   GPU: RGB48LE → XYZ12 (fast color conversion)
			 *   CPU: XYZ12 → J2K via OpenJPEG (proper EBCOT encoding)
			 * This produces identical quality to the CPU path. */
			auto image = frame.frame()->image(
				[](AVPixelFormat) { return AV_PIX_FMT_RGB48LE; },
				VideoRange::FULL,
				false
			);
			auto size = image->size();
			int rgb_stride = image->stride()[0] / static_cast<int>(sizeof(uint16_t));
			int pixels = size.width * size.height;

			/* GPU color conversion */
			std::vector<int32_t> xyz_buf(3 * pixels);
			bool ok = _cuda_j2k->gpu_rgb_to_xyz(
				reinterpret_cast<const uint16_t*>(image->data()[0]),
				size.width, size.height, rgb_stride,
				xyz_buf.data()
			);
			if (!ok) {
				LOG_ERROR(N_("GPU RGB→XYZ conversion failed, falling back to CPU"));
				goto cpu_fallback;
			}

			/* Build OpenJPEGImage from GPU-computed XYZ planes */
			auto xyz = make_shared<dcp::OpenJPEGImage>(size);
			memcpy(xyz->data(0), xyz_buf.data(),              pixels * sizeof(int32_t));
			memcpy(xyz->data(1), xyz_buf.data() + pixels,     pixels * sizeof(int32_t));
			memcpy(xyz->data(2), xyz_buf.data() + 2 * pixels, pixels * sizeof(int32_t));

			/* OpenJPEG J2K encoding (proper EBCOT — identical quality to CPU path) */
			auto enc = dcp::compress_j2k(
				xyz,
				frame.video_bit_rate(),
				frame.frames_per_second(),
				frame.eyes() == Eyes::LEFT || frame.eyes() == Eyes::RIGHT,
				frame.is_4k(),
				comment.empty() ? "libdcp" : comment
			);

			return make_shared<dcp::ArrayData>(enc);
		}

	cpu_fallback:
		{
			/* Fallback: full CPU path (colour conversion + J2K encoding) */
			auto xyz = DCPVideo::convert_to_xyz(frame.frame());
			auto enc = dcp::compress_j2k(
				xyz,
				frame.video_bit_rate(),
				frame.frames_per_second(),
				frame.eyes() == Eyes::LEFT || frame.eyes() == Eyes::RIGHT,
				frame.is_4k(),
				comment.empty() ? "libdcp" : comment
			);
			return make_shared<dcp::ArrayData>(enc);
		}

	} catch (std::exception& e) {
		LOG_ERROR(N_("CUDA J2K GPU encode failed ({})"), e.what());
	}

	return {};
}
