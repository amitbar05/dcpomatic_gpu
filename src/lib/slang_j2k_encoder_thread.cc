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


#include "slang_j2k_encoder_thread.h"
#include "cross.h"
#include "dcpomatic_log.h"
#include "dcp_video.h"
#include "j2k_encoder.h"
#include "player_video.h"
#include "image.h"
#include "util.h"
#include <dcp/openjpeg_image.h>
#include <dcp/rgb_xyz.h>
#include <optional>

#include "i18n.h"


using std::make_shared;
using std::shared_ptr;


SlangJ2KEncoderThread::SlangJ2KEncoderThread(J2KEncoder& encoder,
                                               shared_ptr<SlangJ2KEncoder> slang_j2k)
	: J2KSyncEncoderThread(encoder)
	, _slang_j2k(slang_j2k)
{
}


/**
 * V17r: Pipelined run() — overlaps CPU memcpy with GPU compute for ~60% throughput gain.
 *
 * Flow:
 *   encode(frame_N) launches GPU graph for frame_N and returns frame_{N-1}'s codestream.
 *   After the thread loop exits (interruption), flush() collects the last in-flight frame.
 *
 * pending_frame tracks the last submitted frame whose result has NOT yet been written.
 * has_pending is false only before the very first encode() call.
 */
void
SlangJ2KEncoderThread::run()
try
{
	log_thread_start();

	_pending.reset();  /* in-flight frame (no previous on first call) */

	while (true) {
		if (auto wait = backoff()) {
			LOG_ERROR(N_("Encoder thread sleeping (due to backoff) for {}s"), wait);
			boost::this_thread::sleep(boost::posix_time::seconds(wait));
		}

		LOG_TIMING("encoder-sleep thread={}", thread_id());
		auto frame = _encoder.pop();
		LOG_TIMING("encoder-pop thread={} frame={} eyes={}",
		           thread_id(), frame.index(), static_cast<int>(frame.eyes()));

		/* Submit frame_N — returns frame_{N-1}'s codestream (empty on first call). */
		auto encoded = encode(frame);

		if (_pending) {
			/* encoded is the result for *_pending (frame_{N-1}). */
			if (encoded) {
				boost::this_thread::disable_interruption dis;
				_encoder.write(encoded, _pending->index(), _pending->eyes());
			} else {
				boost::this_thread::disable_interruption dis;
				_encoder.retry(*_pending);
			}
		}
		_pending = frame;  /* frame_N is now in-flight */
	}
} catch (boost::thread_interrupted&) {
	/* Drain the pipeline — collect the last in-flight frame. */
	if (_pending) {
		auto last = _slang_j2k->flush();
		if (!last.empty()) {
			auto result = make_shared<dcp::ArrayData>(last.size());
			memcpy(result->data(), last.data(), last.size());
			boost::this_thread::disable_interruption dis;
			_encoder.write(result, _pending->index(), _pending->eyes());
		} else {
			boost::this_thread::disable_interruption dis;
			_encoder.retry(*_pending);
		}
	}
} catch (...) {
	store_current();
}


void
SlangJ2KEncoderThread::log_thread_start() const
{
	start_of_thread("SlangJ2KEncoderThread");
	LOG_TIMING("start-encoder-thread thread={} server=gpu-slang-j2k", thread_id());
}


/**
 * Build GpuColourParams from a libdcp ColourConversion.
 * Identical to the same function in nvjpeg_j2k_encoder_thread.cc.
 * Called once per thread on the first frame (or when colour params change).
 */
static GpuColourParams
build_slang_gpu_colour_params(dcp::ColourConversion const& conv)
{
	GpuColourParams p;

	auto const& lut_in_d = conv.in()->double_lut(0, 1, 12, false);
	for (int i = 0; i < 4096; ++i)
		p.lut_in[i] = static_cast<float>(lut_in_d[i]);

	auto lut_out = dcp::make_inverse_gamma_lut(conv.out());
	for (int i = 0; i < 4096; ++i)
		p.lut_out[i] = static_cast<uint16_t>(lut_out.lookup(i / 4095.0));  /* V48: was int32_t */

	double mat[9];
	dcp::combined_rgb_to_xyz(conv, mat);
	for (int i = 0; i < 9; ++i)
		p.matrix[i] = static_cast<float>(mat[i]);

	p.valid = true;
	return p;
}


shared_ptr<dcp::ArrayData>
SlangJ2KEncoderThread::encode(DCPVideo const& frame)
{
	try {
		/* On first frame, extract colour conversion params and upload to GPU. */
		if (!_slang_j2k->has_colour_params()) {
			auto const& colour_conv = frame.frame()->colour_conversion();
			if (colour_conv) {
				auto params = build_slang_gpu_colour_params(colour_conv.get());
				_slang_j2k->set_colour_params(params);
			}
		}

		std::vector<uint8_t> encoded;

		if (_slang_j2k->has_colour_params()) {
			/* V17 path: get raw RGB48LE and do colour conversion on GPU */
			auto image = frame.frame()->image(
				[](AVPixelFormat) { return AV_PIX_FMT_RGB48LE; },
				VideoRange::FULL,
				false
			);
			auto size = image->size();
			int rgb_stride = image->stride()[0] / static_cast<int>(sizeof(uint16_t));

			encoded = _slang_j2k->encode_from_rgb48(
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
			/* Fallback: CPU colour conversion */
			auto xyz = DCPVideo::convert_to_xyz(frame.frame());
			auto size = xyz->size();
			const int32_t* planes[3] = {
				xyz->data(0),
				xyz->data(1),
				xyz->data(2)
			};
			encoded = _slang_j2k->encode(
				planes,
				size.width,
				size.height,
				frame.video_bit_rate(),
				frame.frames_per_second(),
				frame.eyes() == Eyes::LEFT || frame.eyes() == Eyes::RIGHT,
				frame.is_4k()
			);
		}

		/* V17r: empty on first pipelined call is expected (no previous frame ready).
		 * Genuine failures also return {} — run() handles both via has_pending. */
		if (encoded.empty()) return {};

		auto result = make_shared<dcp::ArrayData>(encoded.size());
		memcpy(result->data(), encoded.data(), encoded.size());
		return result;

	} catch (std::exception& e) {
		LOG_ERROR(N_("Slang J2K GPU encode failed ({})"), e.what());
	}

	return {};
}
