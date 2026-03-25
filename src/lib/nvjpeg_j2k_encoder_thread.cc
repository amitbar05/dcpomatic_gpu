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


shared_ptr<dcp::ArrayData>
NvjpegJ2KEncoderThread::encode(DCPVideo const& frame)
{
	try {
		/* Convert frame to XYZ color space (same as CPU path) */
		auto xyz = DCPVideo::convert_to_xyz(frame.frame());
		auto size = xyz->size();

		/* Pass XYZ planar data to GPU J2K encoder */
		const int32_t* planes[3] = {
			xyz->data(0),
			xyz->data(1),
			xyz->data(2)
		};

		auto encoded = _cuda_j2k->encode(
			planes,
			size.width,
			size.height,
			100000000,  /* 100 Mbit/s default */
			24,
			frame.eyes() == Eyes::LEFT || frame.eyes() == Eyes::RIGHT,
			false       /* not 4K for now */
		);

		if (encoded.empty()) {
			LOG_ERROR(N_("CUDA J2K encode returned empty data for frame {}"), frame.index());
			return {};
		}

		/* Wrap in ArrayData for the writer */
		auto result = make_shared<dcp::ArrayData>(encoded.size());
		memcpy(result->data(), encoded.data(), encoded.size());
		return result;
	} catch (std::exception& e) {
		LOG_ERROR(N_("CUDA J2K GPU encode failed ({})"), e.what());
	}

	return {};
}
