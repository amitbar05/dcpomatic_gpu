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

#include "i18n.h"


using std::make_shared;
using std::shared_ptr;


NvjpegJ2KEncoderThread::NvjpegJ2KEncoderThread(J2KEncoder& encoder, shared_ptr<NvjpegEncoder> nvjpeg)
	: J2KSyncEncoderThread(encoder)
	, _nvjpeg(nvjpeg)
{

}


void
NvjpegJ2KEncoderThread::log_thread_start() const
{
	start_of_thread("NvjpegJ2KEncoder");
	LOG_TIMING("start-encoder-thread thread={} server=gpu-nvjpeg", thread_id());
}


shared_ptr<dcp::ArrayData>
NvjpegJ2KEncoderThread::encode(DCPVideo const& frame)
{
	try {
		/* Get the frame as XYZ via OpenJPEGImage */
		auto image = DCPVideo::convert_to_xyz(frame.frame());
		auto size = image->size();

		/* Extract XYZ planar data into interleaved 8-bit RGB for nvJPEG.
		 * We take the upper 4 bits of the 12-bit XYZ data (values 0-4095)
		 * and scale to 8-bit range.
		 */
		int const width = size.width;
		int const height = size.height;
		int const stride = width * 3;
		std::vector<uint8_t> rgb8(static_cast<size_t>(height) * stride);

		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int const pixel_idx = y * width + x;
				int const dst_idx = y * stride + x * 3;
				/* OpenJPEGImage stores 12-bit values (0-4095) in int32_t planes */
				rgb8[dst_idx + 0] = static_cast<uint8_t>(std::min(255, image->data(0)[pixel_idx] >> 4));
				rgb8[dst_idx + 1] = static_cast<uint8_t>(std::min(255, image->data(1)[pixel_idx] >> 4));
				rgb8[dst_idx + 2] = static_cast<uint8_t>(std::min(255, image->data(2)[pixel_idx] >> 4));
			}
		}

		auto encoded = _nvjpeg->encode(rgb8.data(), width, height, stride, 95);
		return make_shared<dcp::ArrayData>(std::move(encoded));
	} catch (std::exception& e) {
		LOG_ERROR(N_("nvJPEG GPU encode failed ({})"), e.what());
	}

	return {};
}
