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


#ifndef DCPOMATIC_NVJPEG_J2K_ENCODER_THREAD_H
#define DCPOMATIC_NVJPEG_J2K_ENCODER_THREAD_H


#include "j2k_sync_encoder_thread.h"
#include "nvjpeg_encoder.h"
#include <dcp/data.h>
#include <memory>


class DCPVideo;


/** @class NvjpegJ2KEncoderThread
 *  @brief Encoder thread that uses nvJPEG (GPU) for image compression.
 */
class NvjpegJ2KEncoderThread : public J2KSyncEncoderThread
{
public:
	NvjpegJ2KEncoderThread(J2KEncoder& encoder, std::shared_ptr<NvjpegEncoder> nvjpeg);

	void log_thread_start() const override;
	std::shared_ptr<dcp::ArrayData> encode(DCPVideo const& frame) override;

private:
	std::shared_ptr<NvjpegEncoder> _nvjpeg;
};


#endif
