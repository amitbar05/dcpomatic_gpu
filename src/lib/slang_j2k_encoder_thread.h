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


#ifndef DCPOMATIC_SLANG_J2K_ENCODER_THREAD_H
#define DCPOMATIC_SLANG_J2K_ENCODER_THREAD_H


#include "j2k_sync_encoder_thread.h"
#include "slang_j2k_encoder.h"
#include <dcp/data.h>
#include <memory>


/** @class SlangJ2KEncoderThread
 *  @brief Encoder thread that uses the Slang GPU encoder for JPEG2000 compression.
 *
 *  Uses SlangJ2KEncoder (V17+) to perform DWT and quantization on the GPU.
 *  Supports GPU colour conversion when colour params are available.
 *
 *  Each encoder thread should receive its OWN SlangJ2KEncoder instance
 *  to avoid mutex contention — the V17 encoder is not thread-safe across
 *  concurrent callers, but is safe for exclusive single-thread use. */
class SlangJ2KEncoderThread : public J2KSyncEncoderThread
{
public:
	SlangJ2KEncoderThread(J2KEncoder& encoder, std::shared_ptr<SlangJ2KEncoder> slang_j2k, bool fast_mode = false);

	void log_thread_start() const override;
	std::shared_ptr<dcp::ArrayData> encode(DCPVideo const& frame) override;

private:
	std::shared_ptr<SlangJ2KEncoder> _slang_j2k;
	bool _fast_mode;
};


#endif
