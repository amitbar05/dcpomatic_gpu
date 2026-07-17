/*
    Copyright (C) 2026 the DCP-o-matic Slang GPU integration

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


/** @file  src/lib/slang_bitrate_probe_job.h
 *  @brief SlangBitrateProbeJob: runs slang_equivalent_j2k_bit_rate() on the
 *  JobManager's background thread instead of the UI thread.
 */

#ifdef DCPOMATIC_SLANG

#ifndef DCPOMATIC_SLANG_BITRATE_PROBE_JOB_H
#define DCPOMATIC_SLANG_BITRATE_PROBE_JOB_H


#include "job.h"
#include <boost/optional.hpp>
#include <cstdint>
#include <memory>
#include <string>


/** @class SlangBitrateProbeJob
 *  @brief Probes each FFmpeg content's real bit rate for the GPU export's
 *  match_source_bitrate feature.  The probe opens every source file and
 *  calls avformat_find_stream_info(), which can take several seconds on a
 *  real master (worse on a slow/networked disk) -- too slow to run
 *  synchronously in a menu handler on the UI thread.  This job does only
 *  the read-only probe; the caller still owns flooring/capping/rounding
 *  and the actual Film::set_video_bit_rate() call, since those touch live
 *  Film state and must stay on the UI thread.
 */
class SlangBitrateProbeJob : public Job
{
public:
	explicit SlangBitrateProbeJob(std::shared_ptr<const Film> film);

	std::string name() const override;
	std::string json_name() const override;
	void run() override;

	/** @return the highest J2K-equivalent bit rate (bits/sec) across the
	 *  film's FFmpeg content, or none if nothing could be probed.  Valid
	 *  after the job finishes. */
	boost::optional<int64_t> rate() const {
		return _rate;
	}

private:
	boost::optional<int64_t> _rate;
};


#endif

#endif
