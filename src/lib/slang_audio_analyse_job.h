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


/** @file  src/lib/slang_audio_analyse_job.h
 *  @brief SlangAudioAnalyseJob: GPU audio analysis + auto-gain to just
 *  under -3 dBFS, part of the Slang GPU export.
 */

#ifdef DCPOMATIC_SLANG

#ifndef DCPOMATIC_SLANG_AUDIO_ANALYSE_JOB_H
#define DCPOMATIC_SLANG_AUDIO_ANALYSE_JOB_H


#include "dcpomatic_time.h"
#include "job.h"
#include <memory>
#include <vector>


class AudioBuffers;
class SlangFrameClient;


/** @class SlangAudioAnalyseJob
 *  @brief Play the film's final audio mix once (video ignored), measure its
 *  true sample peak — on the GPU via the Slang frame server's J2KA analysis
 *  requests, with a local fallback when the server is unreachable — and then
 *  REDUCE every audio content's gain by the same amount so the mix peak lands
 *  just under -3 dBFS.  A mix already at or below the target is left alone.
 */
class SlangAudioAnalyseJob : public Job
{
public:
	explicit SlangAudioAnalyseJob(std::shared_ptr<const Film> film);
	~SlangAudioAnalyseJob();

	std::string name() const override;
	std::string json_name() const override;
	void run() override;

	/** Auto-gain target: "just under -3 dBFS" (the 0.1 dB margin keeps the
	 *  24-bit-quantised peak strictly below -3.0). */
	static constexpr double TARGET_PEAK_DBFS = -3.1;

	/** @return gain applied to every audio content, in dB (<= 0; 0 = the
	 *  mix was already under the target). Valid after the job finished. */
	double gain_applied_db() const {
		return _gain_applied_db;
	}

	/** @return measured mix peak in dBFS before the gain change. */
	double peak_dbfs() const {
		return _peak_dbfs;
	}

	/** @return true if the peak was measured on the GPU (false = the frame
	 *  server was unreachable and the local fallback measured it). */
	bool used_gpu() const {
		return _used_gpu;
	}

private:
	void analyse(std::shared_ptr<const AudioBuffers> buffers, dcpomatic::DCPTime time);

	std::unique_ptr<SlangFrameClient> _client;
	std::vector<float> _interleave;      ///< reused per-block scratch
	double _local_peak = 0;              ///< fallback / cross-check accumulator
	double _server_peak = 0;             ///< latest cumulative GPU peak
	bool _gpu_failed = false;            ///< sticky local fallback
	uint32_t _seq = 0;
	bool _used_gpu = false;
	double _gain_applied_db = 0;
	double _peak_dbfs = 0;
};


#endif

#endif
