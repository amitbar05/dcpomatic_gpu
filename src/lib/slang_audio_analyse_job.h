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
#include <string>
#include <vector>


class AudioBuffers;
class SlangFrameClient;


/** @class SlangAudioAnalyseJob
 *  @brief Play the film's final audio mix once (video ignored), measure its
 *  true sample peak — on the GPU via the Slang frame server's J2KA analysis
 *  requests, with a local fallback when the server is unreachable — and then
 *  NORMALIZE every audio content's gain so the mix peak lands just under
 *  -3.5 dBFS (a quiet mix is boosted up, a loud one turned down).
 *
 *  Runs both at content-add time (Film::maybe_analyse_audio_gain) and at
 *  "Make DCP using GPU". The gain change is ABSOLUTE and idempotent: the job
 *  backs out its own previously-baked contribution (Film::slang_auto_gain_db)
 *  before applying the new correction, so a re-run whose mix is unchanged
 *  applies exactly 0 dB — running it on import AND at export never
 *  accumulates. A matching mix digest short-circuits the (expensive) audio
 *  replay entirely.
 */
class SlangAudioAnalyseJob : public Job
{
public:
	explicit SlangAudioAnalyseJob(std::shared_ptr<const Film> film);
	~SlangAudioAnalyseJob();

	std::string name() const override;
	std::string json_name() const override;
	void run() override;
	/** Appends the measured peak / gain-change summary to the base OK/error
	 *  status once finished, so it shows up inline in the Jobs panel (rather
	 *  than a separate popup) — the same place the user already watches
	 *  export progress. */
	std::string status() const override;

	/** Auto-gain target peak, in dBFS. */
	static constexpr double TARGET_PEAK_DBFS = -3.5;

	/** Cap on how far a quiet mix gets boosted, in dB (mirrors
	 *  audio_gpu.MAX_BOOST_DB). Peak-based normalize has no notion of
	 *  dialogue level/crest factor, so an unbounded boost can push a
	 *  properly dynamic, dub-stage-leveled mix well past SMPTE RP 200 / ISO
	 *  2969 reference level; a mix that would need more than this to reach
	 *  TARGET_PEAK_DBFS is left that far under it instead. */
	static constexpr double MAX_BOOST_DB = 6.0;

	/** @return gain applied to every audio content, in dB (0 = the mix was
	 *  already exactly at the target, or silent). Valid after the job
	 *  finished. */
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
	void flush_audio_batch();
	/** Stable key of the *natural* mix (content digests + user gains +
	 *  processor + channel count + rate), independent of the auto-gain's own
	 *  contribution — a matching key means nothing relevant changed, so the
	 *  audio replay can be skipped. */
	std::string mix_digest() const;

	std::unique_ptr<SlangFrameClient> _client;
	std::vector<float> _interleave;      ///< reused per-block scratch
	std::vector<float> _batch;           ///< accumulated interleaved samples
	int64_t _batch_frames = 0;           ///< frames buffered in _batch
	double _local_peak = 0;              ///< fallback / cross-check accumulator
	double _server_peak = 0;             ///< latest cumulative GPU peak
	bool _gpu_failed = false;            ///< sticky local fallback
	uint32_t _seq = 0;
	bool _used_gpu = false;
	bool _cache_hit = false;             ///< skipped the replay (mix unchanged)
	double _gain_applied_db = 0;
	double _peak_dbfs = 0;
};


#endif

#endif
