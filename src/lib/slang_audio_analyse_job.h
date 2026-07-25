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
#include <limits>
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
 *  The local peak is computed unconditionally over every sample regardless of
 *  server reachability, so it doubles as a GROUND-TRUTH cross-check on the
 *  server's answer (see analyse()/flush_audio_batch()): the wire carries
 *  float32 of the exact same samples, so the two are expected to agree
 *  EXACTLY, and a disagreement makes the run fall back to the (always-correct)
 *  local peak rather than trust a server that has just been caught not
 *  measuring what was sent — the same "verify, don't just trust the ACK"
 *  discipline the coder-switch path applies to J2KO (see verify_encode_contract).
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
	 *  audio_gpu.MAX_BOOST_DB). UNBOUNDED since 2026-07-25: the requirement
	 *  for this pipeline is that the loudest DCP channel lands just under
	 *  TARGET_PEAK_DBFS, so whatever boost that takes is applied and the
	 *  measured peak always reaches the target.
	 *
	 *  It was 6.0 until then, for a reason that has not gone away and is
	 *  recorded here rather than deleted: peak normalisation has no notion of
	 *  dialogue level or crest factor, so on a mix with real dynamic range a
	 *  large boost raises everything -- including the noise floor -- and can
	 *  push a dub-stage-levelled mix past SMPTE RP 200 / ISO 2969 reference
	 *  level. Reaching the target peak was chosen over that bound
	 *  deliberately. Anything that needs the old behaviour sets this back to a
	 *  finite number: it is the single knob, and mix_digest() folds it in, so
	 *  changing it re-runs the analysis on existing projects instead of
	 *  leaving them on a stale gain. */
	static constexpr double MAX_BOOST_DB = std::numeric_limits<double>::infinity();

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
	 *  server was unreachable and the local fallback measured it). NOTE this
	 *  only means the server ANSWERED -- see peak_verified() for whether its
	 *  answer was actually checked against the same samples measured locally. */
	bool used_gpu() const {
		return _used_gpu;
	}

	/** @return false if the server's peak ever DISAGREED with the local
	 *  cross-check over the run (see analyse()/flush_audio_batch()). The two
	 *  are computed from the same float32 samples and are expected to agree
	 *  EXACTLY (mirrors audio_gpu.py's proven GPU==CPU peak guarantee); a
	 *  mismatch means the server's own dispatch is not trustworthy this run
	 *  (protocol desync, a device fault the server didn't report, a genuine
	 *  miscompute) and run() has already fallen back to the local peak.
	 *  Always true when used_gpu() is false (nothing to disagree with). */
	bool peak_verified() const {
		return !_peak_mismatch;
	}

	/** @return true if the (expensive) audio replay was skipped because the
	 *  mix is unchanged since the last run, so this job neither measured
	 *  anything nor changed any gain.  The film's stored analysis (peak, gain,
	 *  per-channel levels) is the one still in effect. */
	bool cache_hit() const {
		return _cache_hit;
	}

	/** @return per-DCP-channel NATURAL (i.e. with any previously-applied slang
	 *  gain backed out, matching peak_dbfs()) sample peak, linear; empty if
	 *  nothing was measured.  Taken from the GPU's own per-channel reduction
	 *  when the run's server peak was trusted, and from the local ground-truth
	 *  accumulator otherwise -- the same choice, made on the same flags, that
	 *  slang_selected_peak() makes for the overall peak. */
	std::vector<float> channel_peak() const {
		return _channel_peak;
	}

	/** @return per-DCP-channel NATURAL RMS over the whole mix, linear. */
	std::vector<float> channel_rms() const {
		return _channel_rms;
	}

private:
	void analyse(std::shared_ptr<const AudioBuffers> buffers, dcpomatic::DCPTime time);
	void flush_audio_batch();
	/** Stable key of the *natural* mix (content digests + user gains +
	 *  processor + channel count + rate), independent of the auto-gain's own
	 *  contribution — a matching key means nothing relevant changed, so the
	 *  audio replay can be skipped. */
	std::string mix_digest() const;

	/** Pick the per-channel stats source (GPU vs local) and convert to the
	 *  NATURAL, linear per-channel peak/RMS reported by channel_peak()/
	 *  channel_rms(). @param prior slang's own gain already baked into the
	 *  measured mix, in dB. */
	void finish_channel_stats(double prior);

	std::unique_ptr<SlangFrameClient> _client;
	std::vector<float> _batch;           ///< accumulated interleaved samples
	int64_t _batch_frames = 0;           ///< frames buffered in _batch
	double _local_peak = 0;              ///< fallback / cross-check accumulator
	std::vector<double> _local_channel_peak;   ///< per-channel local peak
	std::vector<double> _local_channel_sumsq;  ///< per-channel local sum of squares
	int64_t _local_frames = 0;           ///< frames the local accumulators cover
	double _server_peak = 0;             ///< latest cumulative GPU peak
	std::vector<double> _server_channel_peak;  ///< latest cumulative GPU per-channel peak
	std::vector<double> _server_channel_sumsq; ///< latest cumulative GPU per-channel sum of squares
	uint64_t _server_frames = 0;         ///< frames the GPU accumulators cover
	bool _gpu_failed = false;            ///< sticky local fallback
	bool _peak_mismatch = false;         ///< sticky: server/local disagreed at least once
	uint32_t _seq = 0;
	bool _used_gpu = false;
	bool _cache_hit = false;             ///< skipped the replay (mix unchanged)
	double _gain_applied_db = 0;
	double _peak_dbfs = 0;
	/** Absolute slang auto-gain in effect after this run (== the value
	 *  persisted via Film::set_slang_auto_gain); natural peak + this = the
	 *  true resulting mix peak, reported by status(). */
	double _slang_gain_abs_db = 0;
	/** natural per-channel peak/RMS (linear) reported by channel_peak()/channel_rms() */
	std::vector<float> _channel_peak;
	std::vector<float> _channel_rms;
};


#endif

#endif
