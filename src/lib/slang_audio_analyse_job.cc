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

#ifdef DCPOMATIC_SLANG

#include "slang_audio_analyse_job.h"
#include "slang_frame_client.h"
#include "audio_buffers.h"
#include "audio_content.h"
#include "audio_processor.h"
#include "config.h"
#include "content.h"
#include "dcpomatic_log.h"
#include "film.h"
#include "player.h"
#include "playlist.h"
#include <fmt/format.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "i18n.h"


using std::const_pointer_cast;
using std::make_shared;
using std::shared_ptr;
using std::string;
using namespace dcpomatic;

/* Accumulate roughly this many seconds of audio per J2KA request, rather than
 * one request per Player callback: a larger, less frequent GPU dispatch (and
 * socket round-trip) amortises the fixed per-dispatch overhead ~3x while
 * keeping the peak exact (a max/sum reduction is partition-independent). */
static int const AUDIO_BATCH_SECONDS = 4;

/* SLANG_AUTO_GAIN_MATH_BEGIN (kept dependency-free and inside these markers:
 * tests/test_slang_audio_gain_math.py slices it out and compiles it standalone,
 * so the case table below is executed rather than merely asserted in prose). */

/** The ABSOLUTE slang auto-gain, in dB, that should be in effect given the
 *  mix's NATURAL peak (i.e. with any previously-applied slang gain already
 *  backed out). This -- not the per-run change -- is the quantity `max_boost`
 *  bounds, mirroring audio_gpu.auto_gain_db, which likewise returns an
 *  absolute, capped gain.
 *
 *  The caller derives the per-run change as `result - prior` and must NOT clamp
 *  it again. An earlier version did, and that second clamp bounded the wrong
 *  quantity: `result - prior` is a boost above natural only when `prior` is
 *  zero; when `prior` is negative it is the CORRECTION of a previously-applied
 *  reduction, and the clamp fired spuriously. The failure it caused: a hot
 *  master (natural +2.5 dBFS) gets prior = -6.0; the operator then trims their
 *  own content gain by -12 dB (a normal edit -- mix_digest folds content gain
 *  in, so the idempotency cache correctly invalidates and the job re-runs).
 *  Effective natural is now -9.5 dBFS, so the absolute gain wanted is
 *  -3.5 - (-9.5) = +6.0 -- exactly AT and within the legal cap. But the per-run
 *  change is 6.0 - (-6.0) = 12.0, which the old clamp cut to 6.0: only half the
 *  correction was applied, the soundtrack shipped at -9.5 dBFS instead of -3.5,
 *  the shortfall was persisted in _slang_gain_abs_db AND the digest was stored,
 *  so every re-run hit the cache and the mix stayed 6 dB quiet permanently.
 *
 *  Case table (target -3.5, cap +6; N = natural peak, D = this function's
 *  result, applied = D - prior, stored absolute = D):
 *    prior  0.0, N -3.5 (first run, already at target) -> D   0.0, applied   0.0
 *    prior  0.0, N +2.5 (hot master)                   -> D  -6.0, applied  -6.0
 *    prior -6.0, mix unchanged                         -> D  -6.0, applied   0.0 (idempotent)
 *    prior -6.0, operator trims content -12 dB         -> D  +6.0, applied +12.0 (the bug case)
 *    prior -6.0, operator raises content +12 dB        -> D -18.0, applied -12.0 (reduction, uncapped)
 *    prior +6.0, mix unchanged (boost capped short)    -> D  +6.0, applied   0.0 (idempotent)
 *    prior +6.0, content raised until N 0.0            -> D  -3.5, applied  -9.5
 *    prior +3.0, content now much quieter, N -30       -> D  +6.0, applied  +3.0 (capped short)
 *  In every row the absolute slang gain ending in effect is D, which is capped,
 *  so the boost-above-natural bound still holds -- it is now enforced on the
 *  quantity it actually describes.
 *
 *  NB the "changed content set" the removed clamp cited is NOT covered by a
 *  delta clamp, so nothing was lost: when content is added or removed, `prior`
 *  no longer describes the surviving content's baked gain, and a uniform
 *  set_gain() over the new set converges to the same wrong peak with or without
 *  the clamp (the clamp only spread the same error over more runs). Nor can it
 *  be fixed by invalidating `prior` whenever mix_digest changes -- the digest
 *  changes on every legitimate re-run too (that is what makes the job re-run at
 *  all), and dropping `prior` there would break the absolute/idempotent
 *  semantics for the ordinary trim-and-re-run case above. Tracking slang's
 *  contribution as one per-film scalar is the real limitation; it is left
 *  visible rather than papered over. */
static double
slang_auto_gain_absolute_db(double natural_peak_dbfs, double target_dbfs, double max_boost_db)
{
	double const want = target_dbfs - natural_peak_dbfs;
	/* Cap the BOOST only -- a reduction (want < 0) is left alone: a mix that is
	 * too hot must always be brought down, however far. */
	return want > max_boost_db ? max_boost_db : want;
}

/* SLANG_AUTO_GAIN_MATH_END */

/* SLANG_PEAK_SOURCE_BEGIN -- kept dependency-free and sliced out for a
 * standalone test the same way as the block just above (see
 * test_slang_audio_gain_math.py, which extracts both). Which peak source
 * `run()` trusts: the server's cumulative J2KA answer, or the local ground-
 * truth accumulator computed unconditionally over every sample regardless of
 * server reachability. `peak_mismatch` must override `used_gpu` even when the
 * server is (as far as `used_gpu`/`gpu_failed` can tell) healthy -- that is
 * the entire point of the cross-check in flush_audio_batch(): a server that
 * ANSWERED is not the same as a server that measured what was sent. */
static double
slang_selected_peak(bool used_gpu, bool gpu_failed, bool peak_mismatch,
		    double server_peak, double local_peak)
{
	return (used_gpu && !gpu_failed && !peak_mismatch) ? server_peak : local_peak;
}

/* A finite peak grossly above full scale (1.0) is upstream corruption -- a
 * decoder fault, an uninitialised/aliased buffer, an int reinterpreted as
 * float -- not a hot master. Auto-gain must REFUSE it: normalising a 1e12 peak
 * to target computes an enormous negative gain that scales the real mix to ~0,
 * shipping a valid-but-silent soundtrack off one garbage sample. This mirrors
 * audio_gpu.SANE_PEAK_MAX on the Python side; the C++ mirror is load-bearing
 * because rejecting the J2KA request there just makes this job fall back to its
 * own _local_peak, which carries the same huge value. Full scale is 1.0, so
 * ~60 dB of headroom is far beyond any real pre-normalise content. */
static double const SLANG_SANE_PEAK_MAX = 1.0e3;

static bool
slang_peak_is_sane(double peak)
{
	return peak > 0.0 && peak <= SLANG_SANE_PEAK_MAX;
}

/* SLANG_PEAK_SOURCE_END */

#if BOOST_VERSION >= 106100
using namespace boost::placeholders;
#endif


SlangAudioAnalyseJob::SlangAudioAnalyseJob(shared_ptr<const Film> film)
	: Job(film)
	, _client(new SlangFrameClient(Config::instance()->slang().socket))
{

}


SlangAudioAnalyseJob::~SlangAudioAnalyseJob()
{
	stop_thread();
}


string
SlangAudioAnalyseJob::name() const
{
	return _("Analysing audio on the GPU");
}


string
SlangAudioAnalyseJob::json_name() const
{
	return N_("slang_analyse_audio");
}


void
SlangAudioAnalyseJob::analyse(shared_ptr<const AudioBuffers> b, DCPTime time)
{
	auto const channels = b->channels();
	auto const frames = b->frames();
	if (channels == 0 || frames == 0) {
		return;
	}

	/* Local peak: the fallback when the server is unreachable, AND the
	 * ground-truth cross-check when it isn't -- the wire carries float32 of
	 * the same samples, so the two are expected to agree EXACTLY (mirrors
	 * audio_gpu.py's proven GPU==CPU peak guarantee). flush_audio_batch()
	 * below actually performs that comparison on every batch; this used to be
	 * computed and never checked against anything -- an ACK ("the server
	 * answered") standing in for verification ("the server measured what we
	 * sent"), the same gap the coder-switch incident found in J2KO. */
	for (int c = 0; c < channels; ++c) {
		auto const* d = b->data()[c];
		for (int i = 0; i < frames; ++i) {
			auto const a = std::fabs(d[i]);
			if (a > _local_peak) {
				_local_peak = a;
			}
		}
	}

	if (!_gpu_failed) {
		/* Interleave this callback's planar samples onto the tail of the
		 * batch buffer (frame-major, the J2KA wire layout). */
		auto const base = static_cast<size_t>(_batch_frames) * channels;
		_batch.resize(base + static_cast<size_t>(frames) * channels);
		for (int c = 0; c < channels; ++c) {
			auto const* d = b->data()[c];
			for (int i = 0; i < frames; ++i) {
				_batch[base + static_cast<size_t>(i) * channels + c] = d[i];
			}
		}
		_batch_frames += frames;
		if (_batch_frames >= static_cast<int64_t>(AUDIO_BATCH_SECONDS) * _film->audio_frame_rate()) {
			flush_audio_batch();
		}
	}

	set_progress(time.get() / static_cast<double>(std::max<int64_t>(1, _film->length().get())), false);
}


/** Send whatever audio has accumulated in _batch to the GPU as one J2KA
 *  request. Fewer, larger dispatches than one-per-callback; the peak is exact
 *  regardless of how the stream is partitioned. */
void
SlangAudioAnalyseJob::flush_audio_batch()
{
	if (_gpu_failed || _batch_frames == 0) {
		return;
	}
	auto const channels = static_cast<int>(_batch.size() / _batch_frames);
	SlangFrameClient::AudioStats stats;
	std::vector<uint8_t> err;
	auto const rc = _client->analyze_audio(
		_batch.data(), static_cast<uint32_t>(_batch_frames), channels,
		_film->audio_frame_rate(), _seq++, stats, err);
	if (rc == 0) {
		_server_peak = stats.overall_peak();
		_used_gpu = true;
		/* Ground-truth check: _local_peak was updated over exactly the
		 * samples just flushed (and nothing more -- analyse() updates it and
		 * appends to _batch from the same callback, and this flush runs
		 * synchronously before the next callback can add anything past what
		 * was just sent), so the two are cumulative peaks over the identical
		 * sample set and must match bit-for-bit. A mismatch is only possible
		 * if the server is not actually measuring what this connection sent
		 * it -- exactly the failure an ACK cannot rule out. Sticky: once
		 * caught, stop trusting this run's server peak even if a later batch
		 * happens to agree again. */
		if (_server_peak != _local_peak) {
			LOG_GENERAL(N_("Slang audio analysis: server/local peak MISMATCH "
				       "({} vs {}) -- the GPU path is not measuring the "
				       "samples this connection sent; the server's peak is "
				       "no longer trusted for the rest of this run"),
				    _server_peak, _local_peak);
			_peak_mismatch = true;
		}
	} else {
		LOG_GENERAL(N_("Slang audio analysis: server unavailable ({}); measuring locally"),
			    rc > 0 ? string(reinterpret_cast<char const*>(err.data()), err.size()) : "transport error");
		_gpu_failed = true;
		_used_gpu = false;
	}
	_batch.clear();
	_batch_frames = 0;
}


void
SlangAudioAnalyseJob::run()
{
	bool has_any_audio = false;
	for (auto c: _film->content()) {
		if (c->audio) {
			has_any_audio = true;
			break;
		}
	}

	/* Idempotency short-circuit: if the natural mix is byte-for-byte the same
	 * as when we last normalised it, the gain already applied is still correct
	 * — skip the whole (expensive) audio replay. Guarded on a prior run having
	 * happened (slang_auto_gain_db != 0 OR a stored digest), so the first-ever
	 * analysis always runs. */
	auto const digest = mix_digest();
	if (has_any_audio && !_film->slang_audio_digest().empty()
	    && digest == _film->slang_audio_digest()) {
		_cache_hit = true;
		_gain_applied_db = 0;
		set_progress(1);
		set_state(FINISHED_OK);
		return;
	}

	auto player = make_shared<Player>(_film, _film->playlist(), false);
	player->set_ignore_video();
	player->set_ignore_text();
	player->set_fast();
	player->set_play_referenced();
	player->Audio.connect(bind(&SlangAudioAnalyseJob::analyse, this, _1, _2));

	if (has_any_audio) {
		while (!player->pass()) {}
		flush_audio_batch();                 /* the last partial batch */
	}

	auto const peak = slang_selected_peak(_used_gpu, _gpu_failed, _peak_mismatch,
					      _server_peak, _local_peak);
	if (peak > 0 && !slang_peak_is_sane(peak)) {
		/* Corruption, not a hot master (see slang_peak_is_sane): refuse to
		 * apply auto-gain -- an enormous reduction would silence the whole
		 * film. Leave the content gain untouched and make it LOUD in the log
		 * rather than shipping a silent soundtrack off one bad sample. */
		LOG_GENERAL(N_("Slang audio analysis: peak {} is far above full scale "
			       "-- a grossly-out-of-range sample (source/decode "
			       "corruption); NOT applying auto-gain, which would "
			       "silence the soundtrack. Fix the source and re-run."),
			    peak);
		_peak_dbfs = 20 * std::log10(peak);
		_gain_applied_db = 0;
		set_progress(1);
		set_state(FINISHED_OK);
		return;
	}
	if (peak > 0) {
		/* Absolute (idempotent) apply: back out the contribution slang itself
		 * already baked into the measured mix, then normalise the *natural*
		 * peak to the target and apply only the difference vs what is already
		 * applied. A no-change re-run therefore adjusts by exactly 0 dB. */
		double const prior = _film->slang_auto_gain_db();
		double const measured_dbfs = 20 * std::log10(peak);
		_peak_dbfs = measured_dbfs - prior;                  /* natural peak, for reporting */
		_slang_gain_abs_db = slang_auto_gain_absolute_db(
			_peak_dbfs, TARGET_PEAK_DBFS, MAX_BOOST_DB);
		_gain_applied_db = _slang_gain_abs_db - prior;
		if (_gain_applied_db != 0) {
			for (auto c: _film->content()) {
				if (c->audio) {
					c->audio->set_gain(c->audio->gain() + _gain_applied_db);
				}
			}
		}
		const_pointer_cast<Film>(_film)->set_slang_auto_gain(_slang_gain_abs_db, digest);
	} else {
		_peak_dbfs = -std::numeric_limits<double>::infinity();
	}

	/* Report which source the applied peak actually came from -- not just
	 * whether the server was reachable. "GPU-mismatch->local" makes a caught
	 * cross-check failure visible in the same log line an operator already
	 * checks for used_gpu(), rather than only in the WARNING a few lines up
	 * (which is easy to miss in a busy log and does not by itself say what
	 * the job then did about it). */
	char const* peak_source = !_used_gpu ? "local"
		: _peak_mismatch ? "GPU-mismatch->local" : "GPU";
	LOG_GENERAL(N_("Slang audio analysis: natural peak {} dBFS ({}), gain change {} dB"),
		    _peak_dbfs, peak_source, _gain_applied_db);

	set_progress(1);
	set_state(FINISHED_OK);
}


string
SlangAudioAnalyseJob::mix_digest() const
{
	double const prior = _film->slang_auto_gain_db();
	string key;
	for (auto c: _film->content()) {
		if (c->audio) {
			/* user gain = total gain minus slang's own (uniform) contribution,
			 * so the key is stable across auto-gain re-normalisations. */
			key += c->digest();
			key += fmt::format(":{:.4f}", c->audio->gain() - prior);
			/* Position/trim/fade/mapping all change the resulting mix (hence its
			 * peak) without touching the content digest -- fold them in so a
			 * re-trim/move/re-map invalidates the cache instead of shipping a
			 * stale (possibly clipped) gain. */
			key += fmt::format(":pos={};ts={};te={};fi={};fo={};map={};",
					   c->position().get(),
					   c->trim_start().get(),
					   c->trim_end().get(),
					   c->audio->fade_in().get(),
					   c->audio->fade_out().get(),
					   c->audio->mapping().digest());
		}
	}
	auto proc = _film->audio_processor();
	key += fmt::format("|proc={}|ch={}|rate={}",
			   proc ? proc->id() : string("none"),
			   _film->audio_channels(), _film->audio_frame_rate());
	return key;
}


string
SlangAudioAnalyseJob::status() const
{
	auto s = Job::status();
	if (!finished_ok()) {
		return s;
	}

	if (_cache_hit) {
		s += _("; audio unchanged, gain already normalised");
	} else if (!std::isfinite(_peak_dbfs)) {
		s += _("; mix was silent, no gain applied");
	} else if (_gain_applied_db == 0) {
		s += fmt::format(_("; mix peaked at {:.1f} dB, already at target"), _peak_dbfs);
	} else {
		/* Report the ACTUAL resulting peak, not TARGET_PEAK_DBFS -- a boost
		 * capped by MAX_BOOST_DB may land short of target. The resulting peak
		 * is the natural peak plus the ABSOLUTE slang gain now in effect (not
		 * the per-run change), so it stays correct across re-runs. */
		double const peak_after = _peak_dbfs + _slang_gain_abs_db;
		bool const capped = _gain_applied_db > 0
			&& peak_after < TARGET_PEAK_DBFS - 0.05;
		s += fmt::format(
			_gain_applied_db < 0
				? _("; mix peaked at {:.1f} dB, gain reduced by {:.1f} dB to {:.1f} dB")
				: capped
					? _("; mix peaked at {:.1f} dB, gain increased by {:.1f} dB (boost capped) to {:.1f} dB")
					: _("; mix peaked at {:.1f} dB, gain increased by {:.1f} dB to {:.1f} dB"),
			_peak_dbfs, std::abs(_gain_applied_db), peak_after
			);
	}
	return s;
}

#endif
