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

	/* Local peak: the fallback when the server is unreachable (and a free
	 * cross-check when it isn't — the wire carries float32 of the same
	 * samples, so the two agree exactly). */
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

	auto const peak = _used_gpu && !_gpu_failed ? _server_peak : _local_peak;
	if (peak > 0) {
		/* Absolute (idempotent) apply: back out the contribution slang itself
		 * already baked into the measured mix, then normalise the *natural*
		 * peak to the target and apply only the difference vs what is already
		 * applied. A no-change re-run therefore adjusts by exactly 0 dB. */
		double const prior = _film->slang_auto_gain_db();
		double const measured_dbfs = 20 * std::log10(peak);
		_peak_dbfs = measured_dbfs - prior;                  /* natural peak, for reporting */
		double new_delta = TARGET_PEAK_DBFS - _peak_dbfs;
		if (new_delta > MAX_BOOST_DB) {
			/* Cap the BOOST only -- a reduction (new_delta < 0) is left alone. */
			new_delta = MAX_BOOST_DB;
		}
		_gain_applied_db = new_delta - prior;
		/* Cap the boost AFTER backing out the prior contribution: on a changed
		 * content set 'prior' no longer reflects the current mix, so the per-run
		 * applied change (new_delta - prior) can exceed MAX_BOOST_DB even though
		 * new_delta itself is capped. Bound the actually-applied POSITIVE boost
		 * too; a reduction (negative) is left alone. */
		if (_gain_applied_db > MAX_BOOST_DB) {
			_gain_applied_db = MAX_BOOST_DB;
		}
		/* Absolute slang gain now in effect = what was already applied plus the
		 * change applied this run; persist THIS (not new_delta) so the stored
		 * value stays coherent with the gain actually baked into the content. */
		_slang_gain_abs_db = prior + _gain_applied_db;
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

	LOG_GENERAL(N_("Slang audio analysis: natural peak {} dBFS ({}), gain change {} dB"),
		    _peak_dbfs, _used_gpu ? "GPU" : "local", _gain_applied_db);

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
