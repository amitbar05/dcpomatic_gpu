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
#include "config.h"
#include "content.h"
#include "dcpomatic_log.h"
#include "film.h"
#include "player.h"
#include "playlist.h"
#include <algorithm>
#include <cmath>
#include <limits>

#include "i18n.h"


using std::make_shared;
using std::shared_ptr;
using std::string;
using namespace dcpomatic;
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
		_interleave.resize(static_cast<size_t>(frames) * channels);
		for (int c = 0; c < channels; ++c) {
			auto const* d = b->data()[c];
			for (int i = 0; i < frames; ++i) {
				_interleave[static_cast<size_t>(i) * channels + c] = d[i];
			}
		}
		SlangFrameClient::AudioStats stats;
		std::vector<uint8_t> err;
		auto const rc = _client->analyze_audio(
			_interleave.data(), frames, channels,
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
	}

	set_progress(time.get() / static_cast<double>(std::max<int64_t>(1, _film->length().get())), false);
}


void
SlangAudioAnalyseJob::run()
{
	auto player = make_shared<Player>(_film, _film->playlist(), false);
	player->set_ignore_video();
	player->set_ignore_text();
	player->set_fast();
	player->set_play_referenced();
	player->Audio.connect(bind(&SlangAudioAnalyseJob::analyse, this, _1, _2));

	bool has_any_audio = false;
	for (auto c: _film->content()) {
		if (c->audio) {
			has_any_audio = true;
		}
	}

	if (has_any_audio) {
		while (!player->pass()) {}
	}

	auto const peak = _used_gpu && !_gpu_failed ? _server_peak : _local_peak;
	if (peak > 0) {
		_peak_dbfs = 20 * std::log10(peak);
		if (_peak_dbfs > TARGET_PEAK_DBFS) {
			_gain_applied_db = TARGET_PEAK_DBFS - _peak_dbfs;
			for (auto c: _film->content()) {
				if (c->audio) {
					c->audio->set_gain(c->audio->gain() + _gain_applied_db);
				}
			}
		}
	} else {
		_peak_dbfs = -std::numeric_limits<double>::infinity();
	}

	LOG_GENERAL(N_("Slang audio analysis: mix peak {} dBFS ({}), gain change {} dB"),
		    _peak_dbfs, _used_gpu ? "GPU" : "local", _gain_applied_db);

	set_progress(1);
	set_state(FINISHED_OK);
}

#endif
