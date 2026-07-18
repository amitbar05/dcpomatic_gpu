/*
    Copyright (C) 2012-2021 Carl Hetherington <cth@carlh.net>

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


/** @file src/j2k_encoder.cc
 *  @brief J2K encoder class.
 */


#include "config.h"
#include "cross.h"
#include "dcp_video.h"
#include "dcpomatic_log.h"
#include "encode_server_description.h"
#include "encode_server_finder.h"
#include "exceptions.h"
#include "film.h"
#include "cpu_j2k_encoder_thread.h"
#ifdef DCPOMATIC_GROK
#include "grok/context.h"
#include "grok_j2k_encoder_thread.h"
#endif
#ifdef DCPOMATIC_SLANG
#include "slang_j2k_encoder_thread.h"
#endif
#include "remote_j2k_encoder_thread.h"
#include "j2k_encoder.h"
#include "log.h"
#include "player_video.h"
#include "util.h"
#include "writer.h"
#include <libcxml/cxml.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "i18n.h"


using std::cout;
using std::dynamic_pointer_cast;
using std::exception;
using std::list;
using std::make_shared;
using std::shared_ptr;
using std::weak_ptr;
using boost::optional;
using dcp::Data;
using namespace dcpomatic;

#ifdef DCPOMATIC_GROK

namespace grk_plugin {

IMessengerLogger* sLogger = nullptr;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
void setMessengerLogger(grk_plugin::IMessengerLogger* logger)
{
 	delete sLogger;
 	sLogger = logger;
}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
grk_plugin::IMessengerLogger* getMessengerLogger(void)
{
	return sLogger;
}

}

#endif


/** @param film Film that we are encoding.
 *  @param writer Writer that we are using.
 */
J2KEncoder::J2KEncoder(shared_ptr<const Film> film, Writer& writer)
	: VideoEncoder(film, writer)
	, _waker(Waker::Reason::ENCODING)
#ifdef DCPOMATIC_GROK
	, _give_up(false)
#endif
{
#ifdef DCPOMATIC_GROK
	auto grok = Config::instance()->grok();
	_dcpomatic_context = new grk_plugin::DcpomaticContext(film, writer, _history, grok.binary_location);
	if (grok.enable) {
		_context = new grk_plugin::GrokContext(_dcpomatic_context);
	}
#endif
}


J2KEncoder::~J2KEncoder()
{
	_server_found_connection.disconnect();

	/* One of our encoder threads may be waiting on Writer::write() to return, if that method
	 * is blocked with the writer queue full waiting for _full_condition.  In that case, the
	 * attempt to terminate the encoder threads below (in terminate_threads()) will fail because
	 * the encoder thread waiting for ::write() will have interruption disabled.
	 *
	 * To work around that, make the writer into a zombie to unblock any pending write()s and
	 * not block on any future ones.
	 */
	_writer.zombify();

	terminate_threads();

#ifdef DCPOMATIC_GROK
	delete _context;
	delete _dcpomatic_context;
#endif
}


#ifdef DCPOMATIC_SLANG
/* The effective Slang coder for this encode run: "ht" or "mq".
 *
 * A single reel's picture MXF cannot legally mix HTJ2K (Part 15) and
 * Part-1/MQ essence.  Heterogeneous CPU+GPU encoding (DCPOMATIC_SLANG_HETERO)
 * keeps OpenJPEG (Part-1/MQ) CPU encoder threads running alongside the Slang
 * GPU thread(s), so if the configured Slang coder is "ht" the two pools would
 * write incompatible essence into the same track.  In that case force the
 * whole track to "mq" so every thread produces consistent Part-1 essence.
 * This only affects the local value used to build threads / the end() mop-up;
 * it never persists back to Config.
 */
static std::string
slang_effective_coder()
{
	auto const slang = Config::instance()->slang();
	bool const slang_enable = getenv("DCPOMATIC_SLANG") != nullptr || slang.enable;
	bool const slang_hetero = slang_enable && getenv("DCPOMATIC_SLANG_HETERO");
	if (slang_hetero && slang.coder != "mq") {
		return "mq";
	}
	return slang.coder;
}
#endif


void
J2KEncoder::servers_list_changed()
{
	auto config = Config::instance();
#ifdef DCPOMATIC_GROK
	auto const grok_enable = config->grok().enable;
#else
	auto const grok_enable = false;
#endif
#ifdef DCPOMATIC_SLANG
	/* GUI/config switch (Preferences → GPU (Slang)) or the original env
	 * flag — either enables the GPU path. */
	auto const slang_enable = getenv("DCPOMATIC_SLANG") != nullptr || config->slang().enable;
#else
	auto const slang_enable = false;
#endif

	/* Optional heterogeneous CPU+GPU encode (opt-in via DCPOMATIC_SLANG_HETERO):
	 * normally, when the Slang GPU path is on we run GPU feeder threads only
	 * (cpu=0), exactly as before.  When DCPOMATIC_SLANG_HETERO is set, we ALSO
	 * keep a CPU (OpenJPEG) pool running so CPU cores and GPU(s) both drain the
	 * shared frame queue.  NOTE: measured to REGRESS on a fast-CPU / single-slow-
	 * GPU box (the in-order writer stalls behind the ~10x-slower GPU frames), so
	 * it is off by default; useful on CPU-poor or GPU-rich machines.
	 * DCPOMATIC_SLANG_GPU_THREADS sets the Slang feeder-thread count (default:
	 * heterogeneous=2, otherwise the full master pool).  Grok is unchanged. */
	bool const slang_hetero = slang_enable && getenv("DCPOMATIC_SLANG_HETERO");
	int slang_gpu = 0;
	if (slang_enable) {
		char const* g = getenv("DCPOMATIC_SLANG_GPU_THREADS");
		slang_gpu = g ? atoi(g) : (slang_hetero ? 2 : config->master_encoding_threads());
		if (slang_gpu < 1) {
			slang_gpu = 1;
		}
		if (slang_gpu > config->master_encoding_threads()) {
			slang_gpu = config->master_encoding_threads();
		}
	}
	auto const cpu = (grok_enable || slang_enable || config->only_servers_encode())
		? (slang_hetero ? std::max(1, config->master_encoding_threads() - slang_gpu) : 0)
		: config->master_encoding_threads();
	auto const gpu = grok_enable ? config->master_encoding_threads() : slang_gpu;

	LOG_GENERAL("Thread counts from: grok={}, slang={}, hetero={}, only_servers={}, master={}, cpu={}, gpu={}", grok_enable ? "yes" : "no", slang_enable ? "yes" : "no", slang_hetero ? "yes" : "no", config->only_servers_encode() ? "yes" : "no", config->master_encoding_threads(), cpu, gpu);
	remake_threads(cpu, gpu, EncodeServerFinder::instance()->servers());
}


void
J2KEncoder::begin()
{
	_server_found_connection = EncodeServerFinder::instance()->ServersListChanged.connect(
		boost::bind(&J2KEncoder::servers_list_changed, this)
		);
	servers_list_changed();
}


void
J2KEncoder::pause()
{
#ifdef DCPOMATIC_GROK
	if (!Config::instance()->grok().enable) {
		return;
	}
	return;

	/* XXX; the same problem may occur here as in the destructor, perhaps? */

	terminate_threads();

	/* Something might have been thrown during terminate_threads */
	rethrow();

	delete _context;
	_context = nullptr;
#endif
}


void J2KEncoder::resume()
{
#ifdef DCPOMATIC_GROK
	if (!Config::instance()->grok().enable) {
		return;
	}

	_context = new grk_plugin::GrokContext(_dcpomatic_context);
	servers_list_changed();
#endif
}


void
J2KEncoder::end()
{
	boost::mutex::scoped_lock lock(_queue_mutex);

	LOG_GENERAL(N_("Clearing queue of {}"), _queue.size());

	/* Keep waking workers until the queue is empty */
	while (!_queue.empty()) {
		rethrow();
#ifdef DCPOMATIC_SLANG
		/* Timed for the same reason as in encode(): all-threads-dead must
		   surface the stored exception, not deadlock. */
		_full_condition.timed_wait(lock, boost::posix_time::milliseconds(500));
#else
		_full_condition.wait(lock);
#endif
	}
	lock.unlock();

	LOG_GENERAL(N_("Terminating encoder threads"));

	terminate_threads();

	/* Something might have been thrown during terminate_threads */
	rethrow();

	LOG_GENERAL(N_("Mopping up {}"), _queue.size());

#ifdef DCPOMATIC_SLANG
	/* A8: the mop-up loop below encodes any leftover frames locally with
	 * OpenJPEG (Part-1/MQ).  When the Slang GPU path is active with the HT
	 * coder, that would inject Part-1 essence into an HTJ2K reel MXF and
	 * corrupt the track.  The Slang encoder thread has already been terminated
	 * above, so we can no longer re-dispatch these frames to it; fail loudly
	 * rather than ship a mixed-essence MXF.  (If HETERO forced the coder to MQ
	 * (A9), local encoding is correct and the mop-up proceeds normally.  If
	 * Grok is handling the queue, the loop routes frames to it, not to local
	 * encoding, so this does not apply.) */
	{
		bool grok_handling = false;
#ifdef DCPOMATIC_GROK
		grok_handling = Config::instance()->grok().enable;
#endif
		auto const slang_config = Config::instance()->slang();
		bool const slang_enable = getenv("DCPOMATIC_SLANG") != nullptr || slang_config.enable;
		if (!_queue.empty() && !grok_handling && slang_enable && slang_effective_coder() == "ht") {
			try {
				throw EncodeError(fmt::format(
					N_("GPU (Slang) HT encode left {} frame(s) un-encoded at the end of the run; refusing to encode them with OpenJPEG (Part-1) as that would corrupt the HTJ2K reel."),
					_queue.size()
					));
			} catch (...) {
				store_encode_thread_exception();
				throw;
			}
		}
	}
#endif

	/* The following sequence of events can occur in the above code:
	     1. a remote worker takes the last image off the queue
	     2. the loop above terminates
	     3. the remote worker fails to encode the image and puts it back on the queue
	     4. the remote worker is then terminated by terminate_threads

	     So just mop up anything left in the queue here.
	*/
	for (auto & i: _queue) {
#ifdef DCPOMATIC_GROK
		if (Config::instance()->grok().enable) {
			if (!_context->scheduleCompress(i)){
				LOG_GENERAL(N_("[{}] J2KEncoder thread pushes frame {} back onto queue after failure"), thread_id(), i.index());
				// handle error
			}
		} else {
#else
		{
#endif
			LOG_GENERAL(N_("Encode left-over frame {}"), i.index());
			try {
				_writer.write(
					make_shared<dcp::ArrayData>(i.encode_locally()),
					i.index(),
					i.eyes()
					);
				frame_done(i.eyes());
			} catch (std::exception& e) {
				LOG_ERROR(N_("Local encode failed ({})"), e.what());
			}
		}
	}

#ifdef DCPOMATIC_GROK
	delete _context;
	_context = nullptr;
#endif
}


/** Should be called when a frame has been encoded successfully */
void
J2KEncoder::frame_done(Eyes eyes)
{
	if (eyes == Eyes::BOTH || eyes == Eyes::LEFT) {
		_history.event();
	}
}


/** Called to request encoding of the next video frame in the DCP.  This is called in order,
 *  so each time the supplied frame is the one after the previous one.
 *  pv represents one video frame, and could be empty if there is nothing to encode
 *  for this DCP frame.
 *
 *  @param pv PlayerVideo to encode.
 *  @param time Time of \p pv within the DCP.
 */
void
J2KEncoder::encode(shared_ptr<PlayerVideo> pv, DCPTime time)
{
#ifdef DCPOMATIC_GROK
	if (_give_up) {
		throw EncodeError(_("GPU acceleration is enabled but the grok decoder is not working.  Please check your configuration and license, and ensure that you are connected to the internet."));
	}
#endif

	_waker.nudge();

	size_t threads = 0;
	{
		boost::mutex::scoped_lock lm(_threads_mutex);
		threads = _threads.size();
	}

	boost::mutex::scoped_lock queue_lock(_queue_mutex);

	/* Wait until the queue has gone down a bit.  Allow one thing in the queue even
	   when there are no threads.

	   DCPOMATIC_SLANG: the wait must be timed and re-check rethrow() — if every
	   encoder thread has died with a stored exception (e.g. the Slang thread
	   refusing to encode with the wrong coder), nothing will ever pop the queue
	   or signal this condition, and an untimed wait deadlocks the export
	   instead of failing it with the stored error.
	*/
	while (_queue.size() >= (threads * 2) + 1) {
		LOG_TIMING("decoder-sleep queue={} threads={}", _queue.size(), threads);
#ifdef DCPOMATIC_SLANG
		rethrow();
		_full_condition.timed_wait(queue_lock, boost::posix_time::milliseconds(500));
#else
		_full_condition.wait(queue_lock);
#endif
		LOG_TIMING("decoder-wake queue={} threads={}", _queue.size(), threads);
	}

	_writer.rethrow();
	/* Re-throw any exception raised by one of our threads.  If more
	   than one has thrown an exception, only one will be rethrown, I think;
	   but then, if that happens something has gone badly wrong.
	*/
	rethrow();

	auto const position = time.frames_floor(_film->video_frame_rate());

	if (_writer.can_fake_write(position)) {
		/* We can fake-write this frame */
		LOG_DEBUG_ENCODE("Frame @ {} FAKE", to_string(time));
		_writer.fake_write(position, pv->eyes());
		frame_done(pv->eyes());
	} else if (pv->has_j2k() && !_film->reencode_j2k()) {
		LOG_DEBUG_ENCODE("Frame @ {} J2K", to_string(time));
		/* This frame already has J2K data, so just write it */
		_writer.write(pv->j2k(), position, pv->eyes());
		frame_done(pv->eyes());
	} else if (_last_player_video[pv->eyes()] && _writer.can_repeat(position) && pv->same(_last_player_video[pv->eyes()])) {
		LOG_DEBUG_ENCODE("Frame @ {} REPEAT", to_string(time));
		_writer.repeat(position, pv->eyes());
		frame_done(pv->eyes());
	} else {
		LOG_DEBUG_ENCODE("Frame @ {} ENCODE", to_string(time));
		/* Queue this new frame for encoding */
		LOG_TIMING("add-frame-to-queue queue={}", _queue.size());
		auto dcpv = DCPVideo(
				pv,
				position,
				_film->video_frame_rate(),
				_film->video_bit_rate(VideoEncoding::JPEG2000),
				_film->resolution()
				);
		_queue.push_back(dcpv);

		/* The queue might not be empty any more, so notify anything which is
		   waiting on that.
		*/
		_empty_condition.notify_all();
	}

	_last_player_video[pv->eyes()] = pv;
}


void
J2KEncoder::terminate_threads()
{
	boost::mutex::scoped_lock lm(_threads_mutex);
	boost::this_thread::disable_interruption dis;

	for (auto& thread: _threads) {
		thread->stop();
	}

	_threads.clear();
	_ending = true;
}


void
J2KEncoder::remake_threads(int cpu, int gpu, list<EncodeServerDescription> servers)
{
	LOG_GENERAL("Making threads: CPU={}, GPU={}, Remote={}", cpu, gpu, servers.size());
	if ((cpu + gpu + servers.size()) == 0) {
		/* Make at least one thread, even if all else fails.  Maybe we are configured
		 * for "only servers encode" but no servers have been registered yet.
		 */
		++cpu;
	}

	boost::mutex::scoped_lock lm(_threads_mutex);
	if (_ending) {
		return;
	}

	auto remove_threads = [this](int wanted, int current, std::function<bool (shared_ptr<J2KEncoderThread>)> predicate) {
		for (auto i = wanted; i < current; ++i) {
			auto iter = std::find_if(_threads.begin(), _threads.end(), predicate);
			if (iter != _threads.end()) {
				(*iter)->stop();
				_threads.erase(iter);
			}
		}
	};


	/* CPU */

	auto const is_cpu_thread = [](shared_ptr<J2KEncoderThread> thread) {
		return static_cast<bool>(dynamic_pointer_cast<CPUJ2KEncoderThread>(thread));
	};

	auto const current_cpu_threads = std::count_if(_threads.begin(), _threads.end(), is_cpu_thread);

	for (auto i = current_cpu_threads; i < cpu; ++i) {
		auto thread = make_shared<CPUJ2KEncoderThread>(*this);
		thread->start();
		_threads.push_back(thread);
	}

	remove_threads(cpu, current_cpu_threads, is_cpu_thread);

#ifdef DCPOMATIC_GROK
	/* GPU */

	auto const is_grok_thread = [](shared_ptr<J2KEncoderThread> thread) {
		return static_cast<bool>(dynamic_pointer_cast<GrokJ2KEncoderThread>(thread));
	};

	auto const current_gpu_threads = std::count_if(_threads.begin(), _threads.end(), is_grok_thread);

	for (auto i = current_gpu_threads; i < gpu; ++i) {
		auto thread = make_shared<GrokJ2KEncoderThread>(*this, _context);
		thread->start();
		_threads.push_back(thread);
	}

	remove_threads(gpu, current_gpu_threads, is_grok_thread);
#endif

#ifdef DCPOMATIC_SLANG
	/* GPU (Slang/Vulkan, via the external frame server) */

	auto const is_slang_thread = [](shared_ptr<J2KEncoderThread> thread) {
		return static_cast<bool>(dynamic_pointer_cast<SlangJ2KEncoderThread>(thread));
	};

	auto const current_slang_threads = std::count_if(_threads.begin(), _threads.end(), is_slang_thread);

	/* Multi-GPU: DCPOMATIC_SLANG_SOCKET may be a comma-separated list of sockets
	 * (one frame_server process per GPU). Round-robin the Slang feeder threads
	 * across them so the GPUs run in parallel — separate processes dodge the
	 * Python GIL that makes thread-based multi-GPU give zero speedup. */
	auto const slang_config = Config::instance()->slang();
	std::vector<std::string> sockets;
	{
		char const* sp = getenv("DCPOMATIC_SLANG_SOCKET");
		std::string s = sp ? sp : slang_config.socket;
		size_t pos = 0, comma;
		while ((comma = s.find(',', pos)) != std::string::npos) {
			if (comma > pos) sockets.push_back(s.substr(pos, comma - pos));
			pos = comma + 1;
		}
		if (pos < s.size()) sockets.push_back(s.substr(pos));
		if (sockets.empty()) sockets.push_back("/tmp/j2k_frames.sock");
	}
	/* A9: heterogeneous CPU+GPU encoding (DCPOMATIC_SLANG_HETERO) shares one
	 * reel MXF between the OpenJPEG (Part-1/MQ) CPU threads and the Slang GPU
	 * thread(s).  slang_effective_coder() forces MQ in that case so the whole
	 * track is consistent Part-1 essence rather than a mix of HT + Part-1. */
	auto const slang_coder = slang_effective_coder();
	if (slang_coder != slang_config.coder) {
		LOG_WARNING(N_("DCPOMATIC_SLANG_HETERO is set: forcing the Slang coder from '{}' to '{}' so CPU (OpenJPEG/Part-1) and GPU threads produce consistent essence in the shared reel MXF."), slang_config.coder, slang_coder);
	}
	for (auto i = current_slang_threads; i < gpu; ++i) {
		auto thread = make_shared<SlangJ2KEncoderThread>(*this, sockets[i % sockets.size()], slang_coder);
		thread->start();
		_threads.push_back(thread);
	}

	remove_threads(gpu, current_slang_threads, is_slang_thread);
#endif

	/* Remote */

	for (auto const& server: servers) {
		if (!server.current_link_version()) {
			continue;
		}

		auto is_remote_thread = [server](shared_ptr<J2KEncoderThread> thread) {
			auto remote = dynamic_pointer_cast<RemoteJ2KEncoderThread>(thread);
			return remote && remote->server().host_name() == server.host_name();
		};

		auto const current_threads = std::count_if(_threads.begin(), _threads.end(), is_remote_thread);

		auto const wanted_threads = server.threads();

		if (wanted_threads > current_threads) {
			LOG_GENERAL(N_("Adding {} worker threads for remote {}"), wanted_threads - current_threads, server.host_name());
		} else if (wanted_threads < current_threads) {
			LOG_GENERAL(N_("Removing {} worker threads for remote {}"), current_threads - wanted_threads, server.host_name());
		}

		for (auto i = current_threads; i < wanted_threads; ++i) {
			auto thread = make_shared<RemoteJ2KEncoderThread>(*this, server);
			thread->start();
			_threads.push_back(thread);
		}

		remove_threads(wanted_threads, current_threads, is_remote_thread);
	}

	_writer.set_encoder_threads(_threads.size());
}


DCPVideo
J2KEncoder::pop()
{
	boost::mutex::scoped_lock lock(_queue_mutex);
	while (_queue.empty()) {
		_empty_condition.wait(lock);
	}

	LOG_TIMING("encoder-wake thread={} queue={}", thread_id(), _queue.size());

	auto vf = _queue.front();
	_queue.pop_front();

	_full_condition.notify_all();
	return vf;
}


void
J2KEncoder::retry(DCPVideo video)
{
#ifdef DCPOMATIC_GROK
	{
		/* We might be destroying or remaking these threads, and hopefully in that case we'll come back here
		 * to check again; we definitely don't want to block in that case waiting to be allowed to check
		 * _threads.
		 */
		boost::mutex::scoped_lock lock(_threads_mutex, boost::try_to_lock);
		if (lock) {
			auto is_grok_thread_with_errors = [](shared_ptr<const J2KEncoderThread> thread) {
				auto grok = dynamic_pointer_cast<const GrokJ2KEncoderThread>(thread);
				return grok && grok->errors();
			};

			_give_up = std::any_of(_threads.begin(), _threads.end(), is_grok_thread_with_errors);
		}
	}
#endif

	{
		boost::mutex::scoped_lock lock(_queue_mutex);
		_queue.push_front(video);
		_empty_condition.notify_all();
	}
}


void
J2KEncoder::write(shared_ptr<const dcp::Data> data, int index, Eyes eyes)
{
	_writer.write(data, index, eyes);
	frame_done(eyes);
}
