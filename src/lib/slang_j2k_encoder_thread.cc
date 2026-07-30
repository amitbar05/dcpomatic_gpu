/*
    GPU (Slang/Vulkan) JPEG2000 encoder thread for DCP-o-matic — see header.

    The wire protocol lives in slang_frame_client.h (shared, dependency-free, and
    tested standalone against the live frame server). This file is just the
    DCP-o-matic glue: hand each frame to the client, wrap the returned bytes in
    dcp::ArrayData.

    I2 (GPU convert_to_xyz offload): when the frame carries a colour conversion
    and its pre-conversion RGB48 image is available, we ship libdcp's OWN
    conversion tables once per connection ("J2KC") and then send the raw RGB48
    ("J2KG") — the server replays rgb_to_xyz bit-exactly on the GPU, removing
    this thread's dominant per-frame CPU cost (~26 Mpx of scalar LUT/matrix
    work at 4K). Any failure falls back to the classic convert_to_xyz + "J2KF"
    path (and a server that rejects RGB48 — e.g. an older one or the Mojo
    backend — disables the offload for the rest of the run).

    T2.31 (shm frame transport): the frame is written STRAIGHT into a reusable
    /dev/shm segment (no client-side pixel buffer at all) and the socket
    carries only the segment name ("J2KS"/"J2KH"), eliminating the
    ~51.8 MB/frame socket copy on both sides — the binding cost in the
    core-starved regime where the export is host-bound, not GPU-bound. Any shm
    failure re-sends the same frame as a classic socket payload; if that shows
    the server alive (it just doesn't speak shm — e.g. pre-T2.31), shm is
    sticky-disabled for the run. DCPOMATIC_SLANG_NO_SHM=1 forces it off (the
    A/B measurement switch).
*/

/* The whole translation unit is Slang-only.  wscript lists this .cc
   unconditionally (as it does every other source), so without the guard a
   plain upstream build -- one that never defines DCPOMATIC_SLANG -- fails
   here on J2KEncoder members that only exist under the flag.  Every other
   slang_*.cc already brackets itself this way. */
#ifdef DCPOMATIC_SLANG

#include "slang_j2k_encoder_thread.h"
#include "slang_frame_client.h"
#include "cross.h"
#include "dcpomatic_log.h"
#include "dcp_video.h"
#include "j2k_encoder.h"
#include "util.h"
#include <dcp/rgb_xyz.h>
#include <dcp/transfer_function.h>
#include <fmt/format.h>
#include <stdexcept>

#include "i18n.h"


using std::make_shared;
using std::shared_ptr;
using std::string;

/* The PiecewiseLUT2 parameters of libdcp's make_inverse_gamma_lut (rgb_xyz.cc).
 * They only shape the tables we SEND; the server's arithmetic is defined
 * entirely by those tables, so bit-parity with the local CPU path holds as
 * long as these match the libdcp we are built against. */
static double constexpr PIECEWISE_BOUNDARY = 0.062;
static int constexpr LOW_BITS = 16;
static int constexpr HIGH_BITS = 12;
static int constexpr OUT_SCALE = 4095;


SlangJ2KEncoderThread::SlangJ2KEncoderThread(J2KEncoder& encoder, string socket_path)
	: J2KSyncEncoderThread(encoder)
	, _client(new SlangFrameClient(std::move(socket_path)))
{
	_shm_disabled = getenv("DCPOMATIC_SLANG_NO_SHM") != nullptr;
}


void
SlangJ2KEncoderThread::maybe_send_options(DCPVideo const& frame)
{
	/* Per-connection options ("J2KO"): the film's real bitrate and fps, so the
	 * server doesn't have to be started with matching flags.  (It used to carry
	 * the HT/MQ coder too; that picker is gone — MQ is the only coder — but the
	 * request still matters, because the rate the DCP is encoded at is the one
	 * this message delivers.)  Per-connection state like the colour tables →
	 * resend after any reconnect.
	 *
	 * A structural REFUSAL stays FATAL — the decision, made explicit when the
	 * coder was removed on 2026-07-31.  The alternative is exporting a whole DCP
	 * at whatever bit rate the server happened to be started with, silently
	 * ignoring the film's own setting and the source-bit-rate match computed for
	 * it; and only half of that is caught downstream, since
	 * verify_encode_contract can only reject frames that are too BIG.  A server
	 * running UNDER the film's rate would produce a quietly lower-quality DCP
	 * that nothing flags.  Failing here names the cause and the remedy instead.
	 *
	 * Transport failure (a pre-J2KO server drops the connection on the unknown
	 * magic) stays non-fatal: such a server may well have been started with
	 * flags that match the film, and refusing to talk to it gains nothing.
	 *
	 * Called at the top of encode() AND immediately before every encode request
	 * that may run on a fresh connection (the in-frame shm/rgb48 fallbacks
	 * reconnect internally) — a reconnect loses the server's per-connection
	 * options exactly like it loses the colour tables, and a frame encoded
	 * options-less runs at the server's DEFAULT bit rate.  Cheap when nothing
	 * changed (the generation check early-returns). */
	if (_options_disabled || !_client->connect()) {
		return;
	}
	if (_options_generation == _client->generation()) {
		return;
	}
	std::vector<uint8_t> err;
	auto const rc = _client->set_options(
		effective_bit_rate(frame) / 1e6, frame.frames_per_second(), err);
	if (rc == 0) {
		_options_generation = _client->generation();
		_options_transport_failures = 0;
	} else if (rc < 0) {
		/* Transport failure: either a pre-J2KO server (drops the connection
		 * on the unknown magic — permanent) or a transient network blip
		 * (would succeed on the next connection).  Allow one retry on a
		 * fresh generation before going sticky, so one blip does not
		 * permanently downgrade a J2KO-capable server to its defaults. */
		if (++_options_transport_failures < 2) {
			LOG_GENERAL_NC(N_("Slang encoder: options (J2KO) transport failure; will retry once on the next connection"));
			return;
		}
		LOG_ERROR(N_("Slang encoder: server does not speak options (J2KO); this DCP will be encoded at the server's own bit rate, not the film's"));
		_options_disabled = true;
	} else {
		auto const message = string(reinterpret_cast<char const*>(err.data()), err.size());
		LOG_ERROR(N_("Slang encoder: server refused options ({}); cannot set the film's bit rate and frame rate"), message);
		throw std::runtime_error(fmt::format(
			"The GPU frame server refused this film's bit rate and frame rate ({}).  "
			"Restart frame_server.py without --workers/--encoder-factory, or start it with "
			"--bitrate-mbps/--fps matching the film.",
			message));
	}
}


int64_t
SlangJ2KEncoderThread::effective_bit_rate(DCPVideo const& frame) const
{
	/* A 3D film's video_bit_rate() is the TOTAL J2K rate for both eyes, but
	 * each eye is encoded here as its own frame, so the per-frame budget for a
	 * stereo eye is half. This mirrors DCPVideo::encode_locally (dcp_video.cc),
	 * which passes eyes()==LEFT||RIGHT as libdcp compress_j2k's stereo flag.
	 * 2D (Eyes::BOTH) is returned unchanged, so 2D behaviour is byte-identical. */
	auto rate = frame.video_bit_rate();
	if (frame.eyes() == Eyes::LEFT || frame.eyes() == Eyes::RIGHT) {
		rate /= 2;
	}
	return rate;
}


void
SlangJ2KEncoderThread::verify_encode_contract(std::vector<uint8_t> const& j2c, DCPVideo const& frame) const
{
	/* Ground-truth checks that the server encoded what was asked of it —
	 * on the OUTPUT bytes, which cannot lie, rather than on the server's
	 * acknowledgements, which can (a stale long-running frame_server.py
	 * once acked coder=mq without switching and produced a 22k-frame HT
	 * DCP from an explicit MQ preference).
	 *
	 * (1) Codestream family: JPEG 2000 Part 15 (HTJ2K) sets bit 14 of Rsiz
	 *     (SOC | SIZ | Lsiz | Rsiz -> the big-endian uint16 at bytes 6..7).
	 *     A DCP may not carry Part 15 — SMPTE ST 429-4 defines Part 1 essence
	 *     only, deployed cinema servers do not decode it, and third-party
	 *     verifiers reject it — so this is now an UNCONDITIONAL refusal rather
	 *     than a comparison against a configured coder.  That matters: the old
	 *     form only ran when a coder was configured, which made it a check that
	 *     could disarm ITSELF (an empty coder, an env-only run, a future caller
	 *     that forgot to pass one) exactly when a wrongly-configured or stale
	 *     server is the thing it exists to catch.  There is no longer any
	 *     configuration under which an HT frame is acceptable, so there is no
	 *     longer any condition on the check.
	 * (2) Bit rate: a DCI frame can never exceed video_bit_rate/8/fps
	 *     bytes; an oversized frame means the server ignored the J2KO
	 *     bitrate (or was started with the wrong flags) and the DCP would
	 *     be rejected downstream anyway — fail on the first frame instead. */
	auto const frame_index = frame.index();
	if (j2c.size() < 8 || j2c[0] != 0xff || j2c[1] != 0x4f || j2c[2] != 0xff || j2c[3] != 0x51) {
		LOG_ERROR(N_("Slang encoder: frame {} is not a JPEG2000 codestream (no SOC/SIZ)"), frame_index);
		throw std::runtime_error("The GPU frame server returned data that is not a JPEG2000 codestream.");
	}
	auto const rsiz = static_cast<uint16_t>((j2c[6] << 8) | j2c[7]);
	if ((rsiz & 0x4000) != 0) {
		LOG_ERROR(N_("Slang encoder: frame {} Rsiz=0x{:04x} has the extended-capabilities bit set (JPEG 2000 Part 15 / HTJ2K)"),
			  frame_index, rsiz);
		throw std::runtime_error(fmt::format(
			"The GPU frame server returned a JPEG 2000 Part 15 (HTJ2K) codestream for frame {}, "
			"which a DCP may not carry (SMPTE ST 429-4 defines Part 1 essence only).  The server "
			"is running stale code or a build with the HT coder — restart frame_server.py from "
			"the current source.",
			frame_index));
	}
	auto const fps = frame.frames_per_second();
	auto const bit_rate = effective_bit_rate(frame);
	if (fps > 0 && bit_rate > 0) {
		auto const max_bytes = static_cast<size_t>(bit_rate / 8.0 / fps) + 64;
		if (j2c.size() > max_bytes) {
			LOG_ERROR(N_("Slang encoder: frame {} is {} bytes but the film's bit rate allows at most {}"),
				  frame_index, j2c.size(), max_bytes);
			throw std::runtime_error(fmt::format(
				"The GPU frame server returned a {}-byte frame but the film's J2K bandwidth "
				"allows at most {} bytes per frame.  The server is running with a higher bit "
				"rate than the film is configured for — restart frame_server.py.",
				j2c.size(), max_bytes));
		}
	}
}


SlangJ2KEncoderThread::~SlangJ2KEncoderThread() = default;


void
SlangJ2KEncoderThread::log_thread_start() const
{
	start_of_thread("SlangJ2KEncoder");
	LOG_TIMING("start-encoder-thread thread={} server=slang", thread_id());
}


bool
SlangJ2KEncoderThread::maybe_send_tables(ColourConversion const& conversion)
{
	if (!_client->connect()) {
		return false;
	}
	auto const id = conversion.identifier();
	if (_tables_id == id && _tables_generation == _client->generation()) {
		return true;                 /* this connection already has them */
	}

	auto const& lut_in = conversion.in()->double_lut(0, 1, 12, false);
	auto const& lut_low = conversion.out()->int_lut(0, PIECEWISE_BOUNDARY, LOW_BITS, true, OUT_SCALE);
	auto const& lut_high = conversion.out()->int_lut(PIECEWISE_BOUNDARY, 1, HIGH_BITS, true, OUT_SCALE);
	double matrix[9];
	dcp::combined_rgb_to_xyz(conversion, matrix);

	auto const payload = SlangFrameClient::build_colour_tables_payload(
		PIECEWISE_BOUNDARY, matrix, lut_in, lut_low, lut_high);

	std::vector<uint8_t> err;
	if (_client->set_colour_tables(payload.data(), payload.size(), err) != 0) {
		return false;
	}
	_tables_id = id;
	_tables_generation = _client->generation();
	return true;
}


shared_ptr<dcp::ArrayData>
SlangJ2KEncoderThread::encode(DCPVideo const& frame)
try {
	return encode_locked(frame);
} catch (boost::thread_interrupted&) {
	throw;
} catch (...) {
	/* A throw out of here kills this thread for good (the base run() stores
	 * the exception on the THREAD's ExceptionStore, which nothing ever
	 * polls).  Store it on the J2KEncoder too, whose encode()/end() rethrow
	 * it — otherwise an export whose Slang threads all give up (e.g. coder
	 * mismatch) deadlocks on the queue conditions instead of failing with
	 * our message. */
	_encoder.store_encode_thread_exception();
	throw;
}


shared_ptr<dcp::ArrayData>
SlangJ2KEncoderThread::encode_locked(DCPVideo const& frame)
{
	auto const size = frame.get_size();
	auto const H = static_cast<uint32_t>(size.height);
	auto const W = static_cast<uint32_t>(size.width);
	auto const index = static_cast<uint32_t>(frame.index());
	auto const samples = static_cast<size_t>(H) * W * 3;
	auto const bytes = samples * sizeof(uint16_t);

	std::vector<uint8_t> data;
	bool rgb48_transport_failed = false;

	maybe_send_options(frame);

	/* I2: try the RGB48 path (GPU-side convert_to_xyz) first. */
	if (!_rgb48_disabled) {
		auto const conversion = frame.colour_conversion();
		if (conversion) {
			/* T2.31: write the frame straight into the shm segment when we
			 * can — no client-side pixel buffer, no socket payload. */
			uint16_t* dst = _shm_disabled ? nullptr : _client->shm_pixels(bytes);
			bool const use_shm = dst != nullptr;
			if (!use_shm) {
				_rgb.resize(samples);
				dst = _rgb.data();
			}
			if (frame.rgb48(dst)) {
				maybe_send_options(frame);   // tables send below may open a fresh connection
				if (maybe_send_tables(*conversion)) {
					int rc;
					if (use_shm) {
						rc = _client->encode_rgb48_shm(H, W, index, data);
						if (rc < 0) {
							/* rc<0 ONLY: a transport/setup failure (a pre-T2.31
							 * server drops the connection on the unknown "J2KH"
							 * magic, losing the tables; a new one reports a
							 * no-segment/too-small error). rc>0 means the shm
							 * frame WAS delivered and the server returned a
							 * structured per-frame error — that is not an shm
							 * problem, so it must fall through to the rc>0 branch
							 * below WITHOUT sticky-disabling shm. Retry this frame
							 * as a payload (`dst` still points into the mapping);
							 * disable shm only if the retry shows the server
							 * alive, so a dead server doesn't cost the
							 * optimization once it comes back. */
							maybe_send_options(frame);   // the retry runs on a fresh connection
							if (maybe_send_tables(*conversion)) {
								rc = _client->encode_rgb48(H, W, index, dst, data);
							} else {
								rc = -1;
							}
							if (rc >= 0) {
								LOG_GENERAL_NC(N_("Slang encoder: server does not speak shm frames; using socket payloads"));
								_shm_disabled = true;
								_client->drop_shm();
							}
						}
					} else {
						rc = _client->encode_rgb48(H, W, index, dst, data);
					}
					if (rc == 0) {
						verify_encode_contract(data, frame);
						_backoff = 0;
						/* A frame came back, so the socket is alive: clear the
						 * consecutive-transport-failure count exactly as the
						 * classic path below does.  Without this the counter is
						 * only ever reset by a frame that took the CLASSIC
						 * path, while this RGB48/shm path is the normal one for
						 * a real export -- so transport blips accumulated
						 * across the whole run and a long export could abort
						 * with "could not reach the GPU frame server for 30
						 * frames in a row" after 30 non-consecutive failures.
						 * The member's own doc comment already promised "reset
						 * only by a frame that actually came back". */
						_consecutive_transport_failures = 0;
						return make_shared<dcp::ArrayData>(data.data(), static_cast<int>(data.size()));
					}
					if (rc > 0) {
						/* Structured server rejection (backend without
						 * RGB48): stop trying, fall through to XYZ. */
						LOG_GENERAL(N_("Slang encoder: server rejected RGB48 ({}); using convert_to_xyz"),
							    string(reinterpret_cast<char const*>(data.data()), data.size()));
						_rgb48_disabled = true;
					} else {
						rgb48_transport_failed = true;
					}
				} else {
					/* A pre-I2 server drops the connection on the unknown
					 * "J2KC" magic — a transport failure, not a structured
					 * one. Note it; if the XYZ path below then succeeds on
					 * the reconnected socket, the server is alive but
					 * doesn't speak RGB48 → disable the offload. */
					rgb48_transport_failed = true;
				}
			}
		}
	}

	/* Classic path: convert_to_xyz on the CPU — written straight into the shm
	 * segment when available (T2.31), else the local scratch buffer. */
	uint16_t* xdst = _shm_disabled ? nullptr : _client->shm_pixels(bytes);
	bool const xyz_shm = xdst != nullptr;
	if (!xyz_shm) {
		_xyz.resize(samples);
		xdst = _xyz.data();
	}
	frame.convert_to_xyz(xdst);                  // interleaved 12-bit XYZ in uint16

	/* An rgb48/tables/shm failure above may have dropped + re-established the
	 * connection; the server forgot this connection's options with it. */
	maybe_send_options(frame);

	int rc;
	if (xyz_shm) {
		rc = _client->encode_shm(H, W, index, data);
		if (rc < 0) {
			/* rc<0 ONLY (transport/setup): rc>0 is a structured per-frame
			 * server error over a delivered shm frame — let it fall through to
			 * the rc>0 branch below without sticky-disabling shm for the run. */
			maybe_send_options(frame);   // the payload retry runs on a fresh connection
			rc = _client->encode(H, W, index, xdst, data);
			if (rc >= 0) {
				LOG_GENERAL_NC(N_("Slang encoder: server does not speak shm frames; using socket payloads"));
				_shm_disabled = true;
				_client->drop_shm();
			}
		}
	} else {
		rc = _client->encode(H, W, index, xdst, data);
	}
	if (rc == 0 && rgb48_transport_failed) {
		LOG_GENERAL_NC(N_("Slang encoder: server does not speak RGB48; using convert_to_xyz"));
		_rgb48_disabled = true;
	}

	if (rc < 0) {
		LOG_ERROR(N_("Slang encoder: transport error for frame {}"), frame.index());
		_backoff = 1;
		/* Bound this the way the rc>0 branch below already bounds a structured
		 * server error -- and with its OWN counter, because the two failures
		 * differ in shape.  A structured error is about one frame, so it counts
		 * against _last_failed_index; a socket that will not open fails EVERY
		 * index, so that counter would reset on every frame and never trip.
		 *
		 * Without this the export simply never ends: with the frame server not
		 * running, every Slang thread requeues its frame, sleeps a second and
		 * tries again, for ever.  The job sits at 0 %, no dialog appears, and
		 * nothing reaches store_encode_thread_exception() for the producer to
		 * rethrow -- and by then jobs_make_dcp_gpu_continue() has persisted
		 * slang.enable, so every later export does it again.  The audio pre-pass
		 * does not warn either: it falls back to measuring locally and finishes
		 * OK.
		 *
		 * ~30 s of a completely unreachable server is well past any restart or
		 * hiccup worth riding out. */
		++_consecutive_transport_failures;
		if (_consecutive_transport_failures >= 30) {
			throw std::runtime_error(fmt::format(
				"Could not reach the GPU frame server at {} for {} frames in a row.  "
				"Start frame_server.py, or turn the GPU encoder off in "
				"Preferences -> GPU (Slang).",
				_client->path(),
				_consecutive_transport_failures));
		}
		return {};
	}
	if (rc > 0) {
		auto const message = string(reinterpret_cast<char const*>(data.data()), data.size());
		LOG_ERROR(N_("Slang encode failed for frame {}: {}"), frame.index(), message);
		/* A7: a structured server error normally means "retry this frame" —
		 * the base run() loop requeues it. But backoff 0 makes it re-pop
		 * immediately, so a frame that fails DETERMINISTICALLY (e.g. malformed
		 * input the server can never encode) would busy-spin forever. Bound it:
		 * count consecutive failures on the SAME frame index, back off a real
		 * second so we don't hammer the server, and after a threshold give up
		 * loudly. The throw is stored on the J2KEncoder by encode()'s catch
		 * (same mechanism as maybe_send_options/verify_encode_contract). */
		auto const frame_index = frame.index();
		if (frame_index != _last_failed_index) {
			_last_failed_index = frame_index;
			_consecutive_failures = 0;
		}
		++_consecutive_failures;
		_backoff = 1;
		if (_consecutive_failures >= 3) {
			throw std::runtime_error(fmt::format(
				"The GPU frame server failed to encode frame {} on {} consecutive "
				"attempts ({}).  Aborting the export rather than retrying forever.",
				frame_index, _consecutive_failures, message));
		}
		return {};
	}

	verify_encode_contract(data, frame);
	_backoff = 0;
	_consecutive_transport_failures = 0;
	return make_shared<dcp::ArrayData>(data.data(), static_cast<int>(data.size()));
}

#endif
