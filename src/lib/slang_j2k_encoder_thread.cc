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


SlangJ2KEncoderThread::SlangJ2KEncoderThread(J2KEncoder& encoder, string socket_path, string coder)
	: J2KSyncEncoderThread(encoder)
	, _client(new SlangFrameClient(std::move(socket_path)))
	, _coder(std::move(coder))
{
	_shm_disabled = getenv("DCPOMATIC_SLANG_NO_SHM") != nullptr;
}


void
SlangJ2KEncoderThread::maybe_send_options(DCPVideo const& frame)
{
	/* Per-connection options ("J2KO"): the configured coder (HT/MQ picker)
	 * and the film's real bitrate/fps, so the server doesn't have to be
	 * started with matching flags. Per-connection state like the colour
	 * tables → resend after any reconnect. A server that structurally
	 * REFUSES the options while a coder is configured cannot honour the
	 * user's explicit HT/MQ choice — fail the job loudly rather than
	 * silently exporting with whatever the server defaults to (a stale
	 * server once acknowledged coder=mq and still produced HT: see
	 * verify_encode_contract, the per-frame ground-truth check).
	 * Transport failure (a pre-J2KO server drops the connection on the
	 * unknown magic) stays non-fatal here: the server's default may still
	 * match the request, and verify_encode_contract arbitrates on the
	 * actual output bytes either way.  Called at the top of encode() AND
	 * immediately before every encode request that may run on a fresh
	 * connection (the in-frame shm/rgb48 fallbacks reconnect internally) —
	 * a reconnect loses the server's per-connection options exactly like it
	 * loses the colour tables, and a frame encoded options-less runs on the
	 * server's DEFAULT coder/bitrate.  Cheap when nothing changed (the
	 * generation check early-returns). */
	if (_options_disabled || !_client->connect()) {
		return;
	}
	if (_options_generation == _client->generation()) {
		return;
	}
	std::vector<uint8_t> err;
	auto const rc = _client->set_options(
		_coder, frame.video_bit_rate() / 1e6, frame.frames_per_second(), err);
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
		if (_coder.empty()) {
			LOG_GENERAL_NC(N_("Slang encoder: server does not speak options (J2KO); using its defaults"));
		} else {
			LOG_ERROR(N_("Slang encoder: server does not speak options (J2KO); cannot request coder '{}' — the first frame will abort the export if the server's default does not match"), _coder);
		}
		_options_disabled = true;
	} else {
		auto const message = string(reinterpret_cast<char const*>(err.data()), err.size());
		if (_coder.empty()) {
			LOG_GENERAL(N_("Slang encoder: server refused options ({}); using its defaults"), message);
			_options_generation = _client->generation();
		} else {
			LOG_ERROR(N_("Slang encoder: server refused options ({}); cannot honour configured coder '{}'"), message, _coder);
			throw std::runtime_error(fmt::format(
				"The GPU frame server refused the '{}' coder request ({}).  "
				"Restart frame_server.py without --workers/--encoder-factory, or start it with "
				"J2K_SERVER_CODER={}, or change the coder in Preferences -> GPU (Slang).",
				_coder, message, _coder));
		}
	}
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
	 * (1) Coder: JPEG 2000 Part 15 (HTJ2K) sets bit 14 of Rsiz
	 *     (SOC | SIZ | Lsiz | Rsiz -> the big-endian uint16 at bytes 6..7).
	 * (2) Bit rate: a DCI frame can never exceed video_bit_rate/8/fps
	 *     bytes; an oversized frame means the server ignored the J2KO
	 *     bitrate (or was started with the wrong flags) and the DCP would
	 *     be rejected downstream anyway — fail on the first frame instead. */
	auto const frame_index = frame.index();
	if (j2c.size() < 8 || j2c[0] != 0xff || j2c[1] != 0x4f || j2c[2] != 0xff || j2c[3] != 0x51) {
		LOG_ERROR(N_("Slang encoder: frame {} is not a JPEG2000 codestream (no SOC/SIZ)"), frame_index);
		throw std::runtime_error("The GPU frame server returned data that is not a JPEG2000 codestream.");
	}
	if (!_coder.empty()) {
		auto const rsiz = static_cast<uint16_t>((j2c[6] << 8) | j2c[7]);
		bool const got_ht = (rsiz & 0x4000) != 0;
		bool const want_ht = _coder == "ht";
		if (got_ht != want_ht) {
			LOG_ERROR(N_("Slang encoder: frame {} Rsiz=0x{:04x} is {} but the configured coder is '{}'"),
				  frame_index, rsiz, got_ht ? "HT" : "MQ", _coder);
			throw std::runtime_error(fmt::format(
				"The GPU frame server is encoding with the {} coder but '{}' is configured "
				"in Preferences -> GPU (Slang).  The server is running with other settings or "
				"stale code — restart frame_server.py, or change the configured coder to match.",
				got_ht ? "HT" : "MQ", _coder));
		}
	}
	auto const fps = frame.frames_per_second();
	if (fps > 0 && frame.video_bit_rate() > 0) {
		auto const max_bytes = static_cast<size_t>(frame.video_bit_rate() / 8.0 / fps) + 64;
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
						if (rc != 0) {
							/* A pre-T2.31 server drops the connection on the
							 * unknown "J2KH" magic (losing the tables), a new
							 * one reports a segment error — either way, retry
							 * this frame as a payload (`dst` still points into
							 * the mapping). Disable shm only if the retry shows
							 * the server alive, so a dead server doesn't cost
							 * the optimization once it comes back. */
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
		if (rc != 0) {
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
		return {};
	}
	if (rc > 0) {
		LOG_ERROR(N_("Slang encode failed for frame {}: {}"), frame.index(),
			  string(reinterpret_cast<char const*>(data.data()), data.size()));
		_backoff = 0;
		return {};
	}

	verify_encode_contract(data, frame);
	_backoff = 0;
	return make_shared<dcp::ArrayData>(data.data(), static_cast<int>(data.size()));
}
