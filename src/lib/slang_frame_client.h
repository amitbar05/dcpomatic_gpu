/*
    Header-only C++ client for the GPU JPEG2000 frame server.

    This is the wire-protocol half of the DCP-o-matic integration, kept free of
    any DCP-o-matic / libdcp dependency so it can be (a) reused by
    slang_j2k_encoder_thread.cc and (b) compiled and tested standalone against
    the live Python frame server. It must stay byte-compatible with
    encoder/src/dcp/frame_protocol.py.

      request  : "J2KF" | H:u32 | W:u32 | index:u32 | payload_len:u32 | payload
                 payload = H*W*3 uint16 LE interleaved X'Y'Z'
      response : "J2KR" | status:u32 | length:u64 | data[length]

    I2 (GPU convert_to_xyz offload) adds two request kinds on the same header:

      "J2KC" | 0 | 0 | 0 | payload_len:u32 | payload
                 payload = the colour conversion's own tables, packed as
                 rgb48_gpu.ColourTables.pack() (see build_colour_tables_payload
                 below — built with libdcp's public API, so they ARE the CPU
                 path's tables); set once per connection. Empty-OK response.
      "J2KG" | H | W | index | payload_len:u32 | payload
                 payload = H*W*3 uint16 LE interleaved RGB48 (the frame BEFORE
                 convert_to_xyz); the server replays the conversion on the GPU,
                 byte-identical downstream.

    T2.31 (shared-memory frame transport) adds two more: the frame lives in a
    client-owned POSIX shm segment (both ends share /dev/shm — it's a Unix
    socket) and the wire carries only the segment NAME, eliminating the
    ~51.8 MB/frame socket copy on both sides. Byte-identical output.

      "J2KS" | H | W | index | payload_len:u32 | payload   (XYZ via shm)
      "J2KH" | H | W | index | payload_len:u32 | payload   (RGB48 via shm)
                 payload = the segment name (ASCII, no leading slash). The
                 frame is the first H*W*3 uint16 LE of the segment. The
                 protocol is strictly request→response, so one segment is
                 reused for every frame; it grows by recreating under a NEW
                 name (a server holding the old mapping never sees a resize).
*/

#ifndef SLANG_FRAME_CLIENT_H
#define SLANG_FRAME_CLIENT_H

#include <algorithm>
#include <atomic>
#include <mutex>
#include <random>
#include <string>
#include <vector>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>


class SlangFrameClient
{
public:
	explicit SlangFrameClient(std::string socket_path)
		: _path(std::move(socket_path))
	{
		/* Reap /dev/shm segments leaked by dead clients (see
		 * cleanup_stale_shm_once): a SIGKILL/OOM-kill/segfault runs no
		 * destructor, so each aborted 8-thread 4K export can pin ~414 MB of
		 * tmpfs -- which is charged to RAM -- until reboot. Mirrors the Python
		 * reference client (frame_protocol.ShmSegment.__init__ →
		 * cleanup_stale_shm), and the two name formats agree on the
		 * "j2ks_<pid>_<suffix>" shape so either side reaps either's leak. */
		cleanup_stale_shm_once();
	}

	~SlangFrameClient()
	{
		disconnect();
		drop_shm();
	}

	SlangFrameClient(SlangFrameClient const&) = delete;
	SlangFrameClient& operator=(SlangFrameClient const&) = delete;

	bool connected() const { return _fd >= 0; }

	/** Bumped on every successful (re)connect. Per-connection server state
	 *  (the I2 colour tables) is lost on reconnect, so callers holding such
	 *  state compare generations to know when to re-send it. */
	uint64_t generation() const { return _generation; }

	/** The socket this client talks to.  Reported verbatim when giving up on an
	 *  unreachable server: the thread's own path, not whatever Config says now,
	 *  is the one the user has to go and start something on. */
	std::string const& path() const { return _path; }

	bool connect()
	{
		if (_fd >= 0) {
			return true;
		}
		_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
		if (_fd < 0) {
			return false;
		}
		struct sockaddr_un addr;
		memset(&addr, 0, sizeof(addr));
		addr.sun_family = AF_UNIX;
		strncpy(addr.sun_path, _path.c_str(), sizeof(addr.sun_path) - 1);
		if (::connect(_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
			disconnect();
			return false;
		}
		/* Never block on this socket for ever.
		 *
		 * A server that ACCEPTS a request and then wedges -- a GPU driver hang,
		 * a SIGSTOP on the terminal the README tells you to run it in, a cold
		 * shader compile that never completes -- left recv_exact() parked in a
		 * syscall with nothing to wake it.  That is not merely a slow export:
		 * SlangAudioAnalyseJob flushes its audio batch from a Player callback,
		 * and Job::cancel() interrupts and then JOINS that thread from the UI
		 * thread.  boost::thread::interrupt() cannot break a blocking syscall,
		 * so pressing Cancel -- or simply removing the audio content, which
		 * cancels the job for you -- froze the whole application until SIGKILL.
		 *
		 * Deliberately longer than the server's own 120 s per-frame timeout, so
		 * a legitimately slow frame is never cut off; this bounds only the case
		 * where no answer is coming at all.  Every caller already treats a false
		 * from send_all/recv_exact as a transport failure and disconnects, and
		 * both loops bail on r <= 0, so a timed-out read needs no new handling
		 * -- it becomes the "server unavailable" path that already exists. */
		struct timeval tv;
		tv.tv_sec = 180;
		tv.tv_usec = 0;
		::setsockopt(_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
		::setsockopt(_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
		++_generation;
		return true;
	}

	void disconnect()
	{
		if (_fd >= 0) {
			::close(_fd);
			_fd = -1;
		}
	}

	/** Encode one interleaved-XYZ uint16 frame.
	 *  @return 0 on success (out filled with .j2c), -1 on transport error
	 *          (caller should disconnect + retry), or the server's nonzero
	 *          status (out holds the ASCII error message). */
	int encode(uint32_t H, uint32_t W, uint32_t index,
		   uint16_t const* xyz, std::vector<uint8_t>& out)
	{
		auto const bytes = static_cast<size_t>(H) * W * 3 * sizeof(uint16_t);
		return request("J2KF", H, W, index,
			       reinterpret_cast<uint8_t const*>(xyz), bytes, out);
	}

	/** I2: install the connection's colour-conversion tables (a
	 *  ColourTables.pack() payload — see build_colour_tables_payload).
	 *  Must precede encode_rgb48; re-send after any reconnect.
	 *  Same return convention as encode() (out = error message if any). */
	int set_colour_tables(uint8_t const* payload, size_t payload_bytes,
			      std::vector<uint8_t>& out)
	{
		return request("J2KC", 0, 0, 0, payload, payload_bytes, out);
	}

	/** I2: encode one interleaved-RGB48 uint16 frame; the server runs
	 *  convert_to_xyz on the GPU from the connection's tables.
	 *  Same return convention as encode(). */
	int encode_rgb48(uint32_t H, uint32_t W, uint32_t index,
			 uint16_t const* rgb, std::vector<uint8_t>& out)
	{
		auto const bytes = static_cast<size_t>(H) * W * 3 * sizeof(uint16_t);
		return request("J2KG", H, W, index,
			       reinterpret_cast<uint8_t const*>(rgb), bytes, out);
	}

	/** T2.31: writable pointer into this client's reusable /dev/shm segment
	 *  (created/grown lazily to >= `bytes`; grown under a NEW name so the
	 *  server's cached mapping never sees a resize). The caller writes the
	 *  frame straight into it, then calls encode_shm/encode_rgb48_shm.
	 *  nullptr if shared memory is unavailable — fall back to the payload
	 *  encodes. The pointer stays valid until the next shm_pixels with a
	 *  larger size, drop_shm(), or destruction. */
	uint16_t* shm_pixels(size_t bytes)
	{
		if (_shm_ptr && _shm_size >= bytes) {
			return static_cast<uint16_t*>(_shm_ptr);
		}
		drop_shm();
		/* The name carries RANDOMNESS, not just pid+counter, and O_EXCL failures
		 * are retried. With a purely deterministic "j2ks_<pid>_<counter>" name a
		 * single leaked segment was a permanent, silent transport regression: the
		 * leak survives, Linux recycles the pid, the new process rebuilds the
		 * identical name, shm_open(O_EXCL) returns EEXIST, and -- because the
		 * counter is only bumped on the next call -- EVERY later call collides
		 * with the same name, so the whole export falls back to 52 MB/frame
		 * socket payloads, exactly the cost this transport exists to remove. */
		static std::atomic<uint64_t> counter{0};
		char name[64];
		int fd = -1;
		for (int attempt = 0; attempt < 8 && fd < 0; ++attempt) {
			snprintf(name, sizeof(name), "/j2ks_%d_%llu%08x",
				 static_cast<int>(getpid()),
				 static_cast<unsigned long long>(counter++),
				 shm_name_salt());
			fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
			if (fd < 0 && errno != EEXIST) {
				break;             /* ENOSPC/EACCES etc: retrying won't help */
			}
		}
		if (fd < 0) {
			warn_shm_unavailable("shm_open", errno);
			return nullptr;
		}
		if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
			int const err = errno;     /* before close/shm_unlink clobber it */
			::close(fd);
			shm_unlink(name);
			warn_shm_unavailable("ftruncate", err);
			return nullptr;
		}
		void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		int const map_err = errno;         /* before ::close/shm_unlink */
		::close(fd);
		if (p == MAP_FAILED) {
			shm_unlink(name);
			warn_shm_unavailable("mmap", map_err);
			return nullptr;
		}
		_shm_ptr = p;
		_shm_size = bytes;
		_shm_name = name + 1;              /* wire name has no slash */
		return static_cast<uint16_t*>(p);
	}

	/** The current segment's wire name (no leading slash), empty if there is
	 *  none. The name is randomised per segment, so this is also the only way
	 *  to know which /dev/shm file this client owns. */
	std::string const& shm_name() const { return _shm_name; }

	/** T2.31: encode the XYZ frame currently in the shm segment (written via
	 *  shm_pixels). Same return convention as encode(). */
	int encode_shm(uint32_t H, uint32_t W, uint32_t index, std::vector<uint8_t>& out)
	{
		return shm_request("J2KS", H, W, index, out);
	}

	/** T2.31: encode the RGB48 frame currently in the shm segment. */
	int encode_rgb48_shm(uint32_t H, uint32_t W, uint32_t index, std::vector<uint8_t>& out)
	{
		return shm_request("J2KH", H, W, index, out);
	}

	/** Per-connection options ("J2KO", 2026-07-16): select the Tier-1 block
	 *  coder ("ht" — the server default — or "mq") and/or override the
	 *  server's startup bitrate/fps with the film's real values. Empty
	 *  coder / non-positive numbers are omitted. Re-send after any
	 *  reconnect (per-connection state, like the colour tables).
	 *  Same return convention as encode(); a pre-2026-07-16 server drops
	 *  the connection on the unknown magic (-1), a refusing server answers
	 *  a structured error (>0) and keeps its defaults — treat both as
	 *  non-fatal (log + continue with server defaults). */
	int set_options(std::string const& coder, double bitrate_mbps, int fps,
			std::vector<uint8_t>& out)
	{
		std::string p;
		if (!coder.empty()) {
			p += "coder=" + coder + "\n";
		}
		if (bitrate_mbps > 0) {
			char b[64];
			snprintf(b, sizeof(b), "bitrate_mbps=%.6g\n", bitrate_mbps);
			p += b;
		}
		if (fps > 0) {
			char b[32];
			snprintf(b, sizeof(b), "fps=%d\n", fps);
			p += b;
		}
		return request("J2KO", 0, 0, 0,
			       reinterpret_cast<uint8_t const*>(p.data()), p.size(), out);
	}

	/** Cumulative per-channel audio statistics, as returned by every
	 *  "J2KA" analysis request (peak is a linear |sample| max). */
	struct AudioStats
	{
		uint32_t channels = 0;
		uint64_t frames = 0;
		std::vector<double> peak;
		std::vector<double> sumsq;

		double overall_peak() const
		{
			double m = 0;
			for (auto p: peak) {
				m = std::max(m, p);
			}
			return m;
		}
	};

	/** GPU audio analysis ("J2KA", 2026-07-16): send one interleaved
	 *  float32 block (frames x channels); the server accumulates
	 *  per-channel peak + sum-of-squares across this connection and
	 *  answers with the CUMULATIVE stats. Same return convention as
	 *  encode(); on 0 `stats` is filled, otherwise `err` holds the
	 *  message. */
	int analyze_audio(float const* interleaved, uint32_t frames,
			  uint32_t channels, uint32_t sample_rate, uint32_t seq,
			  AudioStats& stats, std::vector<uint8_t>& err)
	{
		std::vector<uint8_t> out;
		auto const rc = request("J2KA", channels, sample_rate, seq,
					reinterpret_cast<uint8_t const*>(interleaved),
					static_cast<size_t>(frames) * channels * sizeof(float), out);
		if (rc != 0) {
			err = out;
			return rc;
		}
		/* u32 nchan | u64 frames | nchan x (f64 peak | f64 sumsq) */
		if (out.size() < 12) {
			err = out;
			return -1;
		}
		stats.channels = get_u32(out.data());
		stats.frames = get_u64(out.data() + 4);
		if (out.size() < 12 + static_cast<size_t>(stats.channels) * 16) {
			err = out;
			return -1;
		}
		stats.peak.resize(stats.channels);
		stats.sumsq.resize(stats.channels);
		for (uint32_t c = 0; c < stats.channels; ++c) {
			stats.peak[c] = get_f64(out.data() + 12 + c * 16);
			stats.sumsq[c] = get_f64(out.data() + 12 + c * 16 + 8);
		}
		return 0;
	}

	/** Release the shm segment (e.g. after sticky-disabling the shm
	 *  transport, so a whole run doesn't pin ~52 MB of /dev/shm). */
	void drop_shm()
	{
		if (_shm_ptr) {
			munmap(_shm_ptr, _shm_size);
			shm_unlink(("/" + _shm_name).c_str());
			_shm_ptr = nullptr;
			_shm_size = 0;
			_shm_name.clear();
		}
	}

	/** Pack colour tables into the "J2KC" payload (mirrors
	 *  rgb48_gpu.ColourTables.pack(): u32 n_in | u32 n_low | u32 n_high |
	 *  f64 boundary | f64 matrix[9] | f64 lut_in[] | i32 low[] | i32 high[]).
	 *  The caller supplies libdcp's own tables — e.g.
	 *    lut_in  = conversion.in()->double_lut(0, 1, 12, false)
	 *    matrix  = combined_rgb_to_xyz(conversion)
	 *    low/high from PiecewiseLUT2(conversion.out(), 0.062, 16, 12, true, 4095)
	 *  so the server replays exactly the CPU path's arithmetic. */
	static std::vector<uint8_t> build_colour_tables_payload(
		double boundary, double const* matrix9,
		std::vector<double> const& lut_in,
		std::vector<int> const& lut_low, std::vector<int> const& lut_high)
	{
		std::vector<uint8_t> p;
		p.reserve(12 + 8 * (10 + lut_in.size()) +
			  4 * (lut_low.size() + lut_high.size()));
		put_u32(p, static_cast<uint32_t>(lut_in.size()));
		put_u32(p, static_cast<uint32_t>(lut_low.size()));
		put_u32(p, static_cast<uint32_t>(lut_high.size()));
		put_f64(p, boundary);
		for (int i = 0; i < 9; ++i) {
			put_f64(p, matrix9[i]);
		}
		for (auto v: lut_in) {
			put_f64(p, v);
		}
		for (auto v: lut_low) {
			put_u32(p, static_cast<uint32_t>(v));
		}
		for (auto v: lut_high) {
			put_u32(p, static_cast<uint32_t>(v));
		}
		return p;
	}

private:
	/** 32 random bits for the segment name. std::random_device is seeded from
	 *  the OS, so two processes that happen to share a recycled pid (and hence
	 *  the whole deterministic part of the name) still get distinct names. */
	static uint32_t shm_name_salt()
	{
		static std::mt19937 gen(std::random_device{}());
		static std::mutex m;
		std::lock_guard<std::mutex> lock(m);
		return static_cast<uint32_t>(gen());
	}

	/** Say ONCE (per process) that the shm transport is unavailable. Without
	 *  this the fallback to 52 MB-per-frame socket payloads is completely
	 *  silent -- the export just gets slower with no way to tell why. Once,
	 *  not per frame: at 24 fps a per-call message would itself be the
	 *  performance problem. This header stays free of DCP-o-matic's logger
	 *  (it must compile standalone), hence stderr. */
	/*  `err` is the errno CAPTURED AT THE FAILURE SITE, not read here: the
	 *  ftruncate and mmap paths run ::close()/shm_unlink() before warning, and
	 *  POSIX permits even a SUCCESSFUL call to set errno, so reading the global
	 *  here can name an unrelated error. This single line is the only signal an
	 *  operator ever gets that the export silently dropped to 52 MB/frame
	 *  socket payloads — a wrong cause sends them down the wrong path. */
	static void warn_shm_unavailable(char const* what, int err)
	{
		static std::atomic<bool> warned{false};
		if (!warned.exchange(true)) {
			fprintf(stderr,
				"slang: /dev/shm frame transport unavailable (%s: %s); "
				"falling back to socket payloads\n",
				what, strerror(err));
		}
	}

	/** Unlink leaked "j2ks_<pid>_<suffix>" segments whose owning process is
	 *  gone; runs once per process (every client thread constructs one of
	 *  these, and one readdir scan is enough). Port of
	 *  frame_protocol.cleanup_stale_shm. A live owner's segment is always
	 *  kept, so a recycled pid costs at most a wasted segment, never another
	 *  process's live frame buffer. Best-effort throughout: never throws,
	 *  never reports failure -- an un-reapable segment is a leak, not a
	 *  correctness problem. */
	static void cleanup_stale_shm_once()
	{
		static std::once_flag once;
		std::call_once(once, [] {
			DIR* d = opendir("/dev/shm");
			if (!d) {
				return;
			}
			while (auto* e = readdir(d)) {
				std::string const name(e->d_name);
				if (name.compare(0, 5, "j2ks_") != 0) {
					continue;
				}
				/* j2ks_<pid>_<suffix>: pid is between the 1st and 2nd '_'. */
				auto const sep = name.find('_', 5);
				if (sep == std::string::npos || sep == 5) {
					continue;
				}
				char* end = nullptr;
				auto const pid = strtol(name.c_str() + 5, &end, 10);
				if (end != name.c_str() + sep || pid <= 0) {
					continue;
				}
				if (kill(static_cast<pid_t>(pid), 0) == 0 || errno != ESRCH) {
					continue;      /* owner alive, or we can't tell: keep */
				}
				shm_unlink(("/" + name).c_str());
			}
			closedir(d);
		});
	}

	int shm_request(char const magic[4], uint32_t H, uint32_t W, uint32_t index,
			std::vector<uint8_t>& out)
	{
		if (_shm_name.empty() ||
		    _shm_size < static_cast<size_t>(H) * W * 3 * sizeof(uint16_t)) {
			return -1;
		}
		return request(magic, H, W, index,
			       reinterpret_cast<uint8_t const*>(_shm_name.data()),
			       _shm_name.size(), out);
	}

	int request(char const magic[4], uint32_t H, uint32_t W, uint32_t index,
		    uint8_t const* payload, size_t payload_bytes,
		    std::vector<uint8_t>& out)
	{
		if (!connect()) {
			return -1;
		}
		std::vector<uint8_t> hdr;
		hdr.reserve(20);
		hdr.insert(hdr.end(), magic, magic + 4);
		put_u32(hdr, H);
		put_u32(hdr, W);
		put_u32(hdr, index);
		put_u32(hdr, static_cast<uint32_t>(payload_bytes));

		/* uint16 buffers are little-endian on all supported targets → as-is. */
		if (!send_all(hdr.data(), hdr.size()) ||
		    (payload_bytes && !send_all(payload, payload_bytes))) {
			disconnect();
			return -1;
		}

		uint8_t resp[16];
		if (!recv_exact(resp, sizeof(resp)) || memcmp(resp, "J2KR", 4) != 0) {
			disconnect();
			return -1;
		}
		auto const status = get_u32(resp + 4);
		auto const length = get_u64(resp + 8);
		out.resize(length);
		if (length && !recv_exact(out.data(), length)) {
			disconnect();
			return -1;
		}
		return static_cast<int>(status);
	}

	static void put_u32(std::vector<uint8_t>& b, uint32_t v)
	{
		b.push_back(v & 0xff); b.push_back((v >> 8) & 0xff);
		b.push_back((v >> 16) & 0xff); b.push_back((v >> 24) & 0xff);
	}

	static void put_f64(std::vector<uint8_t>& b, double v)
	{
		uint64_t u;
		memcpy(&u, &v, 8);
		for (int i = 0; i < 8; ++i) {
			b.push_back((u >> (8 * i)) & 0xff);
		}
	}

	static uint32_t get_u32(uint8_t const* p)
	{
		return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
		       (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
	}

	static uint64_t get_u64(uint8_t const* p)
	{
		uint64_t v = 0;
		for (int i = 0; i < 8; ++i) {
			v |= uint64_t(p[i]) << (8 * i);
		}
		return v;
	}

	static double get_f64(uint8_t const* p)
	{
		auto const u = get_u64(p);
		double v;
		memcpy(&v, &u, 8);
		return v;
	}

	bool send_all(uint8_t const* data, size_t n)
	{
		size_t sent = 0;
		while (sent < n) {
			auto r = ::send(_fd, data + sent, n - sent, MSG_NOSIGNAL);
			if (r <= 0) {
				return false;
			}
			sent += static_cast<size_t>(r);
		}
		return true;
	}

	bool recv_exact(uint8_t* data, size_t n)
	{
		size_t got = 0;
		while (got < n) {
			auto r = ::recv(_fd, data + got, n - got, 0);
			if (r <= 0) {
				return false;
			}
			got += static_cast<size_t>(r);
		}
		return true;
	}

	std::string _path;
	int _fd = -1;
	uint64_t _generation = 0;
	/* T2.31: the reusable outgoing-frame shm segment. */
	void* _shm_ptr = nullptr;
	size_t _shm_size = 0;
	std::string _shm_name;
};

#endif
