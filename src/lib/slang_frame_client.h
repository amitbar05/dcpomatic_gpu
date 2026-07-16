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
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>


class SlangFrameClient
{
public:
	explicit SlangFrameClient(std::string socket_path)
		: _path(std::move(socket_path)) {}

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
		static std::atomic<uint64_t> counter{0};
		char name[64];
		snprintf(name, sizeof(name), "/j2ks_%d_%llu",
			 static_cast<int>(getpid()),
			 static_cast<unsigned long long>(counter++));
		int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
		if (fd < 0) {
			return nullptr;
		}
		if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
			::close(fd);
			shm_unlink(name);
			return nullptr;
		}
		void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		::close(fd);
		if (p == MAP_FAILED) {
			shm_unlink(name);
			return nullptr;
		}
		_shm_ptr = p;
		_shm_size = bytes;
		_shm_name = name + 1;              /* wire name has no slash */
		return static_cast<uint16_t*>(p);
	}

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
