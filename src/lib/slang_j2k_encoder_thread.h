/*
    GPU (Slang/Vulkan) JPEG2000 encoder thread for DCP-o-matic.

    Drop-in sibling of CPUJ2KEncoderThread / GrokJ2KEncoderThread: a synchronous
    encoder thread whose encode() routes each frame to the external GPU encoder
    (frame_server.py) over a Unix-domain socket and returns the .j2c bytes for
    libdcp to wrap into the picture MXF.

    Place this file (+ the .cc and slang_frame_client.h) in dcpomatic/src/lib/,
    add the .cc to the wscript, and build with -DDCPOMATIC_SLANG. See README.md.
*/

#ifndef DCPOMATIC_SLANG_J2K_ENCODER_THREAD_H
#define DCPOMATIC_SLANG_J2K_ENCODER_THREAD_H


#include "colour_conversion.h"
#include "j2k_sync_encoder_thread.h"
#include <dcp/array_data.h>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>


class DCPVideo;
class J2KEncoder;
class SlangFrameClient;


class SlangJ2KEncoderThread : public J2KSyncEncoderThread
{
public:
	/** @param socket_path Unix-domain socket the frame server listens on.
	 *  @param coder Tier-1 block coder to request from the server for this
	 *  connection ("ht" — the fast default — or "mq"; the Preferences
	 *  picker). Empty = leave the server's own default in charge. */
	SlangJ2KEncoderThread(J2KEncoder& encoder, std::string socket_path, std::string coder = "");
	~SlangJ2KEncoderThread();

	void log_thread_start() const override;
	std::shared_ptr<dcp::ArrayData> encode(DCPVideo const& frame) override;

	/** @return seconds to back off after a server failure (the base run()
	 *  honours this so we don't hammer a dead socket). */
	int backoff() const override { return _backoff; }

private:
	bool maybe_send_tables(ColourConversion const& conversion);
	void maybe_send_options(DCPVideo const& frame);

	std::unique_ptr<SlangFrameClient> _client;
	int _backoff = 0;
	std::vector<uint16_t> _xyz;     ///< reused per-frame XYZ scratch
	std::vector<uint16_t> _rgb;     ///< reused per-frame RGB48 scratch (I2)
	/* I2 (GPU convert_to_xyz offload) state: which conversion's tables the
	 * server connection currently holds, and on which client connection
	 * generation they were installed (reconnect ⇒ the server forgot them). */
	std::string _tables_id;
	uint64_t _tables_generation = 0;
	/* J2KO per-connection options (coder + the film's bitrate/fps): resent
	 * once per connection generation, like the colour tables. Sticky off
	 * after a transport failure (a pre-J2KO server drops the connection on
	 * the unknown magic — everything still works on its defaults). */
	std::string _coder;
	uint64_t _options_generation = 0;
	bool _options_disabled = false;
	bool _rgb48_disabled = false;   ///< sticky off after a server rejection
	/* T2.31 (shm frame transport) state: sticky off after any shm-request
	 * failure (an old server drops the connection on the unknown magic; a new
	 * one reports a structured segment error) — either way the socket-payload
	 * path takes over for the rest of the run. DCPOMATIC_SLANG_NO_SHM=1
	 * disables it up front (the A/B measurement switch). */
	bool _shm_disabled = false;
};


#endif
