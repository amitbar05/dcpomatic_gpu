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


/** @file  src/lib/slang_source_bitrate.cc
 *  @brief Probe the real bit rate of the film's source video, so the GPU
 *  export can match the DCP's JPEG2000 bandwidth to it.
 */


#ifdef DCPOMATIC_SLANG

#include "slang_source_bitrate.h"
#include "content.h"
#include "ffmpeg.h"
#include "ffmpeg_content.h"
#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
extern "C" {
#include <libavformat/avformat.h>
}
LIBDCP_ENABLE_WARNINGS
#include <boost/filesystem.hpp>
#include <algorithm>
#include <cmath>
#include <vector>


using std::dynamic_pointer_cast;
using std::max;
using std::shared_ptr;
using std::vector;
using boost::optional;


namespace {

/** Just enough FFmpeg to see one content's demuxed stream metadata (the
 *  base class opens the format context and finds the streams, FileGroup
 *  aware, exactly as the examiner does).
 */
class BitRateProbe : public FFmpeg
{
public:
	explicit BitRateProbe(shared_ptr<const FFmpegContent> content)
		: FFmpeg(content)
	{}

	AVCodecID video_codec() const
	{
		return _video_stream ? _format_context->streams[*_video_stream]->codecpar->codec_id : AV_CODEC_ID_NONE;
	}

	optional<int64_t> video_bit_rate() const
	{
		if (!_video_stream) {
			return {};
		}

		/* The video stream's own declared rate, where the container records
		 * one (MP4/MOV nearly always do).
		 */
		auto const declared = _format_context->streams[*_video_stream]->codecpar->bit_rate;
		if (declared > 0) {
			return declared;
		}

		/* Otherwise start from the container's overall rate — falling back to
		 * file size over duration — and take off whatever the other streams
		 * declare.
		 */
		auto rate = _format_context->bit_rate;
		if (rate <= 0 && _format_context->duration > 0) {
			int64_t bytes = 0;
			for (auto const& path: _ffmpeg_content->paths()) {
				boost::system::error_code ec;
				auto const size = boost::filesystem::file_size(path, ec);
				if (!ec) {
					bytes += size;
				}
			}
			rate = std::llround(bytes * 8.0 * AV_TIME_BASE / _format_context->duration);
		}
		if (rate <= 0) {
			return {};
		}

		auto video_only = rate;
		for (unsigned int i = 0; i < _format_context->nb_streams; ++i) {
			if (static_cast<int>(i) != *_video_stream) {
				video_only -= max<int64_t>(_format_context->streams[i]->codecpar->bit_rate, 0);
			}
		}

		/* Broken metadata could declare more audio than there is container;
		 * don't let that zero the answer.
		 */
		return video_only > 0 ? video_only : rate;
	}
};


struct Probe
{
	int64_t rate;
	AVCodecID codec;
};


vector<Probe>
probe_all(ContentList const& content)
{
	vector<Probe> probes;
	for (auto i: content) {
		auto ffmpeg = dynamic_pointer_cast<const FFmpegContent>(i);
		if (!ffmpeg) {
			continue;
		}
		try {
			BitRateProbe probe(ffmpeg);
			if (auto rate = probe.video_bit_rate()) {
				probes.push_back({*rate, probe.video_codec()});
			}
		} catch (...) {
			/* An unreadable or undecodable source just doesn't vote. */
		}
	}
	return probes;
}


/** The source's rate reflects its entropy coding, not a quality target, so
 *  a lossless/uncompressed source should just get the configured maximum.
 */
bool
is_lossless(AVCodecID codec)
{
	switch (codec) {
	case AV_CODEC_ID_FFV1:
	case AV_CODEC_ID_HUFFYUV:
	case AV_CODEC_ID_FFVHUFF:
	case AV_CODEC_ID_UTVIDEO:
	case AV_CODEC_ID_MAGICYUV:
	case AV_CODEC_ID_LAGARITH:
	case AV_CODEC_ID_RAWVIDEO:
	case AV_CODEC_ID_V210:
	case AV_CODEC_ID_V410:
	case AV_CODEC_ID_PNG:
	case AV_CODEC_ID_QTRLE:
		return true;
	default:
		return false;
	}
}


/** How many bits intra-only J2K needs per bit of this codec for like
 *  quality, as a rational.  Long-GOP codecs pack 3-6x more quality per bit
 *  than intra J2K (temporal prediction, plus the DCP's 12-bit 4:4:4 XYZ
 *  target vs the source's usual 8-bit 4:2:0); already-intra mezzanine
 *  codecs are near parity.
 */
void
j2k_equivalence(AVCodecID codec, int64_t& num, int64_t& den)
{
	num = 4;
	den = 1;
	switch (codec) {
	case AV_CODEC_ID_MPEG1VIDEO:
	case AV_CODEC_ID_MPEG2VIDEO:
		num = 2;
		break;
	case AV_CODEC_ID_MPEG4:
	case AV_CODEC_ID_VC1:
	case AV_CODEC_ID_WMV3:
	case AV_CODEC_ID_VP8:
		num = 3;
		break;
	case AV_CODEC_ID_H264:
		num = 4;
		break;
	case AV_CODEC_ID_HEVC:
	case AV_CODEC_ID_VP9:
		num = 6;
		break;
	case AV_CODEC_ID_AV1:
		num = 7;
		break;
	case AV_CODEC_ID_PRORES:
	case AV_CODEC_ID_DNXHD:
	case AV_CODEC_ID_MJPEG:
	case AV_CODEC_ID_JPEG2000:
	case AV_CODEC_ID_CFHD:
		/* already intra */
		num = 6;
		den = 5;
		break;
	default:
		/* unknown: treat like H.264 */
		break;
	}
}

}


optional<int64_t>
slang_source_video_bit_rate(ContentList const& content)
{
	optional<int64_t> best;
	for (auto const& probe: probe_all(content)) {
		best = max(best.get_value_or(0), probe.rate);
	}
	return best;
}


optional<int64_t>
slang_equivalent_j2k_bit_rate(ContentList const& content)
{
	optional<int64_t> best;
	for (auto const& probe: probe_all(content)) {
		if (is_lossless(probe.codec)) {
			return SLANG_J2K_BIT_RATE_UNBOUNDED;
		}
		int64_t num, den;
		j2k_equivalence(probe.codec, num, den);
		best = max(best.get_value_or(0), probe.rate * num / den);
	}
	return best;
}


int64_t
slang_floor_cap_round_j2k_bit_rate(int64_t source_rate, dcp::Size size, int frame_rate, int64_t maximum)
{
	auto const floor = max<int64_t>(
		std::llround(0.3 * size.width * size.height * frame_rate),
		10000000
		);
	auto const target = std::min(std::max(source_rate, floor), maximum);
	return std::min(std::max<int64_t>(std::llround(target / 1e6), 1) * 1000000, maximum);
}

#endif
