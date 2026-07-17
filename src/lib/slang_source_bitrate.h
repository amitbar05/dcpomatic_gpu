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


/** @file  src/lib/slang_source_bitrate.h
 *  @brief Probe the real bit rate of the film's source video, so the GPU
 *  export can match the DCP's JPEG2000 bandwidth to it.
 */

#ifdef DCPOMATIC_SLANG

#ifndef DCPOMATIC_SLANG_SOURCE_BITRATE_H
#define DCPOMATIC_SLANG_SOURCE_BITRATE_H


#include "types.h"
#include <dcp/types.h>
#include <boost/optional.hpp>
#include <cstdint>


/** Probe the video bit rate (bits per second) of each FFmpeg content in
 *  @p content and return the highest one, or nullopt if nothing could be
 *  probed (no FFmpeg video content, unreadable files).  Per content the
 *  figure is, in order of preference: the video stream's declared bit rate;
 *  the container bit rate less the other streams' declared rates; the file
 *  size over the duration less the other streams' declared rates.
 */
boost::optional<int64_t> slang_source_video_bit_rate(ContentList const& content);


/** slang_equivalent_j2k_bit_rate() result meaning "no finite target — use
 *  the configured maximum"; large enough that any sane clamp does that.
 */
int64_t constexpr SLANG_J2K_BIT_RATE_UNBOUNDED = INT64_C(1000000000000);

/** As slang_source_video_bit_rate(), but scale each content's probed rate
 *  by how many bits intra-only J2K needs per bit of its codec at like
 *  quality (MPEG-2 x2, H.264 x4, HEVC/VP9 x6, AV1 x7, intra mezzanine
 *  codecs x1.2) before taking the highest.  Lossless/uncompressed sources
 *  return SLANG_J2K_BIT_RATE_UNBOUNDED.  The caller still owns flooring
 *  and capping.
 */
boost::optional<int64_t> slang_equivalent_j2k_bit_rate(ContentList const& content);


/** Turn a probed source rate (see slang_equivalent_j2k_bit_rate()) into the
 *  JPEG2000 bit rate to actually configure: floored at 0.3 bits/pixel of
 *  the DCP raster (or 10 Mbps, whichever is higher), capped at @p maximum,
 *  and rounded to the nearest whole Mbps.  Shared by the GPU-export menu's
 *  explicit probe (dcpomatic.cc) and Film's automatic probe-on-import.
 */
int64_t slang_floor_cap_round_j2k_bit_rate(int64_t source_rate, dcp::Size size, int frame_rate, int64_t maximum);


#endif

#endif
