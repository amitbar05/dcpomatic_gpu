/*
    Copyright (C) 2026 Amit Bar

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


#ifndef DCPOMATIC_SLANG_CONFIG_H
#define DCPOMATIC_SLANG_CONFIG_H

#ifdef DCPOMATIC_SLANG

#include <string>


/** @file src/lib/slang_config.h
 *  @brief The two questions "is the Slang GPU encoder on?" and "which block
 *  coder will it use?", answered in ONE place.
 *
 *  These used to be answered twice with different expressions. J2KEncoder tested
 *  `getenv("DCPOMATIC_SLANG") != nullptr || config->slang().enable` and applied
 *  DCPOMATIC_SLANG_HETERO's forced "mq", while Film::video_identifier() and
 *  Writer::can_fake_write() tested only `config->slang().enable`. Running with
 *  the documented env switch and the config flag off therefore encoded HTJ2K
 *  (Part 15) frames under an identifier that did not mention Slang at all, so a
 *  killed export resumed on the plain CPU path matched the same identifier,
 *  adopted the half-written MXF and appended Part-1 frames to Part-15 ones --
 *  one picture asset, one descriptor, two codestream families, and
 *  verify_encode_contract blind to it because it only sees frames it encodes.
 *
 *  Any second copy of either test will drift the same way; call these instead.
 */

/** @return true if the Slang GPU encode path will be used, from either switch
 *  (Preferences -> GPU (Slang), or the DCPOMATIC_SLANG env var).
 */
bool slang_path_enabled();

/** @return the block coder the Slang path will ACTUALLY use ("ht" or "mq"),
 *  i.e. the configured one unless DCPOMATIC_SLANG_HETERO forces "mq" (the
 *  heterogeneous CPU+GPU mode mixes in CPU-encoded Part-1 frames, so the whole
 *  reel has to be Part-1). Never persisted back to Config.
 */
std::string slang_effective_coder();

#endif

#endif
