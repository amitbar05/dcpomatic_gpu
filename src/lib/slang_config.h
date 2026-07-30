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


/** @file src/lib/slang_config.h
 *  @brief The question "is the Slang GPU encoder on?", answered in ONE place.
 *
 *  It used to be answered twice with different expressions. J2KEncoder tested
 *  `getenv("DCPOMATIC_SLANG") != nullptr || config->slang().enable`, while
 *  Film::video_identifier() and Writer::can_fake_write() tested only
 *  `config->slang().enable`. Running with the documented env switch and the
 *  config flag off therefore encoded GPU frames under an identifier that did not
 *  mention Slang at all, so a killed export resumed on the plain CPU path
 *  matched the same identifier, adopted the half-written MXF and appended
 *  locally-encoded frames to GPU-encoded ones -- one picture asset, one
 *  descriptor, two producers, and verify_encode_contract blind to it because it
 *  only sees frames it encodes.
 *
 *  Any second copy of the test will drift the same way; call this instead.
 *
 *  There used to be a second function here, slang_effective_coder(), returning
 *  "ht" or "mq".  The HTJ2K (JPEG 2000 Part 15) coder was removed from this
 *  integration on 2026-07-31: Part 15 is not what a DCI DCP is specified to
 *  carry, so MQ (Part 1) is now the only coder and there is nothing to select.
 */

/** @return true if the Slang GPU encode path will be used, from either switch
 *  (Preferences -> GPU (Slang), or the DCPOMATIC_SLANG env var).
 */
bool slang_path_enabled();

#endif

#endif
