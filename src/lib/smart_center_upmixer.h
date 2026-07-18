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


/** @file  src/lib/smart_center_upmixer.h
 *  @brief SmartCenterUpmixer class.
 */


#include "audio_processor.h"


/** @class SmartCenterUpmixer
 *  @brief Mono/stereo -> L/C/R with a mid/side centre-EXTRACTION matrix.
 *
 *  Cinema dialogue belongs in the CENTRE speaker, but mono/stereo sources have
 *  no discrete centre.  This processor derives one and REMOVES it from L/R, so
 *  dialogue plays from the centre alone rather than the centre plus a phantom
 *  in L+R ("doubling"):
 *
 *      mid = (L + R) / 2 ;  C = mid ;  L' = L - mid ;  R' = R - mid
 *
 *  L' + C = L and R' + C = R exactly (the stereo image is preserved for an
 *  equidistant listener); fully-correlated content lands in C with |C| <= 1 by
 *  construction (no clipping needed); out-of-phase content cancels in C.  This
 *  is DCP-o-matic's own Mid/Side decode routed to L/C/R.  A mono source goes to
 *  C alone (L' = R' = 0).  LFE/Ls/Rs are left silent.
 *
 *  Mirrors encoder/src/dcp/audio_mix.py (the GPU export's Python reference).
 */
class SmartCenterUpmixer : public AudioProcessor
{
public:
	std::string name() const override;
	std::string id() const override;
	int out_channels() const override;
	std::shared_ptr<AudioProcessor> clone(int) const override;
	void make_audio_mapping_default(AudioMapping& mapping) const override;
	std::vector<NamedChannel> input_names() const override;

protected:
	std::shared_ptr<AudioBuffers> do_run(std::shared_ptr<const AudioBuffers>, int channels) override;
};
