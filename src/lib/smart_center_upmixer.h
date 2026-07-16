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
 *  @brief Mono/stereo -> L/C/R with a synthesized, soft-clipped centre.
 *
 *  Cinema dialogue lives in the CENTRE speaker, but mono/stereo sources have
 *  no centre channel.  This processor passes L and R through and synthesizes
 *  C = softclip((L + R) / sqrt(2)) — the constant-power mono sum, so
 *  correlated (dialogue) content lands ~+3 dB in the centre while
 *  uncorrelated (wide) content stays at its natural level and out-of-phase
 *  content cancels.  Two full-scale correlated channels would sum to
 *  ~+3 dBFS, so the centre is soft-CLIPPED: exactly linear below the -3 dBFS
 *  knee, then a smooth tanh knee that can never reach full scale.
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

	static float soft_clip(float x);

protected:
	std::shared_ptr<AudioBuffers> do_run(std::shared_ptr<const AudioBuffers>, int channels) override;
};
