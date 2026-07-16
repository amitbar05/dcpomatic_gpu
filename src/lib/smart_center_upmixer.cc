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


#include "audio_buffers.h"
#include "audio_mapping.h"
#include "smart_center_upmixer.h"
#include <cmath>

#include "i18n.h"


using std::make_shared;
using std::min;
using std::shared_ptr;
using std::string;
using std::vector;


/* Soft-clip knee: linear below -3 dBFS, tanh knee above, asymptote 1.0.
 * Matches encoder/src/dcp/audio_mix.py (CLIP_KNEE / soft_clip). */
static float constexpr CLIP_KNEE = 0.70794578f;      /* 10^(-3/20) */
static float const INV_SQRT2 = 1.0f / std::sqrt(2.0f);


float
SmartCenterUpmixer::soft_clip(float x)
{
	auto const a = std::fabs(x);
	if (a <= CLIP_KNEE) {
		return x;
	}
	auto const span = 1.0f - CLIP_KNEE;
	auto const y = CLIP_KNEE + span * std::tanh((a - CLIP_KNEE) / span);
	return x < 0 ? -y : y;
}


string
SmartCenterUpmixer::name() const
{
	return _("Smart centre (L/R + soft-clipped C)");
}


string
SmartCenterUpmixer::id() const
{
	return N_("smart-center-upmixer");
}


int
SmartCenterUpmixer::out_channels() const
{
	return 3;
}


shared_ptr<AudioProcessor>
SmartCenterUpmixer::clone(int) const
{
	return make_shared<SmartCenterUpmixer>();
}


shared_ptr<AudioBuffers>
SmartCenterUpmixer::do_run(shared_ptr<const AudioBuffers> in, int channels)
{
	int const N = min(channels, 3);
	auto out = make_shared<AudioBuffers>(channels, in->frames());
	out->make_silent();
	for (int i = 0; i < in->frames(); ++i) {
		auto const left = in->data()[0][i];
		auto const right = in->data()[1][i];
		if (N > 0) {
			out->data()[0][i] = left;
		}
		if (N > 1) {
			out->data()[1][i] = right;
		}
		if (N > 2) {
			out->data()[2][i] = soft_clip((left + right) * INV_SQRT2);
		}
	}

	return out;
}


void
SmartCenterUpmixer::make_audio_mapping_default(AudioMapping& mapping) const
{
	AudioProcessor::make_audio_mapping_default(mapping);

	auto const inputs = mapping.input_channels();

	if (inputs == 1) {
		/* Mono: feed both inputs at half gain — the synthesized centre
		 * gets M/sqrt(2) and L/R get M/2 (gentle width, no doubling). */
		mapping.set(0, 0, 0.5);
		mapping.set(0, 1, 0.5);
		return;
	}

	/* Stereo (or more): first two channels are our L/R. */
	for (int i = 0; i < min(2, inputs); ++i) {
		mapping.set(i, i, 1);
	}
}


vector<NamedChannel>
SmartCenterUpmixer::input_names() const
{
	vector<NamedChannel> names = {
		NamedChannel(_("Left"), 0),
		NamedChannel(_("Right"), 1)
	};

	for (auto name: AudioProcessor::input_names()) {
		names.push_back(name);
	}

	return names;
}
