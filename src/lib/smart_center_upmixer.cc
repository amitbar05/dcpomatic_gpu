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


string
SmartCenterUpmixer::name() const
{
	return _("Smart centre (dialogue extraction to L/C/R)");
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
		if (N >= 3) {
			/* Centre extraction: dialogue -> C, removed from L/R.
			 * L'+C=L, R'+C=R; |C|,|L'|,|R'| <= 1 by construction. */
			auto const mid = 0.5f * (left + right);
			out->data()[0][i] = left - mid;   /* (L - R) / 2 */
			out->data()[1][i] = right - mid;  /* (R - L) / 2 */
			out->data()[2][i] = mid;
		} else {
			/* No centre slot (film pinned < 3 channels): pass L/R
			 * through so dialogue survives as a phantom centre rather
			 * than vanishing into an unwritten mid. */
			if (N > 0) {
				out->data()[0][i] = left;
			}
			if (N > 1) {
				out->data()[1][i] = right;
			}
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
		/* Mono: feed BOTH input legs at unity so the extraction matrix
		 * (mid = (L+R)/2 = M) puts the whole mono signal in the centre
		 * (C = M) with L' = R' = 0 — mono belongs in the centre alone. */
		mapping.set(0, 0, 1);
		mapping.set(0, 1, 1);
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
