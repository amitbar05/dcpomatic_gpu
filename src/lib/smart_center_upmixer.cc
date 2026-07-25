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


/* SLANG_SMART_CENTER_MATRIX_BEGIN (dependency-free and inside these markers:
 * tests/test_smart_center_matrix.py slices this block out and compiles it
 * standalone, so the matrix is EXECUTED by the gate rather than eyeballed). */

/** One sample through the upmix matrix.
 *
 *  `left`/`right` are the stereo legs, `mono` the mono leg (zero unless a mono
 *  source is routed there).  The three outputs are the front DCP channels.
 *
 *  Stereo is a mid/side centre EXTRACTION -- l + c == left and r + c == right,
 *  so the image is preserved and dialogue is not doubled into a phantom centre.
 *  Mono is a centre-dominant SPREAD -- c == mono, l == r == side_gain * mono --
 *  which is why it needs its own leg: with a mono signal on both stereo legs,
 *  l and r come out identically zero and no gain can recover it.
 */
static void
smart_center_sample(float left, float right, float mono, float side_gain,
		    float* l, float* r, float* c)
{
	float const mid = 0.5f * (left + right);
	*l = left - mid + side_gain * mono;
	*r = right - mid + side_gain * mono;
	*c = mid + mono;
}

/* SLANG_SMART_CENTER_MATRIX_END */


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
	/* A film pinned to fewer channels than the mono leg's index has no leg to
	 * read; the mono term then falls out and the stereo maths is untouched. */
	auto const have_mono = in->channels() > MONO_INPUT;
	for (int i = 0; i < in->frames(); ++i) {
		auto const left = in->data()[0][i];
		auto const right = in->data()[1][i];
		auto const mono = have_mono ? in->data()[MONO_INPUT][i] : 0.0f;
		if (N >= 3) {
			/* Centre extraction for stereo, centre-dominant spread
			 * for mono -- see smart_center_sample() above. */
			smart_center_sample(
				left, right, mono, MONO_SIDE_GAIN,
				&out->data()[0][i], &out->data()[1][i], &out->data()[2][i]
				);
		} else {
			/* No centre slot (film pinned < 3 channels): pass L/R
			 * through so dialogue survives as a phantom centre rather
			 * than vanishing into an unwritten mid.  The mono leg has
			 * no centre to be dominant in either, so it goes to both
			 * at unity rather than at MONO_SIDE_GAIN -- 6 dB down with
			 * nothing to be down RELATIVE to would just be quiet. */
			if (N > 0) {
				out->data()[0][i] = left + mono;
			}
			if (N > 1) {
				out->data()[1][i] = right + mono;
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
		/* Mono: its own leg, which do_run() spreads C at unity / L,R at
		 * MONO_SIDE_GAIN.  (Feeding both L/R legs instead -- what this did
		 * until 2026-07-25 -- puts the whole signal in C and leaves L' and
		 * R' at exactly zero.) */
		mapping.set(0, MONO_INPUT, 1);
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
		NamedChannel(_("Right"), 1),
		/* Not a DCP channel: the leg a mono source takes, spread to L/C/R
		 * by do_run().  Named for what routing content here DOES, since
		 * that is what the mapping grid is asking the operator. */
		NamedChannel(_("Mono (to L/C/R)"), SmartCenterUpmixer::MONO_INPUT)
	};

	for (auto name: AudioProcessor::input_names()) {
		names.push_back(name);
	}

	return names;
}
