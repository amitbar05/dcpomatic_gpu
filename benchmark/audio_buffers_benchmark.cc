/*
    Copyright (C) 2026 Carl Hetherington <cth@carlh.net>

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


#include "lib/audio_buffers.h"
#include "lib/timer.h"
#include "libavutil/pixfmt.h"


using std::string;


void audio_buffers_benchmark();


int main()
{
	audio_buffers_benchmark();
	return 0;
}


void
audio_buffers_benchmark()
{
	auto constexpr TRIALS = 4096;
	auto constexpr CHANNELS = 6;
	auto constexpr FRAMES = 1024 * 1024;

	{
		AudioBuffers from(CHANNELS, FRAMES);
		AudioBuffers to(CHANNELS, FRAMES);
		PeriodTimer timer("AudioBuffers::accumulate_channel");
		for (int i = 0; i < TRIALS; ++i) {
			to.accumulate_channel(&from, 0, 3, 0.7);
		}
	}
}

