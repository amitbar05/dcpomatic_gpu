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


#include "lib/image.h"
#include "lib/timer.h"
#include "libavutil/pixfmt.h"


using std::string;


int main()
{
	auto constexpr TRIALS = 1024;

	auto make_black = [](AVPixelFormat fmt, string fmt_name) {
		auto image = std::make_shared<Image>(fmt, dcp::Size{ 3996, 2160 }, Image::Alignment::COMPACT);
		PeriodTimer timer("Image::make_black " + fmt_name);
		for (auto i = 0; i < TRIALS; ++i) {
			image->make_black();
		}
	};

	make_black(AV_PIX_FMT_UYVY422, "AV_PIX_FMT_UYVY422");
	make_black(AV_PIX_FMT_YUV422P9LE, "AV_PIX_FMT_YUV422P9LE");

	return 0;
}

