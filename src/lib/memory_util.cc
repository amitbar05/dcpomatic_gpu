/*
    Copyright (C) 2012-2022 Carl Hetherington <cth@carlh.net>

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


#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
extern "C" {
#include <libavutil/avutil.h>
}
LIBDCP_ENABLE_WARNINGS
#include <algorithm>
#include <new>
#include <stdexcept>


void *
wrapped_av_malloc(size_t s)
{
	auto p = av_malloc(s);
	if (!p) {
		throw std::bad_alloc();
	}
	return p;
}


void
fill_memory(void* ptr, size_t bytes, uint64_t value)
{
	if (bytes == 0) {
		return;
	}

	auto const start = std::min(bytes, sizeof(value) - reinterpret_cast<uintptr_t>(ptr) % sizeof(value));
	auto start_ptr = reinterpret_cast<uint8_t*>(ptr);
	if (start < 8) {
		for (auto i = 0UL; i < start; ++i) {
			*start_ptr++ = value & 0xff;
			value = (value >> 8) | ((value & 0xff) << 56);
		}

		bytes -= start;
		if (bytes == 0) {
			return;
		}
	}

	auto const main = (bytes - (bytes % sizeof(value))) / 8;
	auto main_ptr = reinterpret_cast<uint64_t*>(start_ptr);
	for (auto i = 0UL; i < main; ++i) {
		*main_ptr++ = value;
	}

	bytes -= main * 8;
	if (bytes == 0) {
		return;
	}

	auto end_ptr = reinterpret_cast<uint8_t*>(main_ptr);
	for (auto i = 0UL; i < bytes; ++i) {
		*end_ptr++ = value & 0xff;
		value = (value >> 8) | ((value & 0xff) << 56);
	}
}


uint64_t
copy_16_bit_words_to_64_bit(uint16_t v)
{
	return static_cast<uint64_t>(v) | (static_cast<uint64_t>(v) << 16) | (static_cast<uint64_t>(v) << 32) | (static_cast<uint64_t>(v) << 48);
}


uint64_t
copy_bytes_to_64_bit(uint8_t v)
{
	return static_cast<uint64_t>(v) | (static_cast<uint64_t>(v) << 8) | (static_cast<uint64_t>(v) << 16) | (static_cast<uint64_t>(v) << 24)
	| (static_cast<uint64_t>(v) << 32) | (static_cast<uint64_t>(v) << 40) | (static_cast<uint64_t>(v) << 48) | (static_cast<uint64_t>(v) << 56);
}
