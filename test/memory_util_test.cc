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


#include "lib/memory_util.h"
#include <boost/test/unit_test.hpp>


void check(void* p, int size, uint64_t value)
{
	uint8_t* up = reinterpret_cast<uint8_t*>(p);
	for (int i = 0; i < size; ++i) {
		BOOST_CHECK_EQUAL(*up++, (value >> (i * 8)) & 0xff);
	}
}


/** Aligned, multiple of 64 size */
BOOST_AUTO_TEST_CASE(fill_memory_test1)
{
	int constexpr size = 256;
	auto memory = wrapped_av_malloc(size);
	BOOST_REQUIRE_EQUAL(reinterpret_cast<uintptr_t>(memory) % 8, 0);

	fill_memory(memory, size, 0x1928374654abdfea);
	check(memory, size, 0x1928374654abdfea);
}


/** Aligned, extra bytes at the end */
BOOST_AUTO_TEST_CASE(fill_memory_test2)
{
	int constexpr size = 259;

	auto memory = wrapped_av_malloc(size);
	BOOST_REQUIRE_EQUAL(reinterpret_cast<uintptr_t>(memory) % 8, 0);

	fill_memory(memory, size, 0x1928374654abdfea);
	check(memory, size, 0x1928374654abdfea);
}


/** Non-aligned, extra bytes at start and end */
BOOST_AUTO_TEST_CASE(fill_memory_test3)
{
	int constexpr size = 265;

	auto memory = wrapped_av_malloc(size + 512);
	BOOST_REQUIRE_EQUAL(reinterpret_cast<uintptr_t>(memory) % 8, 0);
	memory = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(memory) + 3);

	fill_memory(memory, size, 0x1928374654abdfea);
	check(memory, size, 0x1928374654abdfea);
}

