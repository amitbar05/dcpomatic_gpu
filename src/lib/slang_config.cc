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


#ifdef DCPOMATIC_SLANG

#include "slang_config.h"
#include "config.h"
#include <cstdlib>


bool
slang_path_enabled()
{
	/* GUI/config switch (Preferences -> GPU (Slang)) or the original env flag --
	 * either enables the GPU path. */
	return getenv("DCPOMATIC_SLANG") != nullptr || Config::instance()->slang().enable;
}

#endif
