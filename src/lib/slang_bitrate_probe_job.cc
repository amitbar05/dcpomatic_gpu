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

#ifdef DCPOMATIC_SLANG

#include "slang_bitrate_probe_job.h"
#include "slang_source_bitrate.h"
#include "film.h"

#include "i18n.h"


using std::shared_ptr;
using std::string;


SlangBitrateProbeJob::SlangBitrateProbeJob(shared_ptr<const Film> film)
	: Job(film)
{

}


string
SlangBitrateProbeJob::name() const
{
	return _("Probing source bit rate");
}


string
SlangBitrateProbeJob::json_name() const
{
	return N_("slang_probe_bitrate");
}


void
SlangBitrateProbeJob::run()
{
	_rate = slang_equivalent_j2k_bit_rate(_film->content());
	set_progress(1);
	set_state(FINISHED_OK);
}

#endif
