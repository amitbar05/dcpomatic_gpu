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


#include "lib/content_factory.h"
#include "lib/job_manager.h"
#include "lib/subtitle_film_encoder.h"
#include "lib/transcode_job.h"
#include "test.h"
#include <boost/test/unit_test.hpp>


using std::make_shared;
using std::vector;


BOOST_AUTO_TEST_CASE(export_srt_test)
{
	vector<boost::filesystem::path> files = { "dcp_sub3.xml", "dcp_sub7.xml" };

	for (auto file: files) {
		auto subs = content_factory("test/data" / file)[0];
		auto film = new_test_film("export_srt_test" + file.string(), { subs });

		auto const srt = dcp::filesystem::change_extension(file, ".srt");
		auto job = make_shared<TranscodeJob>(film, TranscodeJob::ChangedBehaviour::EXAMINE_THEN_STOP);
		job->set_encoder(
			make_shared<SubtitleFilmEncoder>(
				film,
				job,
				"build/test" / srt,
				"",
				false,
				false,
				SubtitleFormat::SRT
			)
		);
		JobManager::instance()->add(job);
		BOOST_REQUIRE(!wait_for_jobs());

		check_text_file("build/test" / srt, "test/data" / srt);
	}
}

