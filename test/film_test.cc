/*
    Copyright (C) 2023 Carl Hetherington <cth@carlh.net>

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


#include "lib/config.h"
#include "lib/content_factory.h"
#include "lib/dcp_content.h"
#include "lib/film.h"
#include "lib/job_manager.h"
#include "test.h"
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


using std::make_shared;


BOOST_AUTO_TEST_CASE(film_contains_atmos_content_test)
{
	auto atmos = content_factory("test/data/atmos_0.mxf")[0];
	auto image = content_factory("test/data/flat_red.png")[0];
	auto sound = content_factory("test/data/white.wav")[0];

	auto film1 = new_test_film("film_contains_atmos_content_test1", { atmos, image, sound });
	BOOST_CHECK(film1->contains_atmos_content());

	auto film2 = new_test_film("film_contains_atmos_content_test2", { sound, atmos, image });
	BOOST_CHECK(film2->contains_atmos_content());

	auto film3 = new_test_film("film_contains_atmos_content_test3", { image, sound, atmos });
	BOOST_CHECK(film3->contains_atmos_content());

	auto film4 = new_test_film("film_contains_atmos_content_test4", { image, sound });
	BOOST_CHECK(!film4->contains_atmos_content());
}


BOOST_AUTO_TEST_CASE(film_possible_reel_types_test1)
{
	auto film = new_test_film("film_possible_reel_types_test1");
	BOOST_CHECK_EQUAL(film->possible_reel_types().size(), 4U);

	film->examine_and_add_content(content_factory("test/data/flat_red.png"));
	BOOST_REQUIRE(!wait_for_jobs());
	BOOST_CHECK_EQUAL(film->possible_reel_types().size(), 4U);

	auto dcp = make_shared<DCPContent>("test/data/reels_test2");
	film->examine_and_add_content({dcp});
	BOOST_REQUIRE(!wait_for_jobs());
	BOOST_CHECK_EQUAL(film->possible_reel_types().size(), 4U);

	/* If we don't do this the set_reference_video will be overridden by the Film's
	 * check_settings_consistency() stuff.
	 */
	film->set_reel_type(ReelType::BY_VIDEO_CONTENT);
	dcp->set_reference_video(true);
	BOOST_CHECK_EQUAL(film->possible_reel_types().size(), 1U);
}


BOOST_AUTO_TEST_CASE(film_possible_reel_types_test2)
{
	auto film = new_test_film("film_possible_reel_types_test2");

	auto dcp = make_shared<DCPContent>("test/data/dcp_digest_test_dcp");
	film->examine_and_add_content({dcp});
	BOOST_REQUIRE(!wait_for_jobs());
	BOOST_CHECK_EQUAL(film->possible_reel_types().size(), 4U);

	dcp->set_reference_video(true);
	BOOST_CHECK_EQUAL(film->possible_reel_types().size(), 2U);
}


BOOST_AUTO_TEST_CASE(film_copy_remembered_assets_test)
{
	dcp::filesystem::remove_all("build/test/film_copy_remembered_assets_test2");

	auto content = content_factory("test/data/flat_red.png")[0];
	auto film = new_test_film("film_copy_remembered_assets_test", { content });
	make_and_verify_dcp(film);

	auto copy = make_shared<Film>(boost::filesystem::path("build/test/film_copy_remembered_assets_test2"));
	copy->copy_from(film, [](float) {});

	auto remembered = copy->read_remembered_assets();
	BOOST_REQUIRE_EQUAL(remembered.size(), 1U);
	auto path = find_asset(remembered, *copy->directory(), dcpomatic::DCPTimePeriod({}, dcpomatic::DCPTime::from_seconds(10)), film->video_identifier());
	BOOST_CHECK(path.has_value());

	for (auto path: dcp::filesystem::directory_iterator(film->dir("info"))) {
		check_file(path.path(), copy->dir("info") / path.path().filename());
	}
}


/** A brand-new Film must get a Studio/Facility value from construction onward
 *  (there is no derivable ground truth for these, unlike e.g. Territory, which
 *  Film::isdcf_name() computes live from the film's own content instead): the
 *  ISDCF-documented "no registered code" sentinel (NULL / NUL) when the user
 *  has never set a default in Preferences, or the real registered value when
 *  they have.
 */
BOOST_AUTO_TEST_CASE(film_default_studio_and_facility_test)
{
	{
		/* No default set in Config: new_test_film()'s constructor must seed the
		 * ISDCF sentinels, not leave Studio/Facility unset.
		 *
		 * This must be a *directory-scoped* ConfigRestorer, not the bare
		 * no-argument one: new_test_film() -> use_template() reads/creates
		 * Config's cached "default.xml" template via a FRESH
		 * State::read_path()/write_path() call every time, so if an earlier
		 * test in this binary used ANY ConfigRestorer (even a bare one -- its
		 * destructor resets State::override_path to boost::none, not back to
		 * the test sandbox), that lookup would silently fall through to the
		 * real machine's ~/.config/dcpomatic2, picking up whatever unrelated
		 * Studio/Facility state happens to be cached there instead of the
		 * "no default" state this block means to test. */
		ConfigRestorer cr(boost::filesystem::current_path() / "build/test/film_default_studio_and_facility_test1_config");

		auto film = new_test_film("film_default_studio_and_facility_test1");
		BOOST_REQUIRE(film->studio());
		BOOST_CHECK_EQUAL(*film->studio(), "NULL");
		BOOST_REQUIRE(film->facility());
		BOOST_CHECK_EQUAL(*film->facility(), "NUL");
	}

	{
		/* A registered default in Config: new_test_film()'s constructor must
		 * pick it up instead of the sentinel. */
		ConfigRestorer cr(boost::filesystem::current_path() / "build/test/film_default_studio_and_facility_test2_config");
		Config::instance()->set_default_studio("ABCD");
		Config::instance()->set_default_facility("XYZ");

		auto film = new_test_film("film_default_studio_and_facility_test2");
		BOOST_REQUIRE(film->studio());
		BOOST_CHECK_EQUAL(*film->studio(), "ABCD");
		BOOST_REQUIRE(film->facility());
		BOOST_CHECK_EQUAL(*film->facility(), "XYZ");
	}
}


/** Regression test for a real bug found by review: Config::default_template_read_path()
 *  lazily snapshots a "default.xml" template the FIRST time any new film is ever made
 *  in a config profile, then reuses that cached file forever. Film::use_template() used
 *  to copy _studio/_facility from that template film, so a Preferences change made
 *  *after* the very first film in a profile was silently ignored by every later "New..."
 *  film -- defeating the whole point of a Preferences default. The fix removed _studio/
 *  _facility from use_template()'s copy (matching how _audio_language was already
 *  excluded), so they always come fresh from Config at construction time instead.
 *
 *  This must reproduce the STALE-cache shape specifically: film_default_studio_and_
 *  facility_test above sets the Config default *before* the first film in its sandbox is
 *  ever created, so the lazily-created default.xml happens to already match -- it cannot
 *  distinguish "reads Config at construction" from "copies a template that happened to
 *  agree with Config". Here the first film is created (freezing default.xml with the
 *  sentinels) *before* the Config default is set, then a second film must still pick up
 *  the new default despite the stale cache.
 */
BOOST_AUTO_TEST_CASE(film_default_studio_and_facility_survives_stale_template_cache_test)
{
	ConfigRestorer cr(boost::filesystem::current_path() / "build/test/film_default_studio_and_facility_survives_stale_template_cache_test_config");

	/* Freeze default.xml with no Config default set (Studio/Facility == the sentinels). */
	auto first = new_test_film("film_default_studio_and_facility_survives_stale_template_cache_test1");
	BOOST_REQUIRE(first->studio());
	BOOST_CHECK_EQUAL(*first->studio(), "NULL");

	/* Only now set a real default -- default.xml already exists and will NOT be
	 * regenerated by this. */
	Config::instance()->set_default_studio("ACME");
	Config::instance()->set_default_facility("XYZ");

	auto second = new_test_film("film_default_studio_and_facility_survives_stale_template_cache_test2");
	BOOST_REQUIRE(second->studio());
	BOOST_CHECK_EQUAL(*second->studio(), "ACME");
	BOOST_REQUIRE(second->facility());
	BOOST_CHECK_EQUAL(*second->facility(), "XYZ");
}
