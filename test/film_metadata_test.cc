/*
    Copyright (C) 2013-2014 Carl Hetherington <cth@carlh.net>

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


/** @file  test/film_metadata_test.cc
 *  @brief Test some basic reading/writing of film metadata.
 *  @ingroup feature
 */


#include "lib/check_content_job.h"
#include "lib/config.h"
#include "lib/content.h"
#include "lib/content_factory.h"
#include "lib/dcp_content.h"
#include "lib/dcp_content_type.h"
#include "lib/image.h"
#include "lib/film.h"
#include "lib/job_manager.h"
#include "lib/player.h"
#include "lib/ratio.h"
#include "lib/text_content.h"
#include "lib/video_content.h"
#include "test.h"
#include <boost/date_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <fstream>


using std::string;
using std::list;
using std::make_shared;
using std::vector;


BOOST_AUTO_TEST_CASE (film_metadata_test)
{
	auto film = new_test_film("film_metadata_test");
	auto dir = test_film_dir ("film_metadata_test");

	film->set_isdcf_date(boost::gregorian::from_undelimited_string("20130211"));
	BOOST_CHECK (film->container() == Ratio::from_id ("185"));
	BOOST_CHECK (film->dcp_content_type() == DCPContentType::from_isdcf_name("TST"));

	film->set_name ("fred");
	film->set_dcp_content_type (DCPContentType::from_isdcf_name ("SHR"));
	film->set_container (Ratio::from_id ("185"));
	film->set_video_bit_rate(VideoEncoding::JPEG2000, 200000000);
	film->set_interop (false);
	film->set_chain (string(""));
	film->set_distributor (string(""));
	film->set_facility (string(""));
	/* This test is about general metadata round-tripping, not about the
	 * Config-derived Studio default a new Film now gets (see
	 * film_default_studio_and_facility_test) -- force it back to unset so the
	 * golden file below (which predates that feature and has no <Studio> tag
	 * at all) still matches. */
	film->set_studio(boost::none);
	film->set_release_territory (dcp::LanguageTag::RegionSubtag("US"));
	film->set_audio_channels(6);
	film->write_metadata ();

	list<Glib::ustring> ignore = { "Key", "ContextID", "LastWrittenBy" };
	check_xml ("test/data/metadata.xml.ref", dir.string() + "/metadata.xml", ignore);

	auto g = make_shared<Film>(dir);
	g->read_metadata ();

	BOOST_CHECK_EQUAL(g->name(), "fred");
	BOOST_CHECK_EQUAL(g->dcp_content_type(), DCPContentType::from_isdcf_name ("SHR"));
	BOOST_CHECK(g->container() == Ratio::from_id("185"));

	g->write_metadata ();
	check_xml ("test/data/metadata.xml.ref", dir.string() + "/metadata.xml", ignore);
}


/** Check a bug where <Content> tags with multiple <Text>s would fail to load */
BOOST_AUTO_TEST_CASE (multiple_text_nodes_are_allowed)
{
	Cleanup cl;

	auto subs = content_factory("test/data/15s.srt")[0];
	auto caps = content_factory("test/data/15s.srt")[0];
	auto film = new_test_film("multiple_text_nodes_are_allowed1", { subs, caps }, &cl);
	caps->only_text()->set_type(TextType::CLOSED_CAPTION);
	make_and_verify_dcp (
		film,
		{
			dcp::VerificationNote::Code::MISSING_CPL_METADATA,
			dcp::VerificationNote::Code::MISSING_SUBTITLE_LANGUAGE,
			dcp::VerificationNote::Code::INVALID_SUBTITLE_FIRST_TEXT_TIME
		});

	auto reload = make_shared<DCPContent>(film->dir(film->dcp_name()));
	auto film2 = new_test_film("multiple_text_nodes_are_allowed2", { reload });
	film2->write_metadata ();

	auto test = make_shared<Film>(boost::filesystem::path("build/test/multiple_text_nodes_are_allowed2"));
	test->read_metadata();

	cl.run();
}


/** Read some metadata from v2.14.x that fails to open on 2.15.x */
BOOST_AUTO_TEST_CASE (metadata_loads_from_2_14_x_1)
{
	namespace fs = boost::filesystem;
	auto dir = fs::path("build/test/metadata_loads_from_2_14_x_1");
	fs::remove_all(dir);
	auto film = make_shared<Film>(dir);
	fs::copy_file("test/data/2.14.x.metadata.1.xml", dir / "metadata.xml");
	auto notes = film->read_metadata(dir / "metadata.xml");
	BOOST_REQUIRE_EQUAL (notes.size(), 0U);
}


/** Read some more metadata from v2.14.x that fails to open on 2.15.x */
BOOST_AUTO_TEST_CASE (metadata_loads_from_2_14_x_2)
{
	namespace fs = boost::filesystem;
	auto dir = fs::path("build/test/metadata_loads_from_2_14_x_2");
	fs::remove_all(dir);
	auto film = make_shared<Film>(dir);
	fs::copy_file("test/data/2.14.x.metadata.2.xml", dir / "metadata.xml");
	auto notes = film->read_metadata(dir / "metadata.xml");
	BOOST_REQUIRE_EQUAL (notes.size(), 1U);
	BOOST_REQUIRE_EQUAL (notes.front(),
		       "A subtitle or closed caption file in this project is marked with the language 'eng', "
		       "which DCP-o-matic does not recognise.  The file's language has been cleared."
		       );
}


BOOST_AUTO_TEST_CASE (metadata_loads_from_2_14_x_3)
{
	namespace fs = boost::filesystem;
	auto dir = fs::path("build/test/metadata_loads_from_2_14_x_3");
	fs::remove_all(dir);
	auto film = make_shared<Film>(dir);
	fs::copy_file("test/data/2.14.x.metadata.3.xml", dir / "metadata.xml");
	auto notes = film->read_metadata(dir / "metadata.xml");

	BOOST_REQUIRE (film->release_territory());
	BOOST_REQUIRE (film->release_territory()->subtag() == dcp::LanguageTag::RegionSubtag("de").subtag());

	BOOST_REQUIRE (film->audio_language());
	BOOST_REQUIRE (*film->audio_language() == dcp::LanguageTag("sv-SE"));

	BOOST_REQUIRE (film->content_versions() == vector<string>{"3"});
	BOOST_REQUIRE (film->ratings() == vector<dcp::Rating>{ dcp::Rating("", "214rating") });
	BOOST_REQUIRE_EQUAL (film->studio().get_value_or(""), "214studio");
	BOOST_REQUIRE_EQUAL (film->facility().get_value_or(""), "214facility");
	BOOST_REQUIRE_EQUAL (film->temp_version(), true);
	BOOST_REQUIRE_EQUAL (film->pre_release(), true);
	BOOST_REQUIRE_EQUAL (film->red_band(), true);
	BOOST_REQUIRE_EQUAL (film->two_d_version_of_three_d(), true);
	BOOST_REQUIRE_EQUAL (film->chain().get_value_or(""), "214chain");
	BOOST_REQUIRE (film->luminance() == dcp::Luminance(14, dcp::Luminance::Unit::FOOT_LAMBERT));
}


/** Film::Film() seeds a brand-new film's Studio/Facility with a Config-derived
 *  default (see film_default_studio_and_facility_test in film_test.cc).  A
 *  project saved by an older DCP-o-matic, whose metadata.xml simply never
 *  mentions <Studio>, <Facility>, <TerritoryType> or <ReleaseTerritory> at
 *  all, must NOT retroactively pick up that new default (or the live
 *  Territory fallback's sentinel-shaped cousin) just because it's opened
 *  again: read_metadata() must overwrite the constructor's default
 *  unconditionally, all the way to boost::none, when a tag is absent.
 *
 *  This is a hand-constructed fixture, not one of the real 2.14.x captures
 *  above: those all happen to carry an (empty-but-present) <ISDCFMetadata>
 *  block with <Studio>/<Facility> tags in it, which isn't the case this test
 *  needs to cover -- a file with no trace of those tags anywhere.
 */
BOOST_AUTO_TEST_CASE(old_project_without_studio_facility_tags_keeps_old_isdcf_name)
{
	ConfigRestorer cr;
	/* A registered default would make this Film's own construction pick up a
	 * non-NULL/NUL value too, which would defeat the point of this test --
	 * make sure there isn't one, regardless of what earlier tests in this
	 * binary might have left behind in the process-wide Config singleton. */
	Config::instance()->unset_default_studio();
	Config::instance()->unset_default_facility();

	namespace fs = boost::filesystem;
	auto dir = fs::path("build/test/old_project_without_studio_facility_tags");
	fs::remove_all(dir);
	fs::create_directories(dir);

	{
		std::ofstream f((dir / "metadata.xml").string().c_str());
		f <<
			"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
			"<Metadata>\n"
			"  <Version>39</Version>\n"
			"  <Name>OldProject</Name>\n"
			"  <UseISDCFName>1</UseISDCFName>\n"
			"  <ISDCFDate>20200101</ISDCFDate>\n"
			"  <DCPContentType>TST</DCPContentType>\n"
			"  <Container>185</Container>\n"
			"  <Resolution>2K</Resolution>\n"
			"  <J2KBandwidth>150000000</J2KBandwidth>\n"
			"  <VideoFrameRate>24</VideoFrameRate>\n"
			"  <AudioChannels>6</AudioChannels>\n"
			"  <ThreeD>0</ThreeD>\n"
			"  <Sequence>1</Sequence>\n"
			"  <Interop>0</Interop>\n"
			"  <Encrypted>0</Encrypted>\n"
			"  <Key>8ef1bd64de5306314046118b277fd07a</Key>\n"
			"  <ReelType>0</ReelType>\n"
			"  <ReelLength>2000000000</ReelLength>\n"
			"  <ReencodeJ2K>0</ReencodeJ2K>\n"
			"  <UserExplicitVideoFrameRate>0</UserExplicitVideoFrameRate>\n"
			"  <Playlist>\n"
			"  </Playlist>\n"
			"</Metadata>\n";
	}

	/* This Film's constructor runs first -- seeding _studio/_facility with the
	 * Config-derived NULL/NUL sentinel, exactly like film_default_studio_and_facility_test
	 * proves for a genuinely new film -- and THEN read_metadata() must overwrite
	 * that back to boost::none, since the file has no <Studio>/<Facility> tags. */
	auto film = make_shared<Film>(dir);
	film->read_metadata();

	BOOST_CHECK(!film->studio());
	BOOST_CHECK(!film->facility());

	/* Territory has no stored Film member at all (by design -- see
	 * Film::isdcf_name()): it's computed live from the film's own content on
	 * every call, with no notion of "new" vs "old" project.  An old project
	 * with no <TerritoryType>/<ReleaseTerritory> tags lands in exactly the
	 * same "SPECIFIC, no release territory" state a brand new film starts in,
	 * so it correctly gets the SAME live TD/TL fallback a new film would --
	 * that is the intended, universally-applied part of this fix (Territory
	 * has a derivable ground truth; Studio/Facility do not).  This fixture has
	 * no subtitle content, so it resolves to INT-TL.  The result is therefore
	 * a 10-part name: the pre-fix 9 parts, PLUS the now-always-present
	 * Territory segment, with Studio and Facility still correctly omitted. */
	BOOST_CHECK_EQUAL(
		film->isdcf_name(false),
		"OldProject_TST-1_F_XX-XX_INT-TL_MOS_2K_20200101_SMPTE_OV"
		);
}


/** Check that an empty <MasteredLuminance> tag results in the film's luminance being unset */
BOOST_AUTO_TEST_CASE (metadata_loads_from_2_14_x_4)
{
	namespace fs = boost::filesystem;
	auto dir = fs::path("build/test/metadata_loads_from_2_14_x_4");
	fs::remove_all(dir);
	auto film = make_shared<Film>(dir);
	fs::copy_file("test/data/2.14.x.metadata.4.xml", dir / "metadata.xml");
	auto notes = film->read_metadata(dir / "metadata.xml");

	BOOST_REQUIRE (!film->luminance());
}


BOOST_AUTO_TEST_CASE (metadata_video_range_guessed_for_dcp)
{
	namespace fs = boost::filesystem;
	auto film = make_shared<Film>(fs::path("test/data/214x_dcp"));
	film->read_metadata();

	BOOST_REQUIRE_EQUAL(film->content().size(), 1U);
	BOOST_REQUIRE(film->content()[0]->video);
	BOOST_CHECK(film->content()[0]->video->range() == VideoRange::FULL);
}


BOOST_AUTO_TEST_CASE (metadata_video_range_guessed_for_mp4_with_unknown_range)
{
	namespace fs = boost::filesystem;
	auto film = make_shared<Film>(fs::path("test/data/214x_mp4"));
	film->read_metadata();

	BOOST_REQUIRE_EQUAL(film->content().size(), 1U);
	BOOST_REQUIRE(film->content()[0]->video);
	BOOST_CHECK(film->content()[0]->video->range() == VideoRange::VIDEO);
}


BOOST_AUTO_TEST_CASE (metadata_video_range_guessed_for_png)
{
	namespace fs = boost::filesystem;
	auto film = make_shared<Film>(fs::path("test/data/214x_png"));
	film->read_metadata();

	BOOST_REQUIRE_EQUAL(film->content().size(), 1U);
	BOOST_REQUIRE(film->content()[0]->video);
	BOOST_CHECK(film->content()[0]->video->range() == VideoRange::FULL);
}


/* Bug #2581 */
BOOST_AUTO_TEST_CASE(effect_node_not_inserted_incorrectly)
{
	auto sub = content_factory("test/data/15s.srt");
	auto film = new_test_film("effect_node_not_inserted_incorrectly", sub);
	film->write_metadata();

	namespace fs = boost::filesystem;
	auto film2 = make_shared<Film>(fs::path("build/test/effect_node_not_inserted_incorrectly"));
	film2->read_metadata();
	film2->write_metadata();

	cxml::Document doc("Metadata");
	doc.read_file("build/test/effect_node_not_inserted_incorrectly/metadata.xml");

	/* There should be no <Effect> node in the text, since we don't want to force the effect to "none" */
	BOOST_CHECK(!doc.node_child("Playlist")->node_child("Content")->node_child("Text")->optional_node_child("Effect"));
}


BOOST_AUTO_TEST_CASE(can_load_film_with_now_invalid_stream_ids)
{
	auto const name = std::string{"can_load_film_with_now_invalid_stream_ids"};
	auto film = new_test_film(name);
	boost::filesystem::remove(film->file("metadata.xml"));
	boost::filesystem::copy_file(boost::filesystem::path("test/data") / (name + ".xml"), film->file("metadata.xml"));
	{
		Editor editor(film->file("metadata.xml"));
		auto const replace = boost::filesystem::current_path() / boost::filesystem::path("test") / "data" / "phil.mkv";
		editor.replace("phil.mkv", replace.string());
	}
	film->read_metadata();

	auto player = make_shared<Player>(film, Image::Alignment::COMPACT, true);

	auto check = make_shared<CheckContentJob>(film);
	JobManager::instance()->add(make_shared<CheckContentJob>(film));
	BOOST_CHECK(!wait_for_jobs());
}

