/*
    Copyright (C) 2019-2021 Carl Hetherington <cth@carlh.net>

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


#include "film.h"
#include "job.h"
#include "player.h"
#include "subtitle_film_encoder.h"
#include <dcp/filesystem.h>
#include <dcp/interop_text_asset.h>
#include <dcp/smpte_text_asset.h>
#include <sub/collect.h>
#include <sub/subrip_writer.h>
#include <fmt/format.h>
#include <boost/bind/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/variant.hpp>

#include "i18n.h"


using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;
using boost::optional;
#if BOOST_VERSION >= 106100
using namespace boost::placeholders;
#endif


/** @param output Directory, if there will be multiple output files, or a filename.
 *  @param initial_name Hint that may be used to create filenames, if @ref output is a directory.
 *  @param include_font true to refer to and export any font file (for Interop; ignored for SMPTE).
 */
SubtitleFilmEncoder::SubtitleFilmEncoder(
	shared_ptr<const Film> film,
	shared_ptr<Job> job,
	boost::filesystem::path output,
	string initial_name,
	bool split_reels,
	bool include_font,
	SubtitleFormat format
	)
	: FilmEncoder(film, job)
	, _split_reels(split_reels)
	, _include_font(include_font)
	, _reel_index(0)
	, _length(film->length())
{
	_player.set_play_referenced();
	_player.set_ignore_video();
	_player.set_ignore_audio();
	_player.Text.connect(boost::bind(&SubtitleFilmEncoder::text, this, _1, _2, _3, _4));

	int const files = split_reels ? film->reels().size() : 1;
	for (int i = 0; i < files; ++i) {

		boost::filesystem::path filename = output;
		if (dcp::filesystem::is_directory(filename)) {
			if (files > 1) {
				/// TRANSLATORS: _reel{} here is to be added to an export filename to indicate
				/// which reel it is.  Preserve the {}; it will be replaced with the reel number.
				filename /= fmt::format("{}_reel{}", initial_name, i + 1);
			} else {
				filename /= initial_name;
			}
		}

		_outputs.push_back(Output(format, filename));
	}

	for (auto i: film->reels()) {
		_reels.push_back(i);
	}

	_default_font = dcp::ArrayData(default_font_file());
}


void
SubtitleFilmEncoder::go()
{
	{
		shared_ptr<Job> job = _job.lock();
		DCPOMATIC_ASSERT(job);
		job->sub(_("Extracting"));
	}

	_reel_index = 0;

	while (!_player.pass()) {}

	int reel = 0;
	for (auto& i: _outputs) {
		i.prepare(_film, reel, {});
		if (_include_font) {
			i.add_fonts(_player.get_subtitle_fonts(), _default_font);
		}
		i.write();
		++reel;
	}
}


void
SubtitleFilmEncoder::text(PlayerText subs, TextType type, optional<DCPTextTrack> track, dcpomatic::DCPTimePeriod period)
{
	if (type != TextType::OPEN_SUBTITLE) {
		return;
	}

	_outputs[_reel_index].prepare(_film, _reel_index, track);

	for (auto i: subs.string) {
		/* XXX: couldn't / shouldn't we use period here rather than getting time from the subtitle? */
		i.set_in (i.in());
		i.set_out(i.out());
		_outputs[_reel_index].add(i, _include_font);
	}

	if (_split_reels && (_reel_index < int(_reels.size()) - 1) && period.from > _reels[_reel_index].from) {
		++_reel_index;
	}

	_last = period.from;

	if (auto job = _job.lock()) {
		job->set_progress(float(period.from.get()) / _length.get());
	}
}


Frame
SubtitleFilmEncoder::frames_done() const
{
	if (!_last) {
		return 0;
	}

	/* XXX: assuming 24fps here but I don't think it matters */
	return _last->seconds() * 24;
}


SubtitleFilmEncoder::Output::Output(SubtitleFormat format, boost::filesystem::path const& path)
	: _format(format)
{
	switch (_format) {
	case SubtitleFormat::XML:
		_path = dcp::filesystem::change_extension(path, ".xml");
		break;
	case SubtitleFormat::MXF:
		_path = dcp::filesystem::change_extension(path, ".mxf");
		break;
	case SubtitleFormat::SRT:
		_path = dcp::filesystem::change_extension(path, ".srt");
		break;
	}
}


void
SubtitleFilmEncoder::Output::prepare(shared_ptr<const Film> film, int reel_index, optional<DCPTextTrack> track)
{
	if (_output.which() == 1 || boost::get<shared_ptr<dcp::TextAsset>>(_output)) {
		return;
	}

	auto const lang = film->open_text_languages();

	switch (_format) {
	case SubtitleFormat::XML:
	{
		auto interop_asset = make_shared<dcp::InteropTextAsset>();
		_output = interop_asset;
		interop_asset->set_movie_title(film->name());
		if (lang.first) {
			interop_asset->set_language(lang.first->as_string());
		}
		interop_asset->set_reel_number(fmt::to_string(reel_index + 1));
		break;
	}
	case SubtitleFormat::MXF:
	{
		auto smpte_asset = make_shared<dcp::SMPTETextAsset>();
		_output = smpte_asset;
		smpte_asset->set_content_title_text(film->name());
		if (lang.first) {
			smpte_asset->set_language(*lang.first);
		} else if (track && track->language) {
			smpte_asset->set_language(track->language.get());
		}
		smpte_asset->set_edit_rate(dcp::Fraction(film->video_frame_rate(), 1));
		smpte_asset->set_reel_number(reel_index + 1);
		smpte_asset->set_time_code_rate(film->video_frame_rate());
		smpte_asset->set_start_time(dcp::Time());
		if (film->encrypted()) {
			smpte_asset->set_key(film->key());
		}
		break;
	}
	case SubtitleFormat::SRT:
		_output = std::vector<sub::RawSubtitle>();
		break;
	}
}


void
SubtitleFilmEncoder::Output::write() const
{
	switch (_output.which()) {
	case 0:
	{
		auto asset = boost::get<shared_ptr<dcp::TextAsset>>(_output);
		DCPOMATIC_ASSERT(asset);
		asset->write(_path);
		break;
	}
	case 1:
	{
		auto subs = sub::collect<std::vector<sub::Subtitle>>(boost::get<std::vector<sub::RawSubtitle>>(_output));
		sub::write_subrip(subs, _path);
		break;
	}
	}
}


void
SubtitleFilmEncoder::Output::add(StringText sub, bool include_font)
{
	switch (_output.which()) {
	case 0:
	{
		auto asset = boost::get<shared_ptr<dcp::TextAsset>>(_output);
		DCPOMATIC_ASSERT(asset);
		if (_format == SubtitleFormat::XML && !include_font) {
			sub.unset_font();
		}
		asset->add(make_shared<dcp::TextString>(sub));
		break;
	}
	case 1:
	{
		auto& raw_subs = boost::get<vector<sub::RawSubtitle>>(_output);
		sub::RawSubtitle raw_sub;
		raw_sub.text = sub.text();
		raw_sub.from = sub::Time::from_hmsf(0, 0, std::floor(sub.in().as_seconds()), sub.in().e, sub::Rational{ sub.in().tcr, 1 });
		raw_sub.to = sub::Time::from_hmsf(0, 0, std::floor(sub.out().as_seconds()), sub.out().e, sub::Rational{ sub.out().tcr, 1});
		raw_sub.bold = sub.bold();
		raw_sub.italic = sub.italic();
		raw_sub.underline = sub.underline();
		raw_sub.vertical_position.proportional = sub.v_position();
		switch (sub.v_align()) {
		case dcp::VAlign::TOP:
			raw_sub.vertical_position.reference = sub::VerticalReference::TOP_OF_SCREEN;
			break;
		case dcp::VAlign::CENTER:
			raw_sub.vertical_position.reference = sub::VerticalReference::VERTICAL_CENTRE_OF_SCREEN;
			break;
		case dcp::VAlign::BOTTOM:
			raw_sub.vertical_position.reference = sub::VerticalReference::BOTTOM_OF_SCREEN;
			break;
		}
		raw_subs.push_back(raw_sub);
		break;
	}
	}
}


void
SubtitleFilmEncoder::Output::add_fonts(vector<shared_ptr<dcpomatic::Font>> const& fonts, dcp::ArrayData default_font)
{
	if (_format == SubtitleFormat::MXF) {
		auto asset = boost::get<shared_ptr<dcp::TextAsset>>(_output);
		DCPOMATIC_ASSERT(asset);
		for (auto font: fonts) {
			asset->add_font(font->id(), font->data().get_value_or(default_font));
		}
	}
}

