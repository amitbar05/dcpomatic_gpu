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

#include "check_box.h"
#include "dcpomatic_choice.h"
#include "dir_dialog.h"
#include "file_dialog.h"
#include "job_manager_view.h"
#include "language_tag_dialog.h"
#include "slang_audio_pipeline_view.h"
#include "slang_simple_panel.h"
#include "slang_ui_theme.h"
#include "wx_util.h"
#include "lib/audio_content.h"
#include "lib/audio_processor.h"
#include "lib/audio_stream.h"
#include "lib/config.h"
#include "lib/content.h"
#include "lib/content_factory.h"
#include "lib/cross.h"
#include "lib/dcp_content_type.h"
#include "lib/dcp_transcode_job.h"
#include "lib/examine_content_job.h"
#include "lib/film.h"
#include "lib/film_util.h"
#include "lib/job_manager.h"
#include "lib/signal_manager.h"
#include "lib/slang_audio_analyse_job.h"
#include "lib/text_content.h"
#include "lib/video_content.h"
#include <dcp/filesystem.h>
#include <dcp/scope_guard.h>
#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
#include <wx/clipbrd.h>
#include <wx/dcbuffer.h>
#include <wx/dnd.h>
#include <wx/graphics.h>
#include <wx/stdpaths.h>
LIBDCP_ENABLE_WARNINGS
#include <fmt/format.h>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <iostream>
#include <memory>


using std::dynamic_pointer_cast;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;
using std::weak_ptr;
using boost::optional;
using dcpomatic::DCPTime;
#if BOOST_VERSION >= 106100
using namespace boost::placeholders;
#endif


/** Width the card column is capped at; wider windows just get bigger margins,
 *  the way a settings page does, rather than one absurdly stretched row. */
static int const COLUMN_WIDTH = 1000;
static int const MARGIN = 24;
static int const CARD_GAP = 16;

/** How often the in-flight audio analysis' progress is pulled into the
 *  pipeline view.  Jobs report progress from their own thread, so this is a
 *  poll rather than a signal, exactly as JobManagerView does it. */
static int const ANALYSIS_POLL_MS = 250;


/** The subtitle formats this screen accepts, in ONE place.
 *
 *  The same list used to be written out three times in this file -- here, in
 *  choose_subtitles()'s wildcard, and in the Subtitles card's prose -- and the
 *  three had already drifted apart from each other and from what
 *  content_factory() can actually load.  A file in a format only one of them
 *  knew about was silently mis-routed: dropped on the video area it went to
 *  add_paths(..., false) and became a piece of "video" with no picture, which
 *  before this round could not be removed at all.
 *
 *  Extensions only, lower-cased; the leading dot is included.
 */
static char const* const SUBTITLE_EXTENSIONS[] = {
	".srt", ".ssa", ".ass", ".vtt", ".stl", ".sub", ".dfxp", ".ttml", ".xml", ".fcpxml"
};


static bool
is_subtitle_path(boost::filesystem::path const& path)
{
	auto const extension = boost::algorithm::to_lower_copy(path.extension().string());
	return std::find_if(
		std::begin(SUBTITLE_EXTENSIONS),
		std::end(SUBTITLE_EXTENSIONS),
		[&extension](char const* known) { return extension == known; }
		) != std::end(SUBTITLE_EXTENSIONS);
}


/** The same list as a wxFileDialog wildcard fragment ("*.srt;*.ssa;..."). */
static wxString
subtitle_wildcard()
{
	wxString out;
	for (auto const extension: SUBTITLE_EXTENSIONS) {
		if (!out.IsEmpty()) {
			out += char_to_wx(";");
		}
		out += char_to_wx("*") + char_to_wx(extension);
	}
	return out;
}


/** A free name for a new folder inside @p base: `<stem>`, then `<stem> 2`, ...
 *
 *  boost::none when everything up to the bound is taken.  Returning a colliding
 *  path on exhaustion -- what both call sites used to do, each with its own copy
 *  of the loop -- is not a harmless fallback: change_output_folder() treats what
 *  it gets back as "empty or newly invented", and its failure rollback DELETES
 *  that directory, so an exhausted search handed the rollback somebody's live
 *  folder to empty.
 *
 *  exists() takes an error_code because an unreadable, ESTALE or symlink-looped
 *  base must not throw out of a UI handler: wxApp::OnExceptionInMainLoop reports
 *  and then TERMINATES the program.
 */
static optional<boost::filesystem::path>
unique_child(boost::filesystem::path const& base, string const& stem)
{
	boost::system::error_code ec;
	for (int i = 1; i < 1000; ++i) {
		auto const path = i == 1 ? base / stem : base / fmt::format("{} {}", stem, i);
		if (!boost::filesystem::exists(path, ec)) {
			return path;
		}
	}
	return {};
}


/** Content this screen's Video card is responsible for: the picture, and any
 *  bare sound file dropped alongside it.
 *
 *  The video drop area's file dialog offers sound files as well as video, so a
 *  .wav added there is content the card accepted -- and it must therefore be
 *  content the card can remove.  It used to be neither shown nor removable by
 *  either card (the subtitle list filters on text tracks), so a stray score.wav
 *  became permanently welded to the project and was mixed into the DCP.
 */
static bool
is_video_card_content(shared_ptr<const Content> content)
{
	return static_cast<bool>(content->video) || (content->audio && content->text.empty());
}


/** Recursively copy a project directory.
 *
 *  Hand-rolled rather than boost::filesystem::copy(..., copy_options::recursive)
 *  for two reasons.  copy_options only exists in Boost >= 1.74, while this
 *  project's wscript accepts >= 1.45 on Linux and >= 1.61 elsewhere, so the bare
 *  symbol configures fine and then fails to compile (upstream guards the same
 *  symbol with #if BOOST_VERSION >= 107400 in test/subtitle_font_id_test.cc).
 *  And going through dcp::filesystem picks up libdcp's long-path handling, which
 *  a project directory full of MXFs on Windows needs.
 *
 *  It descends with a plain directory_iterator and never follows a directory
 *  SYMLINK.  That is load-bearing, not tidiness: boost::filesystem::copy has no
 *  cycle detection, so a symlinked route back into the source subtree re-copies
 *  each level's earlier siblings (measured at 195x amplification) until the path
 *  hits ENAMETOOLONG.  change_output_folder() refuses a target inside the
 *  project by filesystem identity; not following links is the structural half of
 *  the same guard.  boost::filesystem::relative() is avoided for the same
 *  version reason as copy_options (it arrived in 1.60), hence the explicit
 *  recursion rather than recursive_directory_iterator plus a relative path.
 */
static void
copy_tree(boost::filesystem::path const& from, boost::filesystem::path const& to)
{
	dcp::filesystem::create_directories(to);
	for (auto const& i: dcp::filesystem::directory_iterator(from)) {
		auto const destination = to / i.path().filename();
		if (boost::filesystem::is_symlink(boost::filesystem::symlink_status(i.path()))) {
			/* Copy the link itself; do not follow it. */
			boost::filesystem::copy_symlink(i.path(), destination);
		} else if (dcp::filesystem::is_directory(i.path())) {
			copy_tree(i.path(), destination);
		} else {
			/* CopyOptions::NONE: `to` was empty or newly invented, so a
			 * collision means an assumption is wrong and throwing beats
			 * silently replacing a file. */
			dcp::filesystem::copy_file(i.path(), destination, dcp::filesystem::CopyOptions::NONE);
		}
	}
}


SlangSimplePanel::SlangSimplePanel(wxWindow* parent)
	: wxPanel(parent, wxID_ANY)
	, _analysis_timer(this)
{
	SetBackgroundColour(slang_ui::palette().page);
	build();

	/* boost::bind is spelled out: an unqualified bind() with no placeholder
	 * argument finds POSIX bind(2) from <sys/socket.h> by ordinary lookup
	 * rather than boost's by ADL. */
	_job_added_connection = JobManager::instance()->JobAdded.connect(
		boost::bind(&SlangSimplePanel::job_added, this, _1)
		);
	_jobs_changed_connection = JobManager::instance()->ActiveJobsChanged.connect(
		boost::bind(&SlangSimplePanel::jobs_changed, this)
		);

	Bind(wxEVT_TIMER, [this](wxTimerEvent&) { poll_analysis(); }, _analysis_timer.GetId());
	Bind(wxEVT_SIZE, [this](wxSizeEvent& ev) { resized(); ev.Skip(); });

	update_all();
}


void
SlangSimplePanel::build()
{
	auto const p = slang_ui::palette();

	auto outer = new wxBoxSizer(wxVERTICAL);
	outer->Add(build_header(this), 0, wxEXPAND);

	_scroller = new wxScrolledWindow(this, wxID_ANY);
	_scroller->SetBackgroundColour(p.page);
	_scroller->SetScrollRate(0, FromDIP(12));
	outer->Add(_scroller, 1, wxEXPAND);
	SetSizer(outer);

	auto row = new wxBoxSizer(wxHORIZONTAL);
	_left_margin = row->AddSpacer(FromDIP(MARGIN));
	auto column = new wxBoxSizer(wxVERTICAL);
	row->Add(column, 1, wxEXPAND);
	_right_margin = row->AddSpacer(FromDIP(MARGIN));

	column->AddSpacer(FromDIP(MARGIN));
	build_video_card(_scroller, column);
	column->AddSpacer(FromDIP(CARD_GAP));
	build_subtitle_card(_scroller, column);
	column->AddSpacer(FromDIP(CARD_GAP));
	build_output_card(_scroller, column);
	column->AddSpacer(FromDIP(CARD_GAP));
	build_audio_card(_scroller, column);
	column->AddSpacer(FromDIP(CARD_GAP));
	build_action_row(_scroller, column);
	column->AddSpacer(FromDIP(MARGIN));

	_scroller->SetSizer(row);
	_scroller->FitInside();
}


/** The bar across the top: what this screen is, and the way out of it. */
wxWindow*
SlangSimplePanel::build_header(wxWindow* parent)
{
	auto const p = slang_ui::palette();

	auto header = new wxPanel(parent, wxID_ANY);
	header->SetBackgroundStyle(wxBG_STYLE_PAINT);
	header->SetBackgroundColour(p.card);
	header->Bind(wxEVT_PAINT, [header, p](wxPaintEvent&) {
		wxAutoBufferedPaintDC dc(header);
		dc.SetBackground(wxBrush(p.card));
		dc.Clear();
		std::unique_ptr<wxGraphicsContext> gc(wxGraphicsContext::Create(dc));
		if (!gc) {
			return;
		}
		wxRect const rect(header->GetSize());
		/* A hairline under the bar, so the cards below read as a separate
		 * surface without needing a drop shadow we cannot draw portably. */
		gc->SetPen(wxPen(p.border));
		gc->StrokeLine(0, rect.height - 0.5, rect.width, rect.height - 0.5);
	});

	auto sizer = new wxBoxSizer(wxHORIZONTAL);

	auto titles = new wxBoxSizer(wxVERTICAL);
	auto title = new wxStaticText(header, wxID_ANY, _("Make a DCP"));
	title->SetFont(slang_ui::font(header, 4, true));
	title->SetForegroundColour(p.text);
	titles->Add(title);

	auto subtitle = new wxStaticText(
		header, wxID_ANY, _("Add your video, choose where it should go, and press Create DCP.")
		);
	subtitle->SetFont(slang_ui::font(header, -1));
	subtitle->SetForegroundColour(p.muted);
	titles->Add(subtitle, 0, wxTOP, FromDIP(2));

	sizer->Add(titles, 1, wxALIGN_CENTRE_VERTICAL);

	/* This screen runs without the menu bar (DOMFrame::set_simple_mode hides
	 * it), so File -> New has to exist somewhere on the screen itself or a user
	 * who has finished one DCP has no way to start the next one.  It is the
	 * host's own New Film flow, not a second implementation of it: the same
	 * name/location dialog, the same offer to save the project that is open. */
	_new = new SlangFlatButton(header, _("New..."), SlangFlatButton::Kind::SECONDARY);
	_new->SetToolTip(_("Start a new project, saving this one first if you want."));
	_new->on_click([this]() { NewProject(); });
	sizer->Add(_new, 0, wxALIGN_CENTRE_VERTICAL | wxLEFT, FromDIP(12));

	auto advanced = new SlangFlatButton(header, _("Advanced..."), SlangFlatButton::Kind::SECONDARY);
	advanced->SetToolTip(_("Switch to the full interface, with every setting and the menu bar."));
	advanced->on_click([this]() { Advanced(); });
	sizer->Add(advanced, 0, wxALIGN_CENTRE_VERTICAL | wxLEFT, FromDIP(12));

	auto border = new wxBoxSizer(wxVERTICAL);
	border->Add(sizer, 1, wxEXPAND | wxALL, FromDIP(MARGIN) / 2 + FromDIP(6));
	header->SetSizerAndFit(border);

	return header;
}


void
SlangSimplePanel::build_video_card(wxWindow* parent, wxSizer* sizer)
{
	auto const p = slang_ui::palette();

	_video_card = new SlangCard(parent, _("Video"), _("The picture and sound your DCP is made from."), 1);
	sizer->Add(_video_card, 0, wxEXPAND);

	_video_drop = new SlangDropArea(
		_video_card,
		_("Drop a video file here"),
		_("or click to choose one"),
		[this](vector<boost::filesystem::path> paths) { video_dropped(paths); }
		);
	_video_card->body()->Add(_video_drop, 0, wxEXPAND);

	_video_details = new wxPanel(_video_card, wxID_ANY);
	_video_details->SetBackgroundColour(p.card);
	auto details = new wxBoxSizer(wxHORIZONTAL);

	auto text = new wxBoxSizer(wxVERTICAL);
	_video_name = new wxStaticText(_video_details, wxID_ANY, wxEmptyString);
	_video_name->SetFont(slang_ui::font(_video_details, 0, true));
	_video_name->SetForegroundColour(p.text);
	text->Add(_video_name);

	_video_summary = new wxStaticText(_video_details, wxID_ANY, wxEmptyString);
	_video_summary->SetFont(slang_ui::font(_video_details, -1));
	_video_summary->SetForegroundColour(p.muted);
	text->Add(_video_summary, 0, wxTOP, FromDIP(3));

	_video_encoding = new wxStaticText(_video_details, wxID_ANY, wxEmptyString);
	_video_encoding->SetFont(slang_ui::font(_video_details, -1));
	_video_encoding->SetForegroundColour(p.muted);
	text->Add(_video_encoding, 0, wxTOP, FromDIP(3));

	details->Add(text, 1, wxALIGN_CENTRE_VERTICAL);

	_video_replace = new SlangFlatButton(_video_details, _("Replace..."), SlangFlatButton::Kind::GHOST);
	_video_replace->SetToolTip(_("Choose a different file, replacing the one that is here."));
	_video_replace->on_click([this]() { replace_video(); });
	details->Add(_video_replace, 0, wxALIGN_CENTRE_VERTICAL);

	_video_remove = new SlangFlatButton(_video_details, _("Remove"), SlangFlatButton::Kind::GHOST);
	_video_remove->on_click([this]() { remove_video(); });
	details->Add(_video_remove, 0, wxALIGN_CENTRE_VERTICAL);

	_video_details->SetSizer(details);
	_video_card->body()->Add(_video_details, 0, wxEXPAND);
	_video_details->Hide();

	/* What the DCP is, which is a property of the programme rather than of the
	 * file, so it stays visible whether or not a video has been added yet.  It
	 * picks the CPL's ContentKind and the FTR/SHR/CLP part of the DCP name --
	 * and a "feature" is the one kind SMPTE Bv2.1 then requires end-credit
	 * markers on, so choosing correctly here is not only cosmetic. */
	auto type_row = new wxBoxSizer(wxHORIZONTAL);
	auto type_label = new wxStaticText(_video_card, wxID_ANY, _("This DCP is a"));
	type_label->SetFont(slang_ui::font(_video_card, -1));
	type_label->SetForegroundColour(p.muted);
	type_row->Add(type_label, 0, wxALIGN_CENTRE_VERTICAL | wxRIGHT, FromDIP(8));

	_content_type = new Choice(_video_card);
	for (auto type: DCPContentType::all()) {
		_content_type->add_entry(type->pretty_name());
	}
	/* ASCII only inside _(): wx routes a translatable char* through
	 * wxString::FromAscii, which asserts on the first byte >= 0x80 -- an em
	 * dash here stopped the program on the splash screen with
	 * "Non-ASCII value passed to FromAscii()".  Non-ASCII text has to go
	 * through char_to_wx(), as the "%dx%d" summary below does. */
	_content_type->SetToolTip(_("Feature, short, clip, trailer... - this sets the DCP's content kind and part of its name."));
	_content_type->Bind(wxEVT_CHOICE, boost::bind(&SlangSimplePanel::content_type_changed, this));
	type_row->Add(_content_type, 0, wxALIGN_CENTRE_VERTICAL);

	_video_card->body()->Add(type_row, 0, wxEXPAND | wxTOP, FromDIP(10));

	/* End credits.  A feature CPL that carries no FFEC and no FFMC marker is
	 * two SMPTE Bv2.1 errors, and this screen had no way to set them at all --
	 * so every feature made here failed verification, which is exactly what
	 * happened to a real export.  The full interface has a Markers dialog; this
	 * is the one question a feature actually has to answer, asked here.
	 *
	 * Only shown for a feature: no other content kind is required to carry
	 * them, and offering the control everywhere would invite marking credits on
	 * a trailer. */
	_credits_row = new wxPanel(_video_card, wxID_ANY);
	_credits_row->SetBackgroundColour(p.card);
	auto credits = new wxBoxSizer(wxVERTICAL);

	auto credits_line = new wxBoxSizer(wxHORIZONTAL);
	_credits_set = new CheckBox(_credits_row, _("End credits start at"));
	_credits_set->SetForegroundColour(p.muted);
	_credits_set->SetFont(slang_ui::font(_credits_row, -1));
	_credits_set->bind(&SlangSimplePanel::credits_changed, this);
	credits_line->Add(_credits_set, 0, wxALIGN_CENTRE_VERTICAL | wxRIGHT, FromDIP(8));

	/* set_button false: the "set from the current position" button belongs to
	 * the full interface's viewer, which this screen does not have. */
	_credits_at = new Timecode<DCPTime>(_credits_row, false);
	_credits_at->Changed.connect(boost::bind(&SlangSimplePanel::credits_changed, this));
	credits_line->Add(_credits_at, 0, wxALIGN_CENTRE_VERTICAL | wxRIGHT, FromDIP(8));

	_credits_end = new SlangFlatButton(_credits_row, _("At end of film"), SlangFlatButton::Kind::GHOST);
	_credits_end->on_click([this]() { credits_at_end(); });
	credits_line->Add(_credits_end, 0, wxALIGN_CENTRE_VERTICAL);

	credits->Add(credits_line, 0, wxEXPAND);

	_credits_hint = new wxStaticText(_credits_row, wxID_ANY, wxString{});
	_credits_hint->SetFont(slang_ui::font(_credits_row, -2));
	_credits_hint->SetForegroundColour(p.muted);
	credits->Add(_credits_hint, 0, wxEXPAND | wxTOP, FromDIP(4));

	_credits_row->SetSizer(credits);
	_video_card->body()->Add(_credits_row, 0, wxEXPAND | wxTOP, FromDIP(10));
	_credits_row->Hide();
}


bool
SlangSimplePanel::needs_credit_markers() const
{
	if (!_film) {
		return false;
	}
	auto const type = _film->dcp_content_type();
	return type && type->libdcp_kind() == dcp::ContentKind::FEATURE;
}


DCPTime
SlangSimplePanel::last_frame_time() const
{
	if (!_film) {
		return {};
	}
	auto const length = _film->length();
	auto const frame = DCPTime::from_frames(1, _film->video_frame_rate());
	return length > frame ? length - frame : DCPTime();
}


void
SlangSimplePanel::credits_at_end()
{
	if (!_film || !_credits_at || !_credits_set) {
		return;
	}
	_credits_at->set(last_frame_time(), _film->video_frame_rate());
	/* Filling in the time is only half an answer -- the markers are not written
	 * unless the box is ticked, and a user who pressed this button plainly
	 * wants them.  Tick it for them rather than leaving a filled-in time that
	 * does nothing. */
	checked_set(_credits_set, true);
	credits_changed();
}


void
SlangSimplePanel::credits_changed()
{
	if (!_film || !_credits_set || !_credits_at) {
		return;
	}

	if (!_credits_set->GetValue()) {
		_film->unset_marker(dcp::Marker::FFEC);
		_film->unset_marker(dcp::Marker::FFMC);
	} else {
		auto const vfr = _film->video_frame_rate();
		auto time = _credits_at->get(vfr);
		/* Clamp exactly as the full interface's Markers dialog does: a marker
		 * at or past the end is not a position in this film. */
		auto const last = last_frame_time();
		if (time > last) {
			time = last;
			_credits_at->set(time, vfr);
		}
		/* BOTH markers, from the one answer.  FFEC is the first frame of the
		 * end credits and FFMC the first frame of the MOVING (scrolling) part;
		 * they are separable, and the full interface's Markers dialog does
		 * separate them, but a simplified screen asking two nearly identical
		 * questions would get two nearly identical answers.  Setting them
		 * together is the common real-world case and satisfies Bv2.1; anyone
		 * who needs them apart has Advanced. */
		_film->set_marker(dcp::Marker::FFEC, time);
		_film->set_marker(dcp::Marker::FFMC, time);
	}

	update_credits();
	/* Same reason as content_type_changed(): no Save button, and no
	 * ContentChange will come along to carry this into save(). */
	save();
}


void
SlangSimplePanel::update_credits()
{
	if (!_credits_row) {
		return;
	}

	auto const show = needs_credit_markers();
	if (_credits_row->IsShown() != show) {
		_credits_row->Show(show);
		if (_scroller) {
			_scroller->Layout();
			_scroller->FitInside();
		}
	}
	if (!show || !_film) {
		return;
	}

	auto const vfr = _film->video_frame_rate();
	auto const ffec = _film->marker(dcp::Marker::FFEC);
	checked_set(_credits_set, static_cast<bool>(ffec));
	if (ffec) {
		_credits_at->set(*ffec, vfr);
	} else {
		/* Show where "At end of film" would put it, without claiming it is
		 * set: a hint is greyed out and is not read back by get() unless the
		 * field is empty, which is exactly the "unset" state. */
		_credits_at->set_hint(last_frame_time(), vfr);
	}

	auto const enabled = _sensitive && static_cast<bool>(_film);
	_credits_set->Enable(enabled);
	_credits_at->Enable(enabled && _credits_set->GetValue());
	_credits_end->Enable(enabled);

	_credits_hint->SetLabel(
		ffec
		? _("Both end-credit markers (FFEC and FFMC) will be written here.")
		: _("A feature DCP is required to carry end-credit markers; without them it will "
		    "fail verification. If there are no separate end credits, press \"At end of film\".")
		);
	_credits_hint->Wrap(_credits_row->GetSize().GetWidth() > 0 ? _credits_row->GetSize().GetWidth() : FromDIP(600));
}


void
SlangSimplePanel::content_type_changed()
{
	if (!_film || !_content_type) {
		return;
	}
	if (auto const index = _content_type->get()) {
		_film->set_dcp_content_type(DCPContentType::from_index(*index));
		/* This screen has no Save button (see save()), and unlike the content
		 * edits there is no ContentChange to carry it into
		 * content_layout_changed() -- so a type picked here and never followed
		 * by a content edit would be lost on close. */
		save();
	}
}


void
SlangSimplePanel::update_content_type()
{
	if (!_content_type) {
		return;
	}

	_content_type->Enable(_sensitive && static_cast<bool>(_film));
	if (!_film) {
		return;
	}

	if (auto const index = DCPContentType::as_index(_film->dcp_content_type())) {
		checked_set(_content_type, *index);
	}
}


void
SlangSimplePanel::build_subtitle_card(wxWindow* parent, wxSizer* sizer)
{
	auto const p = slang_ui::palette();

	_subtitle_card = new SlangCard(
		parent, _("Subtitles"),
		_("Optional. SRT, ASS/SSA, VTT, STL, SUB, Final Cut Pro XML or DCP subtitle XML."), 2
		);
	sizer->Add(_subtitle_card, 0, wxEXPAND);

	_subtitle_drop = new SlangDropArea(
		_subtitle_card,
		_("Drop subtitle files here"),
		_("or click to choose them - leave empty for no subtitles"),
		[this](vector<boost::filesystem::path> paths) { subtitles_dropped(paths); }
		);
	_subtitle_card->body()->Add(_subtitle_drop, 0, wxEXPAND);

	_subtitle_list = new wxPanel(_subtitle_card, wxID_ANY);
	_subtitle_list->SetBackgroundColour(p.card);
	_subtitle_list_sizer = new wxBoxSizer(wxVERTICAL);
	_subtitle_list->SetSizer(_subtitle_list_sizer);
	_subtitle_card->body()->Add(_subtitle_list, 0, wxEXPAND | wxTOP, FromDIP(8));
	_subtitle_list->Hide();
}


void
SlangSimplePanel::build_output_card(wxWindow* parent, wxSizer* sizer)
{
	auto const p = slang_ui::palette();

	_output_card = new SlangCard(parent, _("Output folder"), _("Where the finished DCP is written."), 3);
	sizer->Add(_output_card, 0, wxEXPAND);

	auto row = new wxBoxSizer(wxHORIZONTAL);

	auto text = new wxBoxSizer(wxVERTICAL);
	_output_path = new wxStaticText(_output_card, wxID_ANY, wxEmptyString);
	_output_path->SetFont(slang_ui::font(_output_card, 0, true));
	_output_path->SetForegroundColour(p.text);
	text->Add(_output_path);

	_output_dcp = new wxStaticText(_output_card, wxID_ANY, wxEmptyString);
	_output_dcp->SetFont(slang_ui::font(_output_card, -1));
	_output_dcp->SetForegroundColour(p.muted);
	text->Add(_output_dcp, 0, wxTOP, FromDIP(3));

	row->Add(text, 1, wxALIGN_CENTRE_VERTICAL);

	_output_change = new SlangFlatButton(_output_card, _("Change..."), SlangFlatButton::Kind::SECONDARY);
	_output_change->on_click([this]() { change_output_folder(); });
	row->Add(_output_change, 0, wxALIGN_CENTRE_VERTICAL | wxLEFT, FromDIP(12));

	_output_copy_path = new SlangFlatButton(_output_card, _("Copy Path"), SlangFlatButton::Kind::GHOST);
	_output_copy_path->SetToolTip(_("Copy the output folder's path to the clipboard."));
	_output_copy_path->on_click([this]() { copy_output_path(); });
	row->Add(_output_copy_path, 0, wxALIGN_CENTRE_VERTICAL | wxLEFT, FromDIP(8));

	_output_card->body()->Add(row, 0, wxEXPAND);
}


void
SlangSimplePanel::build_audio_card(wxWindow* parent, wxSizer* sizer)
{
	auto const p = slang_ui::palette();

	_audio_card = new SlangCard(
		parent, _("Sound"),
		_("Measured on the GPU as soon as your video is added, then levelled for the cinema.")
		);
	sizer->Add(_audio_card, 0, wxEXPAND);

	_pipeline = new SlangAudioPipelineView(_audio_card);
	_audio_card->body()->Add(_pipeline, 1, wxEXPAND);

	/* The spoken language of the soundtrack.  Optional in the DCP's NAME, where
	 * an unset language is written as XX -- the convention's own "not
	 * specified".
	 *
	 * It is NOT optional in the essence: libdcp's SoundAsset takes a mandatory
	 * language and falls back to en-US, which it stamps into every MCA
	 * sub-descriptor of the sound MXF.  So leaving this alone does not produce a
	 * package that declines to say; it produces one whose name says XX and whose
	 * essence says en-US.  That is worth being told about rather than
	 * discovering from a QC report, hence the note under the button. */
	auto language_row = new wxBoxSizer(wxHORIZONTAL);
	auto language_label = new wxStaticText(_audio_card, wxID_ANY, _("Spoken language"));
	language_label->SetFont(slang_ui::font(_audio_card, -1));
	language_label->SetForegroundColour(p.muted);
	language_row->Add(language_label, 0, wxALIGN_CENTRE_VERTICAL | wxRIGHT, FromDIP(8));

	_audio_language = new SlangFlatButton(_audio_card, _("Not specified"), SlangFlatButton::Kind::SECONDARY);
	_audio_language->SetToolTip(_("The language the soundtrack is spoken in; it becomes part of the DCP's name."));
	_audio_language->on_click([this]() { choose_audio_language(); });
	language_row->Add(_audio_language, 0, wxALIGN_CENTRE_VERTICAL);

	_audio_language_clear = new SlangFlatButton(_audio_card, _("Clear"), SlangFlatButton::Kind::GHOST);
	_audio_language_clear->SetToolTip(_("Leave the language unspecified (XX in the DCP name)."));
	_audio_language_clear->on_click([this]() { clear_audio_language(); });
	language_row->Add(_audio_language_clear, 0, wxALIGN_CENTRE_VERTICAL);
	_audio_language_clear->Hide();

	_audio_card->body()->Add(language_row, 0, wxEXPAND | wxTOP, FromDIP(10));

	_audio_language_note = new wxStaticText(_audio_card, wxID_ANY, wxEmptyString);
	_audio_language_note->SetFont(slang_ui::font(_audio_card, -1));
	_audio_language_note->SetForegroundColour(p.muted);
	_audio_card->body()->Add(_audio_language_note, 0, wxEXPAND | wxTOP, FromDIP(4));
}


void
SlangSimplePanel::choose_audio_language()
{
	if (!_film) {
		return;
	}

	LanguageTagDialog dialog(this, _film->audio_language().get_value_or(dcp::LanguageTag("en")));
	auto const result = dialog.ShowModal();
	/* Reported: the Create DCP row (and everything below the Sound card) can be
	 * left showing stale, blank content once this modal closes -- the toolkit
	 * does not reliably repaint the area the dialog covered.  Force it rather
	 * than trust an Expose/frame callback that this window manager may not
	 * send.  Unconditional: the stale paint is a side effect of the modal
	 * itself, not of what the user chose in it. */
	Refresh();
	Update();
	if (result == wxID_OK) {
		_film->set_audio_language(dialog.get());
		save();
	}
}


void
SlangSimplePanel::clear_audio_language()
{
	if (_film) {
		_film->set_audio_language(boost::none);
		save();
	}
}


void
SlangSimplePanel::update_audio_language()
{
	if (!_audio_language) {
		return;
	}

	auto const language = _film ? _film->audio_language() : optional<dcp::LanguageTag>();

	_audio_language->set_label_text(language ? std_to_wx(language->as_string()) : _("Not specified"));
	_audio_language->Enable(_sensitive && static_cast<bool>(_film));
	_audio_language_clear->Enable(_sensitive && static_cast<bool>(_film));
	_audio_language_clear->Show(static_cast<bool>(language));
	if (_audio_language_note) {
		/* Say what "not specified" actually ships as; see build_audio_card(). */
		_audio_language_note->SetLabel(
			language
				? wxString(wxEmptyString)
				: _("Left unset, the sound track is labelled en-US inside the DCP, and the DCP's name says XX.")
			);
		_audio_language_note->Show(!language);
	}
	_audio_card->Layout();
}


void
SlangSimplePanel::build_action_row(wxWindow* parent, wxSizer* sizer)
{
	auto const p = slang_ui::palette();

	auto row = new wxBoxSizer(wxHORIZONTAL);

	_create_note = new wxStaticText(parent, wxID_ANY, wxEmptyString);
	_create_note->SetFont(slang_ui::font(parent, -1));
	_create_note->SetForegroundColour(p.muted);
	row->Add(_create_note, 1, wxALIGN_CENTRE_VERTICAL | wxRIGHT, FromDIP(12));

	_create = new SlangFlatButton(parent, _("Create DCP"), SlangFlatButton::Kind::PRIMARY);
	_create->on_click([this]() { create_dcp(); });
	row->Add(_create, 0, wxALIGN_CENTRE_VERTICAL);

	sizer->Add(row, 0, wxEXPAND);

	auto progress = new SlangCard(parent, _("Progress"));
	sizer->AddSpacer(FromDIP(CARD_GAP));
	sizer->Add(progress, 0, wxEXPAND);

	_jobs = new JobManagerView(progress, false);
	/* Tall enough for the three or four jobs an export queues (probe, sound,
	 * examine, transcode) without the list scrolling away from the user. */
	_jobs->SetMinSize(wxSize(-1, FromDIP(260)));
	progress->body()->Add(_jobs, 1, wxEXPAND);
}


void
SlangSimplePanel::resized()
{
	if (!_left_margin || !_right_margin || !_scroller) {
		return;
	}

	auto const margin = FromDIP(MARGIN);
	auto const extra = std::max(0, _scroller->GetClientSize().GetWidth() - FromDIP(COLUMN_WIDTH) - margin * 2);
	_left_margin->SetMinSize(margin + extra / 2, 1);
	_right_margin->SetMinSize(margin + extra / 2, 1);
	_scroller->Layout();
	_scroller->FitInside();
}


void
SlangSimplePanel::set_film(shared_ptr<Film> film)
{
	if (_film == film) {
		return;
	}

	_film = film;
	_film_changed_connection.disconnect();
	_film_content_changed_connection.disconnect();
	_pending_subtitles.clear();

	/* The analysis members belong to the film that is going away.  Left armed,
	 * the previous film's job kept driving this screen: opening a second film
	 * (Ctrl+O still works here -- the accelerators outlive the hidden menu bar)
	 * showed ITS Sound card animating "Measuring the mix on the GPU... 43%", and
	 * when the old job finished, analysis_finished() stamped the new film with
	 * "Measured on the GPU and cross-checked" beside a gain box reading "Not
	 * measured yet".  A positive verification claim about a soundtrack that was
	 * never measured is the worst shape this screen can take. */
	_analysis_finished_connection.disconnect();
	_analysis_job.reset();
	_analysis_timer.Stop();
	_pipeline->set_analysing(false, {});

	if (_film) {
		_film_changed_connection = _film->Change.connect(
			boost::bind(&SlangSimplePanel::film_changed, this, _1, _2)
			);
		_film_content_changed_connection = _film->ContentChange.connect(
			boost::bind(&SlangSimplePanel::film_content_changed, this, _1, _2)
			);
	}

	_pipeline->set_film(_film);
	_pipeline->set_measurement_note(wxEmptyString);
	update_all();

	if (_film && _active && signal_manager) {
		/* A project made before the mono L/C/R spread opens with its mono
		 * stream still on the upmixer's L/R legs.  The export migrates it too,
		 * but doing it on open as well is what keeps this screen honest: it is
		 * supposed to show what will be exported, and the routing it draws
		 * would otherwise disagree with the DCP until the moment the button is
		 * pressed.  Deferred, so nothing runs while the film is still being
		 * wired up, and it re-measures because the mix it changed is the mix
		 * whose peak is on display. */
		signal_manager->when_idle(boost::bind(&SlangSimplePanel::migrate_mono_mapping, this));
	}
}


/** @return true if a DCP is being written right now.  See the header for why
 *  this must only ever be reached from a user action.
 */
bool
SlangSimplePanel::export_in_flight() const
{
	auto jobs = JobManager::instance()->get();
	return std::any_of(
		jobs.begin(),
		jobs.end(),
		[](shared_ptr<const Job> job) {
			return std::dynamic_pointer_cast<const DCPTranscodeJob>(job) && !job->finished();
		});
}


void
SlangSimplePanel::migrate_mono_mapping()
{
	if (!_film) {
		return;
	}

	/* Never while a DCP is being written.  Both halves of this function change
	 * the film underneath the running export: set_mapping() fires a
	 * ContentChange that makes the in-flight Player rebuild all its pieces
	 * mid-reel, and the analysis job ends by calling AudioContent::set_gain, so
	 * the delivered DCP would no longer match the project it was made from.
	 * Nothing is lost by skipping it -- the migration is idempotent and
	 * jobs_make_dcp_gpu_continue() runs it again, immediately before the
	 * analysis it exists to precede.
	 */
	if (export_in_flight()) {
		return;
	}

	auto const changed = _film->migrate_smart_center_mono_mapping();

	/* Save whatever the answer was.  The Film method sets its one-shot flag
	 * BEFORE looking for anything to rewrite (deliberately -- see its comment),
	 * so a run that matches nothing still dirties the film.  Returning early on
	 * `false` therefore left the beginner-facing screen asking "Save changes to
	 * project?" about a project the user had only opened -- and answering "no"
	 * left the flag unwritten, so the migration ran again next session and could
	 * rewrite a mapping deliberately built in between.  save() is dirty-gated,
	 * so this writes once and later calls cost nothing. */
	save();

	if (!changed) {
		return;
	}

	_film->maybe_analyse_audio_gain();
	update_all();
}


void
SlangSimplePanel::set_active(bool active)
{
	_active = active;
	if (active) {
		update_all();
		if (_film && signal_manager) {
			/* set_film() also posts this, but it early-returns when the film has
			 * not changed -- so switching INTO this mode with a film already
			 * open would otherwise never migrate.  The migration is idempotent,
			 * so posting it twice is harmless. */
			signal_manager->when_idle(boost::bind(&SlangSimplePanel::migrate_mono_mapping, this));
		}
	}
}


void
SlangSimplePanel::set_general_sensitivity(bool sensitive)
{
	_sensitive = sensitive;
	update_action_row();
	/* These edit the film, so they follow the same rule as the controls below:
	 * an export in flight owns the film's settings until it finishes. */
	update_content_type();
	update_audio_language();
	for (auto button: _subtitle_row_buttons) {
		button->Enable(sensitive);
	}

	/* The controls that change the film's CONTENT, which is the class this
	 * function used to miss entirely.  Everything above only edits settings; the
	 * four below add and remove the pieces a running DCPTranscodeJob is reading.
	 * Its Player shares the film's playlist, so removing the video mid-export
	 * collapses the playback length and the writer finalises a picture asset
	 * short of its reel -- a silently truncated DCP, not a failed job.  The
	 * buttons were locals in build_video_card() and rows rebuilt in
	 * update_subtitle_card(), so nothing had ever reached them. */
	if (_video_replace) {
		_video_replace->Enable(sensitive);
	}
	if (_video_remove) {
		_video_remove->Enable(sensitive);
	}
	if (_video_drop) {
		_video_drop->Enable(sensitive);
	}
	if (_subtitle_drop) {
		_subtitle_drop->Enable(sensitive);
	}

	update_output_change_enabled();
	update_output_copy_path_enabled();
	if (_new) {
		/* Same class: starting a new project discards the current film, and
		 * discarding the one an export is writing strands it.  Deliberately
		 * STRICTER than the File -> New menu item this button stands in for,
		 * which upstream leaves ALWAYS enabled -- a menu item two clicks deep
		 * is not the same hazard as a button sitting beside Advanced for the
		 * forty minutes an export runs.  Nothing is lost by it: Advanced
		 * brings the menu bar back, with its own New. */
		_new->Enable(sensitive);
	}
}


void
SlangSimplePanel::film_changed(ChangeType type, FilmProperty property)
{
	if (type != ChangeType::DONE || !_film) {
		return;
	}

	switch (property) {
	case FilmProperty::CONTENT:
		content_layout_changed();
		break;
	case FilmProperty::AUDIO_CHANNELS:
	case FilmProperty::AUDIO_PROCESSOR:
		_pipeline->refresh_state();
		break;
	case FilmProperty::NAME:
	case FilmProperty::USE_ISDCF_NAME:
		update_output_card();
		break;
	case FilmProperty::DCP_CONTENT_TYPE:
		update_content_type();
		/* The end-credits row exists only for a feature, so it appears and
		 * disappears with this. */
		update_credits();
		/* Both of these are ingredients of the ISDCF name the output card
		 * shows, so the card has to be re-read for the change to be visible
		 * where the user is looking for its effect. */
		update_output_card();
		break;
	case FilmProperty::AUDIO_LANGUAGE:
		update_audio_language();
		update_output_card();
		break;
	case FilmProperty::VIDEO_BIT_RATE:
	case FilmProperty::RESOLUTION:
	case FilmProperty::CONTAINER:
		update_video_card();
		break;
	case FilmProperty::VIDEO_FRAME_RATE:
		update_video_card();
		/* The credits timecode is displayed at the film's frame rate, and
		 * "at end of film" is a frame position -- both move with it. */
		update_credits();
		break;
	default:
		break;
	}
}


void
SlangSimplePanel::film_content_changed(ChangeType type, int)
{
	if (type != ChangeType::DONE) {
		return;
	}

	/* Any content property can move the audio (gain, mapping, a stream being
	 * identified by the examine job), so just re-read; the view is cheap and
	 * only repaints what changed. */
	content_layout_changed();
}


void
SlangSimplePanel::content_layout_changed()
{
	apply_simple_defaults();
	update_all();
	if (_active) {
		/* Only THIS screen has no Save button, which is why it writes
		 * metadata.xml on every content change.  Doing that while the full
		 * interface is showing clears Film::dirty(), and every save prompt
		 * there is guarded on it -- so a content edit made in the full
		 * interface would be silently committed and its "close without saving"
		 * prompt would never appear.  DCP-o-matic has no undo. */
		save();
	}
}


/** Write the project out.  The simplified interface has no Save button -- it is
 *  meant to be a screen you can close after pressing Create DCP -- so it saves
 *  whenever the content changes; otherwise a project built here and not
 *  exported would come back empty.  Failures are swallowed rather than
 *  interrupting the user mid-edit: "Create DCP" writes the metadata again
 *  through jobs_make_dcp(), which does report the error.
 */
void
SlangSimplePanel::save()
{
	if (!_film || !_film->directory() || !_film->dirty()) {
		return;
	}

	try {
		_film->write_metadata();
	} catch (std::exception&) {}
}


/** Subtitles added through this screen are meant to end up in the DCP, so tick
 *  the "use as open subtitles" box the full interface would make the user find.
 *  Only content this panel added is touched (tracked in _pending_subtitles), so
 *  a deliberate choice made in the full interface is never overridden.
 *
 *  The interface's other decision -- the smart-centre L/C/R mix for a
 *  mono/stereo source -- deliberately does NOT live here: it has to be in place
 *  before the audio measurement starts, which is a moment earlier than any UI
 *  handler runs.  See Film::maybe_smart_center_upmix().
 */
void
SlangSimplePanel::apply_simple_defaults()
{
	if (!_film || !_active || _applying_defaults) {
		return;
	}

	/* set_use() below emits ContentChange synchronously on this thread, and it
	 * comes straight back here through film_content_changed().  Iterate a COPY,
	 * so the nested call cannot shift or shrink the vector under this loop --
	 * it reassigns _pending_subtitles, which moves the survivors down and
	 * destroys the tail while this loop still holds its old begin/end -- and
	 * make the nested call a no-op so the outer result is the one that is kept.
	 * Without this, adding three subtitles at once silently dropped one: it was
	 * removed from the pending list without ever having set_use() called on it,
	 * so it was simply absent from the DCP. */
	_applying_defaults = true;
	dcp::ScopeGuard sg = [this]() { _applying_defaults = false; };
	auto const pending = _pending_subtitles;

	/* Subtitles added through this screen are meant to be in the DCP. */
	vector<weak_ptr<Content>> still_pending;
	for (auto weak: pending) {
		auto content = weak.lock();
		if (!content) {
			continue;
		}
		if (content->text.empty()) {
			/* Still being examined; try again when it reports back. */
			still_pending.push_back(weak);
			continue;
		}
		for (auto text: content->text) {
			if (!text->use()) {
				text->set_use(true);
				text->set_type(TextType::OPEN_SUBTITLE);
			}
		}
	}
	_pending_subtitles = still_pending;
}


void
SlangSimplePanel::job_added(weak_ptr<Job> weak)
{
	auto job = dynamic_pointer_cast<SlangAudioAnalyseJob>(weak.lock());
	/* Only this screen's film: JobAdded is global, and a job queued against
	 * another film (the batch converter, a second window) must not take over
	 * this one's Sound card. */
	if (!job || job->film() != _film) {
		return;
	}

	_analysis_job = job;
	_pipeline->set_analysing(true, {});
	_pipeline->set_measurement_note(wxEmptyString);
	_analysis_finished_connection.disconnect();
	boost::signals2::connection connection;
	job->when_finished(
		connection,
		boost::bind(&SlangSimplePanel::analysis_finished, this, _1, weak_ptr<SlangAudioAnalyseJob>(job))
		);
	_analysis_finished_connection = connection;
	if (!_analysis_timer.IsRunning()) {
		_analysis_timer.Start(ANALYSIS_POLL_MS);
	}
}


void
SlangSimplePanel::jobs_changed()
{
	/* ActiveJobsChanged can reach us while JobManager holds its own mutex --
	 * cancel_all_jobs() emits it from the UI thread inside the lock, and
	 * Signaller::emit() runs same-thread handlers immediately -- so anything
	 * that asks JobManager a question (work_to_do(), get()) has to wait for
	 * idle or it deadlocks.  DOMFrame::active_jobs_changed() defends its own
	 * menu update the same way, for the same reason. */
	signal_manager->when_idle(boost::bind(&SlangSimplePanel::update_action_row, this));
}


void
SlangSimplePanel::poll_analysis()
{
	auto job = _analysis_job.lock();
	/* Keyed on the FILM as well as the job: a job outliving the film it measured
	 * must not drive this screen (see set_film()).  Job::film() is the same
	 * weak-pointer comparison the export chain already makes. */
	if (!job || job->finished() || job->film() != _film) {
		_analysis_timer.Stop();
		_pipeline->set_analysing(false, {});
		return;
	}

	_pipeline->set_analysing(true, job->progress());
}


void
SlangSimplePanel::analysis_finished(Job::Result result, weak_ptr<SlangAudioAnalyseJob> weak)
{
	auto job = weak.lock();
	if (!job || job != _analysis_job.lock() || job->film() != _film) {
		/* A superseded run (a second content add restarts the analysis), or one
		 * belonging to a film this screen has moved on from; whatever the
		 * current film's current run reports is what counts. */
		return;
	}

	_analysis_timer.Stop();
	_analysis_job.reset();
	_pipeline->set_analysing(false, {});

	if (result == Job::Result::RESULT_CANCELLED) {
		_pipeline->set_measurement_note(wxEmptyString);
		return;
	}

	if (result != Job::Result::RESULT_OK) {
		_pipeline->set_measurement_note(
			_("The sound could not be measured. Check that the GPU frame server is running.")
			);
		_pipeline->refresh_state();
		return;
	}

	/* Say where the number came from.  "Measured on the GPU" is only claimed
	 * when the server's answer was also checked against the local ground truth
	 * -- an answer is not a measurement (see SlangAudioAnalyseJob).
	 *
	 * The no-measurement case has to come FIRST, before any of the provenance
	 * branches.  A job that refuses an out-of-range peak still finishes OK, and
	 * both used_gpu() and peak_verified() are true on the way there -- the
	 * server answered, and it agreed, which is HOW the job knew the peak was
	 * unusable.  Reading provenance first therefore labelled a refusal
	 * "Measured on the GPU and cross-checked" directly under a gain box reading
	 * "Not measured yet". */
	if (job->no_measurement()) {
		_pipeline->set_measurement_note(
			_("The sound level was outside the usable range, so it was left alone -- see Progress.")
			);
		_pipeline->refresh_state();
		signal_manager->when_idle(boost::bind(&SlangSimplePanel::update_action_row, this));
		return;
	}

	if (job->cache_hit()) {
		_pipeline->set_measurement_note(_("Sound unchanged since the last measurement."));
	} else if (job->used_gpu() && job->peak_verified()) {
		_pipeline->set_measurement_note(_("Measured on the GPU and cross-checked."));
	} else if (job->used_gpu()) {
		_pipeline->set_measurement_note(_("The GPU's answer disagreed with the local check, so the local measurement was used."));
	} else {
		_pipeline->set_measurement_note(_("Measured locally (the GPU frame server was not available)."));
	}

	_pipeline->refresh_state();

	/* NOT update_action_row() directly.  It walks JobManager::get() calling
	 * finished() on every job, so it must not run with any JobManager or Job
	 * lock held.  Job::when_finished() no longer holds the job's _state_mutex
	 * when it calls this handler (it did once, and that deadlocked the UI
	 * thread against itself), but jobs_changed() below reaches
	 * update_action_row() from ActiveJobsChanged, which JobManager DOES emit
	 * under its own _mutex.  Deferring keeps both entry points off both locks,
	 * and keeps us out of the reverse of the order the job scheduler takes them
	 * (JobManager::_mutex then Job::_state_mutex). */
	signal_manager->when_idle(boost::bind(&SlangSimplePanel::update_action_row, this));
}


bool
SlangSimplePanel::ensure_film(boost::filesystem::path const& first_content)
{
	if (_film) {
		return true;
	}

	/* No project yet: put one beside the user's other films, named after what
	 * they just added, so nothing has to be answered before the first video
	 * can go in.  The output card shows where that landed and offers to move
	 * it. */
	optional<boost::filesystem::path> path;
	try {
		auto const base = Config::instance()->default_directory_or(
			wx_to_std(wxStandardPaths::Get().GetDocumentsDir())
			);

		auto stem = first_content.stem().string();
		if (stem.empty()) {
			stem = "DCP";
		}

		path = unique_child(base, stem);
		if (!path) {
			error_dialog(this, _("Could not find a free folder name for the new project."));
			return false;
		}
	} catch (std::exception& e) {
		/* Reading the default directory can fail -- an unreadable share, a dead
		 * NFS mount.  Uncaught, that reaches wxApp::OnExceptionInMainLoop, which
		 * reports it and then TERMINATES the program: the user loses whatever
		 * they were doing because a folder could not be stat'd. */
		error_dialog(this, _("A folder for the new project could not be chosen."), std_to_wx(e.what()));
		return false;
	}

	NewFilm(*path);
	return static_cast<bool>(_film);
}


void
SlangSimplePanel::add_paths(vector<boost::filesystem::path> paths, bool as_subtitles)
{
	if (paths.empty()) {
		return;
	}

	std::sort(paths.begin(), paths.end());

	/* Not while a DCP is being written.  The buttons and drop areas are already
	 * disabled for the duration (set_general_sensitivity), but a drop can arrive
	 * from a drag begun before the export started, and this is the function that
	 * actually mutates the playlist the running Player is reading. */
	if (export_in_flight()) {
		error_dialog(this, _("Your DCP is being made.  Wait for it to finish before changing the content."));
		return;
	}

	/* Examine the files BEFORE inventing a project for them.  ensure_film()
	 * creates a directory on disk and binds this panel to a new film, so doing
	 * it first meant that dropping a folder of documents left an orphan,
	 * misnamed project behind -- one that every later legitimate drop then
	 * landed in -- underneath an error message saying nothing usable was found.
	 * content_factory() needs no film. */
	vector<shared_ptr<Content>> content;
	try {
		for (auto const& path: paths) {
			for (auto piece: content_factory(path)) {
				content.push_back(piece);
			}
		}
	} catch (std::exception& e) {
		error_dialog(this, std_to_wx(e.what()));
		return;
	}

	if (content.empty()) {
		error_dialog(this, _("Nothing usable was found in that file."));
		return;
	}

	if (!ensure_film(paths.front())) {
		return;
	}

	if (as_subtitles) {
		for (auto piece: content) {
			_pending_subtitles.push_back(piece);
		}
	}

	/* Name the film after the first video that goes into it, so the DCP is not
	 * called after whatever folder we happened to invent. */
	if (!as_subtitles && _film->content().empty()) {
		auto const stem = paths.front().stem().string();
		if (!stem.empty()) {
			_film->set_name(stem);
		}
	}

	try {
		_film->examine_and_add_content(content);
	} catch (std::exception& e) {
		error_dialog(this, std_to_wx(e.what()));
	}
}


void
SlangSimplePanel::video_dropped(vector<boost::filesystem::path> paths)
{
	if (paths.empty()) {
		choose_video();
		return;
	}

	/* A drop is not curated, so route each file to the step it belongs to
	 * rather than refusing the whole thing. */
	vector<boost::filesystem::path> subtitles;
	vector<boost::filesystem::path> rest;
	for (auto const& path: paths) {
		(is_subtitle_path(path) ? subtitles : rest).push_back(path);
	}

	add_paths(rest, false);
	add_paths(subtitles, true);
}


void
SlangSimplePanel::subtitles_dropped(vector<boost::filesystem::path> paths)
{
	if (paths.empty()) {
		choose_subtitles();
		return;
	}
	add_paths(paths, true);
}


/** Where a file dialog should start, for a screen whose project does not exist
 *  yet.  dcpomatic::film::add_files_override_path() dereferences the film it is
 *  given -- every other caller only ever has a saved project -- but here the
 *  film is not created until the first content is chosen (ensure_film()), so
 *  the very first click has nothing to pass it.  With no project there is no
 *  project folder to open in either, so an unset override (the "same as last
 *  time" behaviour) is also the right answer, not just the safe one.
 */
static optional<boost::filesystem::path>
start_directory_for(shared_ptr<const Film> film)
{
	return film ? dcpomatic::film::add_files_override_path(film) : optional<boost::filesystem::path>();
}


void
SlangSimplePanel::choose_video()
{
	FileDialog dialog(
		this,
		_("Choose your video"),
		char_to_wx("All files|*.*|Video files|*.mp4;*.mov;*.mkv;*.mxf;*.avi;*.m2ts;*.mpg;*.mpeg;*.webm;*.dpx;*.tif;*.tiff|Sound files|*.wav;*.w64;*.flac;*.aif;*.aiff"),
		wxFD_MULTIPLE | wxFD_CHANGE_DIR,
		"AddFilesPath",
		{},
		start_directory_for(_film)
		);

	if (dialog.show()) {
		video_dropped(dialog.paths());
	}
}


/** The "Replace..." button, which has to REPLACE.
 *
 *  It used to be wired straight to choose_video(), so it appended: the film kept
 *  the old cut FIRST (Playlist::add_at_end positions the new one after it), the
 *  card read "holiday.mp4 (+1 more)", the DCP name kept the old stem because
 *  add_paths() only renames a film whose content is empty, and Create DCP made a
 *  22-minute DCP out of two takes of the same film.  The only way back was
 *  Remove, which deletes everything.
 *
 *  Nothing is removed until the user has confirmed a replacement, so cancelling
 *  the dialog leaves the project exactly as it was.
 */
void
SlangSimplePanel::replace_video()
{
	if (!_film || export_in_flight()) {
		return;
	}

	FileDialog dialog(
		this,
		_("Choose the file to use instead"),
		char_to_wx("All files|*.*|Video files|*.mp4;*.mov;*.mkv;*.mxf;*.avi;*.m2ts;*.mpg;*.mpeg;*.webm;*.dpx;*.tif;*.tiff|Sound files|*.wav;*.w64;*.flac;*.aif;*.aiff"),
		wxFD_MULTIPLE | wxFD_CHANGE_DIR,
		"AddFilesPath",
		{},
		start_directory_for(_film)
		);

	if (!dialog.show()) {
		return;
	}

	auto const paths = dialog.paths();
	if (paths.empty()) {
		return;
	}

	remove_video();
	video_dropped(paths);
}


void
SlangSimplePanel::choose_subtitles()
{
	FileDialog dialog(
		this,
		_("Choose your subtitles"),
		char_to_wx("Subtitle files|") + subtitle_wildcard() + char_to_wx("|All files|*.*"),
		wxFD_MULTIPLE | wxFD_CHANGE_DIR,
		"AddFilesPath",
		{},
		start_directory_for(_film)
		);

	if (dialog.show()) {
		add_paths(dialog.paths(), true);
	}
}


void
SlangSimplePanel::remove_video()
{
	if (!_film || export_in_flight()) {
		return;
	}

	/* Everything the Video card accepts, not only content that has a picture.
	 * choose_video()'s dialog offers sound files, so a .wav dropped there became
	 * content that NEITHER card listed and NEITHER Remove reached: it was welded
	 * to the project for good, it named the film and therefore the DCP, and it
	 * was mixed into the delivered soundtrack.  The card's own subtitle already
	 * says "The picture and sound your DCP is made from". */
	for (auto content: _film->content()) {
		if (is_video_card_content(content)) {
			_film->remove_content(content);
		}
	}

	update_all();
}


void
SlangSimplePanel::choose_subtitle_language(weak_ptr<Content> weak)
{
	auto content = weak.lock();
	if (!_film || !content || content->text.empty()) {
		return;
	}

	auto const current = content->text.front()->language();
	LanguageTagDialog dialog(this, current.get_value_or(dcp::LanguageTag("en")));
	auto const result = dialog.ShowModal();
	/* See choose_audio_language(): force a repaint rather than trust the
	 * toolkit to redraw what this modal covered. */
	Refresh();
	Update();
	if (result != wxID_OK) {
		return;
	}

	/* Every text track in the file, not just the first: the file is one
	 * language whichever track of it the name is later built from. */
	for (auto text: content->text) {
		text->set_language(dialog.get());
	}

	update_subtitle_card();
	update_output_card();
	save();
}


void
SlangSimplePanel::remove_subtitle(weak_ptr<Content> weak)
{
	auto content = weak.lock();
	if (!_film || !content || export_in_flight()) {
		return;
	}

	_film->remove_content(content);
	update_all();
}


optional<boost::filesystem::path>
SlangSimplePanel::project_folder_for(boost::filesystem::path const& chosen) const
{
	/* The chosen folder IS the project folder when that is safe -- it does not
	 * exist yet, or it is empty -- and otherwise the project goes in a
	 * subfolder named after the film.
	 *
	 * The difference is not cosmetic.  A project folder holds metadata.xml, the
	 * log, the analysis cache and the finished DCP, and change_output_folder()
	 * copies all of that into the destination and then DELETES the old folder.
	 * Pointed at a folder the user actually keeps things in -- Documents, a
	 * drive root -- it would scatter those files among their own and leave
	 * nothing to undo it from.  ensure_film() invents <base>/<name> when it
	 * creates a project for exactly the same reason; this keeps "change it
	 * later" behaving like "choose it the first time".
	 */
	if (!dcp::filesystem::exists(chosen)) {
		return chosen;
	}

	if (boost::filesystem::is_directory(chosen) && boost::filesystem::directory_iterator(chosen) == boost::filesystem::directory_iterator()) {
		return chosen;
	}

	auto stem = _film ? _film->name() : string();
	boost::algorithm::trim(stem);
	if (stem.empty()) {
		stem = "DCP";
	}

	/* boost::none if every candidate is taken -- see unique_child().  The caller
	 * MUST NOT fall back to a colliding path: it hands what it gets here to a
	 * copy whose failure rollback does remove_all() on the target. */
	return unique_child(chosen, stem);
}


void
SlangSimplePanel::change_output_folder()
{
	/* EVERYTHING here is inside the catch, the dialog included.  An exception
	 * that escapes a UI handler reaches wxApp::OnExceptionInMainLoop, which
	 * reports it and then TERMINATES the program -- losing whatever the user
	 * was doing over a folder that could not be read.  Choosing a folder must
	 * be able to fail without taking the application with it. */

	/* However this returns -- normally, by an early return, or by the catch at
	 * the bottom -- the cards must end up describing the film's REAL state.  The
	 * bare update_all() at the end of the happy path did not: a remove_all()
	 * that threw AFTER the project had already been repointed jumped straight to
	 * the catch, so the output card went on showing the old, now-deleted folder
	 * under a dialog saying the move had failed. */
	dcp::ScopeGuard refresh([this]() { update_all(); });

	try {
		/* Not while anything is still writing into the project.  This function
		 * copies the folder and then DELETES the original, and DCP-o-matic runs
		 * background jobs against a path each one snapshotted when it started --
		 * the audio analysis writes its cache into <project>/analysis at the end
		 * of a run that takes minutes on a feature.  Pulling the directory out
		 * from under them strands the output and fails the job with a filesystem
		 * error the user cannot act on.  Checked here rather than by grey-ing the
		 * button, because a job can start between the last repaint and the click
		 * -- and because deriving it in the sensitivity path would mean walking
		 * JobManager under its own lock (see export_in_flight()). */
		auto jobs = JobManager::instance()->get();
		auto const jobs_busy = std::any_of(
			jobs.begin(), jobs.end(),
			[](shared_ptr<const Job> job) { return !job->finished(); });
		if (jobs_busy) {
			error_dialog(
				this,
				_("Wait for the jobs in Progress to finish before moving the project.")
				);
			return;
		}

		DirDialog dialog(this, _("Choose the folder for your DCP"), wxDD_DEFAULT_STYLE, "SlangSimpleOutput");
		if (!dialog.show()) {
			return;
		}

		if (dialog.path().empty()) {
			return;
		}

		/* Resolve before ANY comparison below.  Film's own directory is
		 * canonical (its constructor runs weakly_canonical) but the dialog's is
		 * not, so a symlinked route into the project's own subtree would pass
		 * every guard here -- and boost::filesystem::copy has no cycle
		 * detection, so it re-copies each level's earlier siblings on the way
		 * down (measured: a 1.5 MB project became 292 MB across 581 directories
		 * in 0.14 s before ENAMETOOLONG stopped it). */
		auto const chosen = dcp::filesystem::weakly_canonical(dialog.path());

		optional<boost::filesystem::path> current;
		if (_film) {
			if (auto directory = _film->directory()) {
				current = dcp::filesystem::weakly_canonical(*directory);
			}
		}
		if (current && dcp::filesystem::exists(*current) && dcp::filesystem::exists(chosen)
		    && boost::filesystem::equivalent(*current, chosen)) {
			return;
		}

		if (dcp::filesystem::exists(chosen / "metadata.xml")) {
			error_dialog(this, _("There is already a project in that folder.  Choose an empty folder instead."));
			return;
		}

		auto const maybe_target = project_folder_for(chosen);
		if (!maybe_target) {
			error_dialog(this, _("Could not find a free folder name inside that folder."));
			return;
		}
		auto const target = *maybe_target;

		if (!_film) {
			NewFilm(target);
			return;
		}

		/* Content stored INSIDE the project is about to be copied to a new
		 * location and then deleted from the old one, while metadata.xml goes on
		 * naming the old absolute path.  The result reopens as a project whose
		 * video "exists" on the card and cannot be opened -- so refuse, and say
		 * which files are in the way, rather than silently producing it.
		 * (Rewriting the paths is the other possible answer; refusing is the one
		 * that cannot get a path wrong.) */
		if (current) {
			auto const under = [](boost::filesystem::path const& child,
					      boost::filesystem::path const& parent) {
				boost::system::error_code ec;
				for (auto p = child.parent_path(); !p.empty() && p != p.parent_path();
				     p = p.parent_path()) {
					if (boost::filesystem::exists(p, ec)
					    && boost::filesystem::equivalent(p, parent, ec) && !ec) {
						return true;
					}
				}
				return false;
			};
			vector<string> inside_names;
			for (auto content: _film->content()) {
				for (auto const& p: content->paths()) {
					if (under(p, *current)) {
						inside_names.push_back(p.filename().string());
						break;
					}
				}
			}
			if (!inside_names.empty()) {
				error_dialog(
					this,
					_("Some of your files are stored inside the project folder, so it cannot be moved. "
					  "Move them somewhere else first, then add them again."),
					std_to_wx(boost::algorithm::join(inside_names, ", "))
					);
				return;
			}
		}

		/* The project folder IS the output folder (the DCP is written inside
		 * it), so moving where the DCP goes means moving the project.  Carry
		 * the work already done -- the examine caches, the audio analysis, the
		 * log -- rather than pointing the film at an empty folder and silently
		 * redoing all of it. */
		if (current && dcp::filesystem::exists(*current)) {
			/* Copying a directory into its own subtree never terminates.
			 * Compare by filesystem IDENTITY, not by name: canonicalising above
			 * handles symlinks, but a bind mount or a case-insensitive
			 * filesystem can still name one directory two ways.  The tree is a
			 * few levels deep, so the extra stats cost nothing. */
			auto const inside = [](boost::filesystem::path const& child, boost::filesystem::path const& parent) {
				boost::system::error_code ec;
				for (auto p = child; !p.empty() && p != p.parent_path(); p = p.parent_path()) {
					if (dcp::filesystem::exists(p) && boost::filesystem::equivalent(p, parent, ec) && !ec) {
						return true;
					}
				}
				return false;
			};
			if (inside(target, *current)) {
				error_dialog(this, _("That folder is inside the project folder.  Choose one outside it."));
				return;
			}

			/* Ask before starting a copy that cannot fit, rather than grinding
			 * through tens of GB and failing at the end.  A warning, not a
			 * refusal: a compressing or de-duplicating filesystem (btrfs, ZFS)
			 * can hold more than the arithmetic allows.  Mirrors the disk-space
			 * confirm that jobs_make_dcp() already shows. */
			{
				boost::system::error_code ec;
				auto const space = boost::filesystem::space(chosen, ec);
				if (!ec) {
					uintmax_t needed = 0;
					boost::system::error_code we;
					boost::filesystem::recursive_directory_iterator it(*current, we), last;
					for (; !we && it != last; it.increment(we)) {
						/* file_size() returns uintmax_t(-1) on failure, so
						 * summing it unchecked turns one file that vanished
						 * between the stat and the size -- Hints writes and
						 * deletes scratch files inside the project from its own
						 * thread -- into a demand for 18 exabytes, and the user
						 * cancels a move that would have worked.  Only add what
						 * was actually measured. */
						boost::system::error_code fe;
						if (!boost::filesystem::is_regular_file(it->status(fe)) || fe) {
							continue;
						}
						auto const size = boost::filesystem::file_size(it->path(), fe);
						if (!fe) {
							needed += size;
						}
					}
					if (needed > space.available) {
						if (!confirm_dialog(
							    this,
							    wxString::Format(
								    _("This project needs about %.1f GB and that folder only has %.1f GB free.  Do you want to try anyway?"),
								    static_cast<double>(needed) / 1e9,
								    static_cast<double>(space.available) / 1e9
								    ))) {
							return;
						}
					}
				}
			}

			/* A project that has already been exported carries its DCP, so this
			 * can be tens of GB; it runs on the UI thread, so at least stop the
			 * window looking dead while it does. */
			wxBusyCursor busy;

			auto const target_existed = dcp::filesystem::exists(target);

			/* No overwrite_existing: `target` is empty or newly invented, so a
			 * name that collides means an assumption is wrong, and throwing
			 * (caught below, reported) beats silently replacing a file. */
			try {
				copy_tree(*current, target);
			} catch (...) {
				/* project_folder_for() guarantees `target` was empty or invented,
				 * so everything in it now is what this copy just wrote.  Leaving
				 * a half-finished tree strands however many GB it got through --
				 * and the next attempt then reports the debris as somebody's
				 * project ("there is already a project in that folder"). */
				boost::system::error_code ec;
				boost::filesystem::remove_all(target, ec);
				if (target_existed) {
					boost::filesystem::create_directory(target, ec);
				}
				throw;
			}
			_film->set_directory(target);
			_film->write_metadata();
			/* Only now is the original expendable.  Deleting it on the strength
			 * of a copy nobody checked is how a project goes missing. */
			if (dcp::filesystem::exists(target / "metadata.xml")) {
				/* error_code, NOT a throw.  By this line the project HAS
				 * moved -- the film is repointed and its metadata is written
				 * at the target -- so a delete that fails (a file held open
				 * on Windows, a non-writable parent, NFS ESTALE) is a leftover
				 * copy, not a failed move.  Throwing here reported "the
				 * project could not be moved" about a move that had already
				 * succeeded, and left the user hunting for it. */
				boost::system::error_code de;
				boost::filesystem::remove_all(*current, de);
				if (de) {
					std::cerr << "dcpomatic2: change_output_folder: could not remove "
						  << current->string() << ": " << de.message() << "\n";
					error_dialog(
						this,
						_("Your project was moved, but the old folder could not be deleted.  "
						  "You can remove it yourself when you like."),
						std_to_wx(current->string())
						);
				}
			}
		} else {
			_film->set_directory(target);
			_film->write_metadata();
		}

		/* The project now lives somewhere else, and File -> Open recent still
		 * points at a path that no longer exists -- which the next Config write
		 * silently prunes, so the project vanishes from the list under both
		 * names.  This screen has no menu bar and no open control, so the only
		 * way back would be Advanced and browsing by hand.  add_to_history()
		 * de-duplicates, and `target` is what project_folder_for() returned (the
		 * dialog's path may have gained a subfolder). */
		Config::instance()->add_to_history(target);
	} catch (std::exception& e) {
		std::cerr << "dcpomatic2: change_output_folder: " << e.what() << "\n";
		error_dialog(this, _("The project could not be moved to that folder."), std_to_wx(e.what()));
	} catch (...) {
		std::cerr << "dcpomatic2: change_output_folder: unknown exception\n";
		error_dialog(this, _("The project could not be moved to that folder."));
	}
}


void
SlangSimplePanel::copy_output_path()
{
	/* update_output_card() keeps the button disabled whenever this is not
	 * true, but a click already queued before that runs (or a caller that
	 * bypasses the button) must not put an empty string on the clipboard. */
	auto const directory = _film ? _film->directory() : optional<boost::filesystem::path>();
	if (!directory) {
		return;
	}

	if (!wxTheClipboard->Open()) {
		return;
	}

	dcp::ScopeGuard sg = []() {
		wxTheClipboard->Close();
	};

	wxTheClipboard->SetData(new wxTextDataObject(std_to_wx(directory->string())));
}


void
SlangSimplePanel::create_dcp()
{
	if (!_film || _film->content().empty()) {
		return;
	}

	/* A feature CPL with no end-credit markers is two SMPTE Bv2.1 errors, and
	 * the DCP is otherwise perfectly good -- so warn rather than refuse, and
	 * let the user go ahead knowingly.  This is the LAST point at which the
	 * question can still be answered cheaply; after the export it costs another
	 * transcode. */
	if (needs_credit_markers() && !_film->marker(dcp::Marker::FFEC)) {
		auto const proceed = confirm_dialog(
			this,
			_("This DCP is a feature, but you have not said where its end credits start.\n\n"
			  "A feature DCP is required to carry end-credit markers (FFEC and FFMC); "
			  "without them it will be reported as non-compliant by DCP verification tools "
			  "and may be rejected by a cinema or QC house.\n\n"
			  "If the film has no separate end credits, press \"At end of film\" next to "
			  "\"End credits start at\" before making the DCP.\n\n"
			  "Make the DCP anyway?")
			);
		if (!proceed) {
			return;
		}
	}

	MakeDCP();
}


void
SlangSimplePanel::update_all()
{
	update_video_card();
	update_subtitle_card();
	update_output_card();
	update_content_type();
	update_credits();
	update_audio_language();
	update_action_row();
	if (_pipeline) {
		_pipeline->refresh_state();
	}
	if (_scroller) {
		_scroller->Layout();
		_scroller->FitInside();
	}
}


void
SlangSimplePanel::update_video_card()
{
	if (!_video_card) {
		return;
	}

	shared_ptr<Content> video;
	int extra = 0;
	if (_film) {
		for (auto content: _film->content()) {
			if (content->video) {
				if (video) {
					++extra;
				} else {
					video = content;
				}
			}
		}
	}

	_video_card->set_done(static_cast<bool>(video));
	_video_card->set_active(!video);

	if (!video) {
		_video_drop->Show();
		_video_details->Hide();
		_video_card->Layout();
		return;
	}

	_video_drop->Hide();
	_video_details->Show();

	auto name = std_to_wx(video->path(0).filename().string());
	if (extra > 0) {
		name += wxString::Format(_(" (+%d more)"), extra);
	}
	_video_name->SetLabel(name);

	wxString summary;
	if (auto const size = video->video->size()) {
		summary += wxString::Format(char_to_wx("%d×%d"), size->width, size->height);
	}
	if (auto const rate = video->video_frame_rate()) {
		if (!summary.IsEmpty()) {
			summary += char_to_wx(" · ");
		}
		summary += wxString::Format(_("%.2f fps"), *rate);
	}
	if (_film->length().get() > 0) {
		if (!summary.IsEmpty()) {
			summary += char_to_wx(" · ");
		}
		summary += time_to_timecode(_film->length(), _film->video_frame_rate());
	}
	_video_summary->SetLabel(summary);

	auto const slang = Config::instance()->slang();
	auto const rate_mbps = _film->video_bit_rate(VideoEncoding::JPEG2000) / 1000000;
	_video_encoding->SetLabel(
		wxString::Format(
			_("DCP: %d x %d, %d Mb/s, GPU %s coder"),
			_film->frame_size().width, _film->frame_size().height,
			static_cast<int>(rate_mbps),
			slang.coder == "mq" ? char_to_wx("MQ") : char_to_wx("HT")
			)
		);

	_video_card->Layout();
}


void
SlangSimplePanel::update_subtitle_card()
{
	if (!_subtitle_card) {
		return;
	}

	vector<shared_ptr<Content>> subtitles;
	if (_film) {
		for (auto content: _film->content()) {
			if (!content->text.empty() && !content->video) {
				subtitles.push_back(content);
			}
		}
	}

	_subtitle_card->set_done(!subtitles.empty());

	/* Drop the borrowed pointers BEFORE Clear(true) destroys what they point
	 * at, so nothing can reach a freed button in between. */
	_subtitle_row_buttons.clear();
	_subtitle_list_sizer->Clear(true);
	if (subtitles.empty()) {
		_subtitle_list->Hide();
		_subtitle_card->Layout();
		return;
	}

	auto const p = slang_ui::palette();
	for (auto content: subtitles) {
		auto row = new wxBoxSizer(wxHORIZONTAL);
		auto label = new wxStaticText(_subtitle_list, wxID_ANY, std_to_wx(content->path(0).filename().string()));
		label->SetFont(slang_ui::font(_subtitle_list, -1));
		label->SetForegroundColour(p.text);
		row->Add(label, 1, wxALIGN_CENTRE_VERTICAL);

		weak_ptr<Content> weak = content;

		/* The language OF THIS FILE, asked per file because that is where
		 * TextContent stores it and where the timed-text asset reads it from.
		 *
		 * It is NOT one language per file in the finished DCP, and the comment
		 * here used to imply it was: every used OPEN subtitle merges into a
		 * single timed-text asset, and the DCP name carries only
		 * open_text_languages().first -- so with two files in different
		 * languages the name names one of them and the asset is tagged with one
		 * of them.  (Nor is the lower-casing per track: DCNC lower-cases the
		 * subtitle field only when every used open text is BURNT IN, and appends
		 * -OCAP/-CCAP for captions.)  Asking per file is still right -- it is
		 * the only place the answer can be stored -- but a multi-language
		 * package needs closed-caption tracks, which the full interface does.
		 *
		 * Same CallAfter discipline as Remove below -- picking a language
		 * rebuilds this list, which would free this button while wx is still
		 * dispatching its click. */
		auto const language = content->text.empty() ? optional<dcp::LanguageTag>() : content->text.front()->language();
		auto language_button = new SlangFlatButton(
			_subtitle_list,
			language ? std_to_wx(language->as_string()) : wxString(_("Set language...")),
			SlangFlatButton::Kind::GHOST
			);
		language_button->SetToolTip(_("The language of these subtitles; it becomes part of the DCP's name."));
		language_button->on_click([this, weak]() {
			CallAfter([this, weak]() { choose_subtitle_language(weak); });
		});
		language_button->Enable(_sensitive);
		_subtitle_row_buttons.push_back(language_button);
		row->Add(language_button, 0, wxALIGN_CENTRE_VERTICAL);

		auto remove = new SlangFlatButton(_subtitle_list, _("Remove"), SlangFlatButton::Kind::GHOST);
		/* CallAfter, not a direct call: removing the content rebuilds this list,
		 * and _subtitle_list_sizer->Clear(true) deletes these buttons -- via
		 * wxSizerItem::DeleteWindows() -> wxWindow::Destroy(), which for a
		 * non-top-level window is a plain "delete this", not a deferred one.
		 * Called straight from here that frees the button while wxWidgets is
		 * still dispatching that button's own wxEVT_BUTTON, and the dispatcher
		 * keeps using the wxEvtHandler after we return.  Deferring lets the
		 * event finish first.  (The video card's buttons are built once and only
		 * shown or hidden, so they do not need this.) */
		remove->on_click([this, weak]() { CallAfter([this, weak]() { remove_subtitle(weak); }); });
		/* Tracked alongside the language button so an export can grey it too --
		 * removing content while a DCP is being written truncates it. */
		remove->Enable(_sensitive);
		_subtitle_row_buttons.push_back(remove);
		row->Add(remove, 0, wxALIGN_CENTRE_VERTICAL);

		_subtitle_list_sizer->Add(row, 0, wxEXPAND | wxBOTTOM, FromDIP(2));
	}

	_subtitle_list->Show();
	_subtitle_list->Layout();
	_subtitle_card->Layout();
}


/** The output "Change..." button's enabled state, computed in ONE place.
 *
 *  It used to be computed in two, with different terms: update_output_card()
 *  said `_sensitive`, set_general_sensitivity() said `_sensitive && _film`.
 *  Whichever ran last won, and on the normal first-run path (simple_ui is
 *  persisted, so the panel comes up before any film exists) the film-less one
 *  ran last -- leaving a dead, greyed button under a card that says "A folder
 *  is picked for you when you add a video; change it here at any time", and
 *  making change_output_folder()'s deliberate `if (!_film)` branch unreachable.
 *
 *  `_sensitive` alone is the correct predicate: the film-less case is designed
 *  for, and an export in flight is what _sensitive already tracks.
 */
void
SlangSimplePanel::update_output_change_enabled()
{
	if (_output_change) {
		_output_change->Enable(_sensitive);
	}
}


void
SlangSimplePanel::update_output_copy_path_enabled()
{
	if (_output_copy_path) {
		/* Unlike _output_change (see update_output_change_enabled()'s comment),
		 * there is nothing useful this button can do with no directory -- so,
		 * unlike that one, its predicate DOES include the directory's presence. */
		auto const directory = _film ? _film->directory() : optional<boost::filesystem::path>();
		_output_copy_path->Enable(_sensitive && static_cast<bool>(directory));
	}
}


void
SlangSimplePanel::update_output_card()
{
	if (!_output_card) {
		return;
	}

	auto const directory = _film ? _film->directory() : optional<boost::filesystem::path>();

	_output_card->set_done(static_cast<bool>(directory));
	update_output_change_enabled();
	update_output_copy_path_enabled();

	if (!directory) {
		_output_path->SetLabel(_("Not chosen yet"));
		_output_dcp->SetLabel(_("A folder is picked for you when you add a video; change it here at any time."));
		_output_card->Layout();
		return;
	}

	_output_path->SetLabel(std_to_wx(directory->string()));
	try {
		_output_dcp->SetLabel(
			wxString::Format(_("The DCP will be created as \"%s\" inside it."), std_to_wx(_film->dcp_name(true)))
			);
	} catch (std::exception&) {
		/* dcp_name() derives an ISDCF name from settings that may not be
		 * complete yet; a name we cannot compute is not worth an error. */
		_output_dcp->SetLabel(wxEmptyString);
	}

	_output_card->Layout();
}


void
SlangSimplePanel::update_action_row()
{
	if (!_create) {
		return;
	}

	/* A VIDEO, not just any content.  Removing the video from a film that still
	 * holds an .srt left this row saying "Everything is ready." beside a live
	 * Create DCP button, directly under a Video card that had gone back to
	 * "Drop a video file here" -- and pressing it queued a real transcode of a
	 * picture-less film, whose length came from the subtitle timings. */
	auto has_video = false;
	/* A subtitle that will be in the DCP but has no language is a Bv2.1 ERROR
	 * (libdcp MISSING_SUBTITLE_LANGUAGE), and it also makes the DCP's own name
	 * deny that the subtitles exist -- the ISDCF field falls back to "-XX", so a
	 * QC tool reports the name and the package contradicting each other.  It is
	 * one click to fix, on a button already sitting in the Subtitles card, so
	 * ask for it rather than shipping a package that fails verification. */
	auto subtitles_need_language = false;
	if (_film) {
		for (auto content: _film->content()) {
			if (content->video) {
				has_video = true;
			}
			for (auto text: content->text) {
				if (text->use() && !text->language()) {
					subtitles_need_language = true;
				}
			}
		}
	}

	/* Only wait for the jobs whose result the export depends on: content still
	 * being examined (its streams and length are not known yet) and the GPU
	 * sound measurement (whose gain has not been applied yet).  DCP-o-matic's
	 * own waveform analysis also runs on import and can take minutes on a
	 * feature; it feeds a display this screen does not show, so blocking the
	 * one button on it would be a long, unexplained wait for nothing. */
	auto busy = false;
	for (auto job: JobManager::instance()->get()) {
		if (job->finished()) {
			continue;
		}
		if (dynamic_pointer_cast<ExamineContentJob>(job) || dynamic_pointer_cast<SlangAudioAnalyseJob>(job)) {
			busy = true;
			break;
		}
	}

	_create->Enable(_sensitive && has_video && !busy && !subtitles_need_language);

	if (!_sensitive) {
		/* A disabled button always gets a reason.  During an export this row
		 * used to read "Everything is ready." beside a dead button. */
		_create_note->SetLabel(_("Your DCP is being made - see Progress below."));
	} else if (!has_video) {
		_create_note->SetLabel(_("Add a video to get started."));
	} else if (subtitles_need_language) {
		_create_note->SetLabel(_("Set the language of your subtitles above."));
	} else if (busy) {
		_create_note->SetLabel(_("Checking your video and measuring the sound..."));
	} else {
		_create_note->SetLabel(_("Everything is ready."));
	}

	if (_create_note->GetContainingSizer()) {
		_create_note->GetContainingSizer()->Layout();
	}
}

#endif
