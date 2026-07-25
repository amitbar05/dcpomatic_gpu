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

#include "dir_dialog.h"
#include "file_dialog.h"
#include "job_manager_view.h"
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


static bool
is_subtitle_path(boost::filesystem::path const& path)
{
	auto extension = boost::algorithm::to_lower_copy(path.extension().string());
	return extension == ".srt" || extension == ".ssa" || extension == ".ass"
		|| extension == ".vtt" || extension == ".stl" || extension == ".sub"
		|| extension == ".dfxp" || extension == ".ttml" || extension == ".xml"
		|| extension == ".fcpxml";
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

	auto replace = new SlangFlatButton(_video_details, _("Replace..."), SlangFlatButton::Kind::GHOST);
	replace->on_click([this]() { choose_video(); });
	details->Add(replace, 0, wxALIGN_CENTRE_VERTICAL);

	auto remove = new SlangFlatButton(_video_details, _("Remove"), SlangFlatButton::Kind::GHOST);
	remove->on_click([this]() { remove_video(); });
	details->Add(remove, 0, wxALIGN_CENTRE_VERTICAL);

	_video_details->SetSizer(details);
	_video_card->body()->Add(_video_details, 0, wxEXPAND);
	_video_details->Hide();
}


void
SlangSimplePanel::build_subtitle_card(wxWindow* parent, wxSizer* sizer)
{
	auto const p = slang_ui::palette();

	_subtitle_card = new SlangCard(parent, _("Subtitles"), _("Optional. SRT, ASS/SSA, VTT, STL or DCP subtitle XML."), 2);
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

	_output_card->body()->Add(row, 0, wxEXPAND);
}


void
SlangSimplePanel::build_audio_card(wxWindow* parent, wxSizer* sizer)
{
	_audio_card = new SlangCard(
		parent, _("Sound"),
		_("Measured on the GPU as soon as your video is added, then levelled for the cinema.")
		);
	sizer->Add(_audio_card, 0, wxEXPAND);

	_pipeline = new SlangAudioPipelineView(_audio_card);
	_audio_card->body()->Add(_pipeline, 1, wxEXPAND);
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
	auto jobs = JobManager::instance()->get();
	auto const transcoding = std::any_of(
		jobs.begin(),
		jobs.end(),
		[](shared_ptr<const Job> job) {
			return std::dynamic_pointer_cast<const DCPTranscodeJob>(job) && !job->finished();
		});
	if (transcoding) {
		return;
	}

	if (!_film->migrate_smart_center_mono_mapping()) {
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
	if (_output_change) {
		/* Moving the project out from under a running job would strand its
		 * output; the full interface disables the same class of control. */
		_output_change->Enable(sensitive && static_cast<bool>(_film));
	}
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
	case FilmProperty::VIDEO_BIT_RATE:
	case FilmProperty::VIDEO_FRAME_RATE:
	case FilmProperty::RESOLUTION:
	case FilmProperty::CONTAINER:
		update_video_card();
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
	if (!job) {
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
	if (!job || job->finished()) {
		_analysis_timer.Stop();
		return;
	}

	_pipeline->set_analysing(true, job->progress());
}


void
SlangSimplePanel::analysis_finished(Job::Result result, weak_ptr<SlangAudioAnalyseJob> weak)
{
	auto job = weak.lock();
	if (!job || job != _analysis_job.lock()) {
		/* A superseded run (a second content add restarts the analysis);
		 * whatever the current one reports is what counts. */
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
	 * -- an answer is not a measurement (see SlangAudioAnalyseJob). */
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
	auto const base = Config::instance()->default_directory_or(
		wx_to_std(wxStandardPaths::Get().GetDocumentsDir())
		);

	auto stem = first_content.stem().string();
	if (stem.empty()) {
		stem = "DCP";
	}

	auto path = base / stem;
	for (int i = 2; dcp::filesystem::exists(path) && i < 1000; ++i) {
		path = base / fmt::format("{} {}", stem, i);
	}

	NewFilm(path);
	return static_cast<bool>(_film);
}


void
SlangSimplePanel::add_paths(vector<boost::filesystem::path> paths, bool as_subtitles)
{
	if (paths.empty()) {
		return;
	}

	std::sort(paths.begin(), paths.end());

	if (!ensure_film(paths.front())) {
		return;
	}

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


void
SlangSimplePanel::choose_subtitles()
{
	FileDialog dialog(
		this,
		_("Choose your subtitles"),
		char_to_wx("Subtitle files|*.srt;*.ssa;*.ass;*.vtt;*.stl;*.sub;*.xml;*.dfxp;*.ttml;*.fcpxml|All files|*.*"),
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
	if (!_film) {
		return;
	}

	for (auto content: _film->content()) {
		if (content->video) {
			_film->remove_content(content);
		}
	}

	update_all();
}


void
SlangSimplePanel::remove_subtitle(weak_ptr<Content> weak)
{
	auto content = weak.lock();
	if (!_film || !content) {
		return;
	}

	_film->remove_content(content);
	update_all();
}


boost::filesystem::path
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

	auto path = chosen / stem;
	for (int i = 2; dcp::filesystem::exists(path) && i < 1000; ++i) {
		path = chosen / fmt::format("{} {}", stem, i);
	}

	return path;
}


void
SlangSimplePanel::change_output_folder()
{
	/* EVERYTHING here is inside the catch, the dialog included.  An exception
	 * that escapes a UI handler reaches wxApp::OnExceptionInMainLoop, which
	 * reports it and then TERMINATES the program -- losing whatever the user
	 * was doing over a folder that could not be read.  Choosing a folder must
	 * be able to fail without taking the application with it. */
	try {
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

		auto const target = project_folder_for(chosen);

		if (!_film) {
			NewFilm(target);
			update_all();
			return;
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
					for (auto& i: boost::filesystem::recursive_directory_iterator(*current)) {
						boost::system::error_code fe;
						if (boost::filesystem::is_regular_file(i.status())) {
							needed += boost::filesystem::file_size(i.path(), fe);
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
				boost::filesystem::remove_all(*current);
			}
		} else {
			_film->set_directory(target);
			_film->write_metadata();
		}

		update_all();
	} catch (std::exception& e) {
		std::cerr << "dcpomatic2: change_output_folder: " << e.what() << "\n";
		error_dialog(this, _("The project could not be moved to that folder."), std_to_wx(e.what()));
	} catch (...) {
		std::cerr << "dcpomatic2: change_output_folder: unknown exception\n";
		error_dialog(this, _("The project could not be moved to that folder."));
	}
}


void
SlangSimplePanel::create_dcp()
{
	if (!_film || _film->content().empty()) {
		return;
	}

	MakeDCP();
}


void
SlangSimplePanel::update_all()
{
	update_video_card();
	update_subtitle_card();
	update_output_card();
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

		auto remove = new SlangFlatButton(_subtitle_list, _("Remove"), SlangFlatButton::Kind::GHOST);
		weak_ptr<Content> weak = content;
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
		row->Add(remove, 0, wxALIGN_CENTRE_VERTICAL);

		_subtitle_list_sizer->Add(row, 0, wxEXPAND | wxBOTTOM, FromDIP(2));
	}

	_subtitle_list->Show();
	_subtitle_list->Layout();
	_subtitle_card->Layout();
}


void
SlangSimplePanel::update_output_card()
{
	if (!_output_card) {
		return;
	}

	auto const directory = _film ? _film->directory() : optional<boost::filesystem::path>();

	_output_card->set_done(static_cast<bool>(directory));
	_output_change->Enable(_sensitive);

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

	auto const has_content = _film && !_film->content().empty();

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

	_create->Enable(_sensitive && has_content && !busy);

	if (!has_content) {
		_create_note->SetLabel(_("Add a video to get started."));
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
