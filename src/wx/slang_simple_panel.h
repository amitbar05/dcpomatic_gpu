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


/** @file  src/wx/slang_simple_panel.h
 *  @brief SlangSimplePanel: the simplified "make me a DCP" interface.
 */


#ifndef DCPOMATIC_SLANG_SIMPLE_PANEL_H
#define DCPOMATIC_SLANG_SIMPLE_PANEL_H


#ifdef DCPOMATIC_SLANG


#include "lib/change_signaller.h"
#include "lib/film_property.h"
#include "lib/job.h"
#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
#include <wx/wx.h>
LIBDCP_ENABLE_WARNINGS
#include <boost/filesystem.hpp>
#include <boost/signals2.hpp>
#include <memory>
#include <vector>


class Choice;
class Content;
class Film;
class JobManagerView;
class SlangAudioAnalyseJob;
class SlangAudioPipelineView;
class SlangCard;
class SlangDropArea;
class SlangFlatButton;
class wxScrolledWindow;
class wxSizerItem;


/** @class SlangSimplePanel
 *  @brief The simplified interface: pick a video, optionally some subtitles,
 *  say where the DCP should go, watch what happens to the audio, press one
 *  button.
 *
 *  It drives the same Film and the same jobs the full interface does -- it is a
 *  different front end onto the existing project, not a parallel pipeline -- so
 *  a film built here opens unchanged in the full editor (View -> Simplified
 *  interface, or the Advanced button, switches between them at any time).
 *
 *  The audio analysis is not something this screen starts: Film::maybe_add_content
 *  already runs the GPU auto-gain pass the moment content lands, so by the time
 *  the video has finished being examined the measurement is usually already in
 *  flight.  This panel watches the JobManager for that job and mirrors it into
 *  the pipeline view.
 */
class SlangSimplePanel : public wxPanel
{
public:
	explicit SlangSimplePanel(wxWindow* parent);

	void set_film(std::shared_ptr<Film> film);

	/** Called when the panel becomes (or stops being) the visible interface. */
	void set_active(bool active);

	void set_general_sensitivity(bool sensitive);

	/** The user wants the full interface. */
	boost::signals2::signal<void ()> Advanced;
	/** The user pressed New: start a fresh project, asking them where it goes
	 *  and offering to save the current one first.  Distinct from @ref NewFilm,
	 *  which is this panel's own plumbing for a destination it has ALREADY
	 *  chosen; this one asks the host to run its full New Film flow, because
	 *  with no menu bar in this mode the button is the only way to reach it. */
	boost::signals2::signal<void ()> NewProject;
	/** Asks the host to create a film in this directory and hand it back via
	 *  set_film(); the panel needs a film before it can hold any content. */
	boost::signals2::signal<void (boost::filesystem::path)> NewFilm;
	/** The user pressed "Create DCP". */
	boost::signals2::signal<void ()> MakeDCP;

private:
	void build();
	wxWindow* build_header(wxWindow* parent);
	void build_video_card(wxWindow* parent, wxSizer* sizer);
	void build_subtitle_card(wxWindow* parent, wxSizer* sizer);
	void build_output_card(wxWindow* parent, wxSizer* sizer);
	void build_audio_card(wxWindow* parent, wxSizer* sizer);
	void build_action_row(wxWindow* parent, wxSizer* sizer);

	void video_dropped(std::vector<boost::filesystem::path> paths);
	void subtitles_dropped(std::vector<boost::filesystem::path> paths);
	void choose_video();
	/** Swap the film's picture (and any bare sound) for a different file --
	 *  which is what a button labelled "Replace" has to do.  Nothing is removed
	 *  until the user has actually chosen a replacement. */
	void replace_video();
	void choose_subtitles();
	void content_type_changed();
	/** Pick the language of the soundtrack, or clear it again.  Optional: the
	 *  DCP name carries XX for "not specified", which is a legitimate answer
	 *  and the one a film gets until someone says otherwise. */
	void choose_audio_language();
	void clear_audio_language();
	/** Pick the language of one subtitle file (its own, not the film's). */
	void choose_subtitle_language(std::weak_ptr<Content> content);
	void change_output_folder();
	void remove_video();
	void remove_subtitle(std::weak_ptr<Content> content);
	void create_dcp();
	/** @return true if a DCP is being written from this film right now.
	 *
	 *  The one place this question is asked from.  It walks JobManager::get(),
	 *  so it may ONLY be called from a context holding no JobManager or Job lock
	 *  -- i.e. from a user action, never from a job signal.  set_general_
	 *  sensitivity() deliberately does not use it: the host already tells this
	 *  panel when an export owns the film, and re-deriving it under
	 *  ActiveJobsChanged is how the UI thread deadlocked against itself before. */
	bool export_in_flight() const;
	/** The single owner of the output "Change..." button's enabled state; two
	 *  copies of this predicate had already drifted apart. */
	void update_output_change_enabled();
	/** @return where the project should live if the user picked @p chosen as
	 *  the output folder: @p chosen itself when it is new or empty, otherwise a
	 *  subfolder of it named after the film (see the definition for why a
	 *  non-empty folder must never be used directly).  boost::none if no free
	 *  name could be found -- the caller must NOT fall back to a colliding path,
	 *  because the move's failure rollback deletes whatever it is given. */
	boost::optional<boost::filesystem::path> project_folder_for(boost::filesystem::path const& chosen) const;
	/** Bring a pre-2026-07-25 mono mapping onto the upmixer's mono leg and
	 *  re-measure, so what this screen draws matches what the export makes. */
	void migrate_mono_mapping();

	/** Make sure there is a film to put content into, inventing a project
	 *  folder next to DCP-o-matic's default one if the user has not chosen a
	 *  destination yet.  @return false if no film could be made. */
	bool ensure_film(boost::filesystem::path const& first_content);
	void add_paths(std::vector<boost::filesystem::path> paths, bool as_subtitles);

	void film_changed(ChangeType type, FilmProperty property);
	void film_content_changed(ChangeType type, int property);
	void job_added(std::weak_ptr<Job> job);
	void jobs_changed();
	void poll_analysis();
	void analysis_finished(Job::Result result, std::weak_ptr<SlangAudioAnalyseJob> job);

	/** Apply the choices the simplified interface makes on the user's behalf:
	 *  use added subtitles as open subtitles, and give a mono/stereo source the
	 *  smart-centre L/C/R mix before the audio is measured. */
	void apply_simple_defaults();

	void update_video_card();
	void update_subtitle_card();
	void update_output_card();
	void update_content_type();
	void update_audio_language();
	void update_action_row();
	void update_all();

	void content_layout_changed();
	void save();
	void resized();

	std::shared_ptr<Film> _film;
	bool _active = false;
	bool _sensitive = true;
	/** set_use() re-enters apply_simple_defaults() through the synchronous
	 *  ContentChange signal; the outer call is the one whose result is kept. */
	bool _applying_defaults = false;

	wxScrolledWindow* _scroller = nullptr;
	wxSizerItem* _left_margin = nullptr;
	wxSizerItem* _right_margin = nullptr;

	SlangCard* _video_card = nullptr;
	SlangDropArea* _video_drop = nullptr;
	/** Held so set_general_sensitivity() can reach them.  They were locals, and
	 *  therefore stayed live through an export: Remove during a 40-minute
	 *  transcode empties the playlist the running Player is reading from. */
	SlangFlatButton* _video_replace = nullptr;
	SlangFlatButton* _video_remove = nullptr;
	wxPanel* _video_details = nullptr;
	wxStaticText* _video_name = nullptr;
	wxStaticText* _video_summary = nullptr;
	wxStaticText* _video_encoding = nullptr;
	/** What this DCP is -- feature, short, clip, trailer... -- which becomes the
	 *  FTR/SHR/CLP part of its name and the CPL's ContentKind. */
	Choice* _content_type = nullptr;

	SlangCard* _subtitle_card = nullptr;
	SlangDropArea* _subtitle_drop = nullptr;
	wxPanel* _subtitle_list = nullptr;
	wxSizer* _subtitle_list_sizer = nullptr;
	/** Every per-file button of the CURRENT list -- "set language" AND "remove"
	 *  -- so a sensitivity change can reach them without rebuilding it.
	 *  Borrowed pointers: the sizer owns the buttons, and this is emptied before
	 *  it destroys them. */
	std::vector<SlangFlatButton*> _subtitle_row_buttons;

	SlangCard* _output_card = nullptr;
	wxStaticText* _output_path = nullptr;
	wxStaticText* _output_dcp = nullptr;
	SlangFlatButton* _output_change = nullptr;

	SlangCard* _audio_card = nullptr;
	SlangAudioPipelineView* _pipeline = nullptr;
	SlangFlatButton* _audio_language = nullptr;
	SlangFlatButton* _audio_language_clear = nullptr;
	/** Shown only while no language is set: says what the essence will claim
	 *  anyway, since libdcp has no "unspecified" to write there. */
	wxStaticText* _audio_language_note = nullptr;

	SlangFlatButton* _new = nullptr;

	SlangFlatButton* _create = nullptr;
	wxStaticText* _create_note = nullptr;

	JobManagerView* _jobs = nullptr;

	std::weak_ptr<SlangAudioAnalyseJob> _analysis_job;
	wxTimer _analysis_timer;
	/** subtitle content added here that still needs its "use" default applied
	 *  once the examine job has filled in its text tracks */
	std::vector<std::weak_ptr<Content>> _pending_subtitles;

	boost::signals2::scoped_connection _film_changed_connection;
	boost::signals2::scoped_connection _film_content_changed_connection;
	boost::signals2::scoped_connection _job_added_connection;
	boost::signals2::scoped_connection _jobs_changed_connection;
	boost::signals2::scoped_connection _analysis_finished_connection;
};


#endif

#endif
