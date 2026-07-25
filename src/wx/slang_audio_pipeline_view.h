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


/** @file  src/wx/slang_audio_pipeline_view.h
 *  @brief SlangAudioPipelineView: what the soundtrack is about to become.
 */


#ifndef DCPOMATIC_SLANG_AUDIO_PIPELINE_VIEW_H
#define DCPOMATIC_SLANG_AUDIO_PIPELINE_VIEW_H


#ifdef DCPOMATIC_SLANG


#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
#include <wx/wx.h>
LIBDCP_ENABLE_WARNINGS
#include <boost/optional.hpp>
#include <memory>
#include <string>
#include <vector>


class Film;
class wxGraphicsContext;


/** @class SlangAudioPipelineView
 *  @brief Draws the film's whole audio path: source channels on the left, what
 *  happens to them in the middle, and the DCP soundtrack that comes out on the
 *  right, with the levels the GPU analysis actually measured.
 *
 *  Everything shown is read back from the Film (and from the analysis results
 *  SlangAudioAnalyseJob persists on it), never from anything this view computes
 *  itself: the point of the display is to show what the export will really do,
 *  so a discrepancy between the picture and the DCP has to be a bug somewhere
 *  it can be seen, not a second implementation of the same maths drifting from
 *  the first.
 */
class SlangAudioPipelineView : public wxPanel
{
public:
	explicit SlangAudioPipelineView(wxWindow* parent);

	void set_film(std::shared_ptr<Film> film);

	/** Re-read the film's content, mapping, processor and stored analysis. */
	void refresh_state();

	/** Show that an analysis is in flight.  @param progress 0..1, or none for
	 *  "started but no progress yet". */
	void set_analysing(bool analysing, boost::optional<float> progress = {});

	/** A short note about where the last measurement came from, e.g.
	 *  "Measured on the GPU"; empty hides the note. */
	void set_measurement_note(wxString note);

private:
	struct InputChannel
	{
		wxString name;
		/** indices (in _mix_inputs) this channel is mapped into */
		std::vector<int> destinations;
		double y = 0;
	};

	struct Source
	{
		wxString title;
		wxString detail;
		std::vector<InputChannel> channels;
	};

	struct MixInput
	{
		int index = 0;
		wxString name;
		double y = 0;
	};

	struct OutputChannel
	{
		wxString name;
		/** natural (pre-auto-gain) peak and RMS, linear; < 0 if not measured */
		double peak = -1;
		double rms = -1;
		/** this channel was covered by the stored analysis.  Per-channel, NOT
		 *  the film-wide _have_measurement: the stored vector is as wide as the
		 *  film was when it was measured, so widening the film afterwards (which
		 *  the smart-centre upmix does, to 6) leaves the new channels
		 *  unmeasured while the film still counts as measured overall.  Without
		 *  this they render as "silent" -- a positive claim about a centre
		 *  channel that may be carrying all the dialogue. */
		bool measured = false;
		/** something is mapped or processed into this channel */
		bool live = false;
		double y = 0;
	};

	void paint();
	void paint_empty(wxGraphicsContext* gc, wxString message);
	void draw_column_heading(wxGraphicsContext* gc, wxString text, double x, double y, double width);
	void draw_meter(wxGraphicsContext* gc, wxRect const& rect, double dbfs, bool live);
	void draw_link(wxGraphicsContext* gc, double x0, double y0, double x1, double y1, bool strong);
	/** Which DCP channels the current audio processor feeds from its own input
	 *  @ref mix_input_index; identity for anything but the smart-centre upmix. */
	std::vector<int> processor_destinations(int mix_input_index) const;

	wxSize DoGetBestSize() const override;

	std::shared_ptr<Film> _film;

	std::vector<Source> _sources;
	std::vector<MixInput> _mix_inputs;
	std::vector<OutputChannel> _outputs;

	/** name of the audio processor in use, empty if none */
	wxString _processor_name;
	std::string _processor_id;
	/** short description lines for the processing box */
	std::vector<wxString> _processor_lines;

	/** the auto-gain currently baked into the mix, dB */
	double _gain_db = 0;
	/** natural peak of the whole mix, dBFS; none if not measured */
	boost::optional<double> _natural_peak_dbfs;
	bool _have_measurement = false;
	bool _any_audio = false;

	bool _analysing = false;
	boost::optional<float> _progress;
	wxString _measurement_note;
};


#endif

#endif
