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

#include "slang_audio_pipeline_view.h"
#include "slang_ui_theme.h"
#include "wx_util.h"
#include "lib/audio_content.h"
#include "lib/audio_mapping.h"
#include "lib/audio_processor.h"
#include "lib/audio_stream.h"
#include "lib/content.h"
#include "lib/film.h"
#include "lib/slang_audio_analyse_job.h"
#include "lib/smart_center_upmixer.h"
#include "lib/util.h"
#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
#include <wx/dcbuffer.h>
#include <wx/graphics.h>
LIBDCP_ENABLE_WARNINGS
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>


using std::shared_ptr;
using std::vector;
using boost::optional;


/** Bottom of the meter scale, dBFS. */
static double const METER_FLOOR = -60;


static double
to_dbfs(double linear)
{
	return linear > 0 ? 20 * std::log10(linear) : -std::numeric_limits<double>::infinity();
}


/** What to call a source channel.  DCP-o-matic names them "stream:channel"
 *  ("1:1", "1:2"), which is exact but means nothing to someone who just dropped
 *  a video in.  For a single mono or stereo stream the convention is
 *  unambiguous -- and is precisely what the smart-centre processor assumes when
 *  it reads legs 0 and 1 -- so name those; anything else (multiple streams,
 *  unusual channel counts, a layout we would have to guess at) keeps the exact
 *  name rather than risking a confident mislabel.
 */
static wxString
source_channel_name(int channel, int channels, std::string const& exact)
{
	if (channels == 1) {
		return _("Mono");
	}
	if (channels == 2) {
		return channel == 0 ? _("Left") : _("Right");
	}
	return std_to_wx(exact);
}


static wxString
format_dbfs(double dbfs)
{
	if (!std::isfinite(dbfs)) {
		/* An all-zero channel: "silent" reads better than "-inf dB", and this
		 * is the normal state of LFE/Ls/Rs for a stereo source. */
		return _("silent");
	}
	return wxString::Format(char_to_wx("%+.1f dB"), dbfs);
}


SlangAudioPipelineView::SlangAudioPipelineView(wxWindow* parent)
	: wxPanel(parent, wxID_ANY)
{
	SetBackgroundStyle(wxBG_STYLE_PAINT);
	SetBackgroundColour(slang_ui::palette().card);
	Bind(wxEVT_PAINT, [this](wxPaintEvent&) { paint(); });
	Bind(wxEVT_SIZE, [this](wxSizeEvent& ev) { Refresh(); ev.Skip(); });
}


void
SlangAudioPipelineView::set_film(shared_ptr<Film> film)
{
	_film = film;
	refresh_state();
}


void
SlangAudioPipelineView::set_analysing(bool analysing, optional<float> progress)
{
	if (_analysing == analysing && _progress == progress) {
		return;
	}
	_analysing = analysing;
	_progress = progress;
	Refresh();
}


void
SlangAudioPipelineView::set_measurement_note(wxString note)
{
	if (_measurement_note == note) {
		return;
	}
	_measurement_note = note;
	Refresh();
}


void
SlangAudioPipelineView::refresh_state()
{
	auto const previous_rows = _sources.size() + _outputs.size();

	_sources.clear();
	_mix_inputs.clear();
	_outputs.clear();
	_processor_name.clear();
	_processor_id.clear();
	_processor_lines.clear();
	_gain_db = 0;
	_natural_peak_dbfs = {};
	_have_measurement = false;
	_any_audio = false;

	if (!_film) {
		Refresh();
		return;
	}

	_gain_db = _film->slang_auto_gain_db();

	/* What the content's channels are mapped into: the audio processor's own
	 * inputs when there is one, otherwise the DCP channels directly. This is
	 * exactly what the audio panel's mapping grid shows as its columns. */
	for (auto const& named: _film->audio_output_channel_names()) {
		MixInput input;
		input.index = named.index;
		input.name = std_to_wx(named.name);
		_mix_inputs.push_back(input);
	}

	if (auto processor = _film->audio_processor()) {
		_processor_name = std_to_wx(processor->name());
		_processor_id = processor->id();
		if (_processor_id == "smart-center-upmixer") {
			/* A mono source is mapped into BOTH of the processor's inputs, so
			 * mid = (L + R) / 2 = M and the two differences come out at exactly
			 * zero: C carries the whole mix and L/R are silent. The stereo
			 * matrix is still what runs, but printing it for a mono source
			 * reads as a contradiction next to two silent meters -- so say what
			 * that matrix DOES here, which is the question an operator looking
			 * at this screen is actually asking. */
			auto mono = false;
			for (auto content: _film->content()) {
				if (!content->audio) {
					continue;
				}
				mono = content->audio->channel_names().size() == 1;
				if (!mono) {
					break;
				}
			}

			if (mono) {
				_processor_lines.push_back(_("Mono source: C = M"));
				_processor_lines.push_back(_("L = R = M / 2     (6 dB below the centre)"));
				_processor_lines.push_back(_("Dialogue stays anchored to the centre speaker"));
			} else {
				/* Spelling the matrix out is the point: an operator needs to see
				 * that this is centre EXTRACTION (L'+C == L) and not a downmix
				 * that doubles dialogue into a phantom centre as well. */
				_processor_lines.push_back(_("C = (L + R) / 2"));
				_processor_lines.push_back(_("L' = L - C     R' = R - C"));
				_processor_lines.push_back(_("Dialogue plays from the centre speaker alone"));
			}
		}
	}

	/* Source side: one block per piece of content that has audio. */
	for (auto content: _film->content()) {
		if (!content->audio) {
			continue;
		}
		_any_audio = true;

		Source source;
		source.title = std_to_wx(content->path(0).filename().string());

		auto const mapping = content->audio->mapping();
		auto const names = content->audio->channel_names();

		int frame_rate = 0;
		for (auto stream: content->audio->streams()) {
			frame_rate = std::max(frame_rate, stream->frame_rate());
		}

		auto const channels = static_cast<int>(names.size());
		wxString layout;
		switch (channels) {
		case 1:
			layout = _("Mono");
			break;
		case 2:
			layout = _("Stereo");
			break;
		case 6:
			layout = _("5.1");
			break;
		case 8:
			layout = _("7.1");
			break;
		default:
			layout = wxString::Format(_("%d channels"), channels);
			break;
		}
		source.detail = frame_rate > 0
			? wxString::Format(char_to_wx("%s \u00b7 %.1f kHz"), layout, frame_rate / 1000.0)
			: layout;

		auto const single_stream = content->audio->streams().size() == 1;
		for (int c = 0; c < channels && c < mapping.input_channels(); ++c) {
			InputChannel channel;
			channel.name = single_stream
				? source_channel_name(c, channels, names[c].name)
				: std_to_wx(names[c].name);
			for (size_t m = 0; m < _mix_inputs.size(); ++m) {
				if (_mix_inputs[m].index < mapping.output_channels()
				    && mapping.get(c, _mix_inputs[m].index) > 0) {
					channel.destinations.push_back(static_cast<int>(m));
				}
			}
			source.channels.push_back(channel);
		}

		_sources.push_back(source);
	}

	/* DCP side: every channel the film will write, with the measured level. */
	auto const peak = _film->slang_audio_channel_peak();
	auto const rms = _film->slang_audio_channel_rms();
	_have_measurement = !peak.empty();

	double natural_peak = 0;
	for (int c = 0; c < _film->audio_channels(); ++c) {
		OutputChannel output;
		output.name = std_to_wx(short_audio_channel_name(c));
		if (c < static_cast<int>(peak.size())) {
			output.peak = peak[c];
			output.rms = c < static_cast<int>(rms.size()) ? rms[c] : -1;
			natural_peak = std::max(natural_peak, static_cast<double>(peak[c]));
		}
		_outputs.push_back(output);
	}

	if (_have_measurement && natural_peak > 0) {
		_natural_peak_dbfs = to_dbfs(natural_peak);
	}

	/* Which DCP channels actually receive something.  With a processor this
	 * follows the processor's own routing; without one it is just the mapping.
	 * A measured non-zero level always counts as live, whatever the routing
	 * says, so the picture can never claim a channel is empty when the
	 * soundtrack has audio in it. */
	for (size_t m = 0; m < _mix_inputs.size(); ++m) {
		bool fed = false;
		for (auto const& source: _sources) {
			for (auto const& channel: source.channels) {
				if (std::find(channel.destinations.begin(), channel.destinations.end(), static_cast<int>(m)) != channel.destinations.end()) {
					fed = true;
				}
			}
		}
		if (!fed) {
			continue;
		}
		for (auto destination: processor_destinations(static_cast<int>(m))) {
			if (destination < static_cast<int>(_outputs.size())) {
				_outputs[destination].live = true;
			}
		}
	}
	for (auto& output: _outputs) {
		if (output.peak > 0) {
			output.live = true;
		}
	}

	if (previous_rows != _sources.size() + _outputs.size()) {
		InvalidateBestSize();
		if (GetContainingSizer()) {
			GetContainingSizer()->Layout();
		}
	}

	Refresh();
}


vector<int>
SlangAudioPipelineView::processor_destinations(int mix_input_index) const
{
	if (mix_input_index < 0 || mix_input_index >= static_cast<int>(_mix_inputs.size())) {
		return {};
	}

	auto const index = _mix_inputs[mix_input_index].index;

	if (_processor_id != "smart-center-upmixer") {
		/* No processor (or one we do not model): the mapping column IS the DCP
		 * channel. */
		return { index };
	}

	/* SmartCenterUpmixer::do_run(): its two legs become L', R' and their mid
	 * becomes C, so each leg reaches both its own channel and the centre.  The
	 * mono leg is spread over all three front channels.  The pass-through legs
	 * (HI/VI/DBP/DBS/Sign) keep their own channel. */
	if (index == SmartCenterUpmixer::MONO_INPUT) {
		return { 0, 1, 2 };
	}
	if (index == 0 || index == 1) {
		return { index, 2 };
	}
	return { index };
}


wxSize
SlangAudioPipelineView::DoGetBestSize() const
{
	auto const row = GetCharHeight() + FromDIP(9);
	int source_rows = 0;
	for (auto const& source: _sources) {
		source_rows += 2 + static_cast<int>(source.channels.size());
	}
	auto const rows = std::max({ source_rows, static_cast<int>(_outputs.size()), 6 });
	/* Always reserve the footnote's line, even before there is a footnote to
	 * put in it: the note appears and disappears as measurements come and go,
	 * and a best size that changed with it would need the whole card column
	 * re-laid-out each time -- which is exactly the sort of thing that gets
	 * missed, leaving the note clipped off the bottom. */
	auto const note = GetCharHeight() + FromDIP(12);
	return wxSize(FromDIP(560), FromDIP(58) + rows * row + note + FromDIP(12));
}


void
SlangAudioPipelineView::draw_column_heading(wxGraphicsContext* gc, wxString text, double x, double y, double width)
{
	auto const p = slang_ui::palette();
	gc->SetFont(gc->CreateFont(slang_ui::font(this, -2, true), p.muted));
	slang_ui::draw_text(gc, text.Upper(), x, y, width);
}


void
SlangAudioPipelineView::draw_link(wxGraphicsContext* gc, double x0, double y0, double x1, double y1, bool strong)
{
	auto const p = slang_ui::palette();
	gc->SetPen(wxPen(strong ? p.accent : slang_ui::mix(p.accent, p.card, 0.35), strong ? 1.6 : 1.2));
	gc->SetBrush(*wxTRANSPARENT_BRUSH);

	auto path = gc->CreatePath();
	path.MoveToPoint(x0, y0);
	auto const bend = (x1 - x0) * 0.5;
	path.AddCurveToPoint(x0 + bend, y0, x1 - bend, y1, x1, y1);
	gc->StrokePath(path);
}


void
SlangAudioPipelineView::draw_meter(wxGraphicsContext* gc, wxRect const& rect, double dbfs, bool live)
{
	auto const p = slang_ui::palette();
	auto const radius = rect.height / 2.0;

	slang_ui::rounded_rect(gc, rect, radius, p.card_sunken, p.border);

	if (!live || !std::isfinite(dbfs)) {
		return;
	}

	auto const fraction = std::min(1.0, std::max(0.0, (dbfs - METER_FLOOR) / -METER_FLOOR));
	auto const width = static_cast<int>(std::lround(fraction * rect.width));
	if (width > 2) {
		/* Anything within half a dB of full scale is about to clip; the DCI
		 * target this pipeline normalises to is -3.5 dBFS, so amber starts
		 * just above it. */
		auto const colour = dbfs >= -0.5 ? p.danger : (dbfs >= -2.0 ? p.warning : p.accent);
		slang_ui::rounded_rect(gc, wxRect(rect.x, rect.y, width, rect.height), radius, colour);
	}

	/* The -3.5 dBFS target the auto-gain aims for. */
	auto const target = (SlangAudioAnalyseJob::TARGET_PEAK_DBFS - METER_FLOOR) / -METER_FLOOR;
	auto const target_x = rect.x + target * rect.width;
	gc->SetPen(wxPen(slang_ui::mix(p.text, p.card_sunken, 0.35), 1));
	auto tick = gc->CreatePath();
	tick.MoveToPoint(target_x, rect.y + 1);
	tick.AddLineToPoint(target_x, rect.y + rect.height - 1);
	gc->StrokePath(tick);
}


void
SlangAudioPipelineView::paint_empty(wxGraphicsContext* gc, wxString message)
{
	auto const p = slang_ui::palette();
	wxRect const rect(GetSize());
	gc->SetFont(gc->CreateFont(slang_ui::font(this, 0), p.muted));
	wxDouble width, height, descent, leading;
	gc->GetTextExtent(message, &width, &height, &descent, &leading);
	gc->DrawText(message, (rect.width - width) / 2, (rect.height - height) / 2);
}


void
SlangAudioPipelineView::paint()
{
	wxAutoBufferedPaintDC dc(this);
	dc.SetBackground(wxBrush(GetBackgroundColour()));
	dc.Clear();

	std::unique_ptr<wxGraphicsContext> gc(wxGraphicsContext::Create(dc));
	if (!gc) {
		return;
	}
	gc->SetAntialiasMode(wxANTIALIAS_DEFAULT);

	auto const p = slang_ui::palette();
	wxRect const client(GetSize());

	if (!_film || !_any_audio) {
		paint_empty(gc.get(), _("Add a video or sound file to see its audio pipeline."));
		return;
	}

	auto const pad = FromDIP(4);
	auto const gap = FromDIP(22);
	auto const row = GetCharHeight() + FromDIP(9);
	double const dot = FromDIP(3);

	auto const usable = client.width - pad * 2;
	auto const source_width = static_cast<double>(usable) * 0.30;
	auto const mix_width = static_cast<double>(usable) * 0.32;
	auto const output_width = usable - source_width - mix_width - gap * 2;

	auto const source_x = static_cast<double>(pad);
	auto const mix_x = source_x + source_width + gap;
	auto const output_x = mix_x + mix_width + gap;

	auto const heading_y = static_cast<double>(FromDIP(4));
	auto const body_y = heading_y + GetCharHeight() + FromDIP(12);

	draw_column_heading(gc.get(), _("Source"), source_x, heading_y, source_width);
	draw_column_heading(gc.get(), _processor_name.IsEmpty() ? _("Routing") : _("Processing"), mix_x, heading_y, mix_width);
	draw_column_heading(gc.get(), _("DCP soundtrack"), output_x, heading_y, output_width);

	/* ---- source column ------------------------------------------------- */

	auto y = body_y;
	for (auto& source: _sources) {
		gc->SetFont(gc->CreateFont(slang_ui::font(this, -1, true), p.text));
		slang_ui::draw_text(gc.get(), source.title, source_x, y, source_width);
		y += GetCharHeight() + FromDIP(2);
		gc->SetFont(gc->CreateFont(slang_ui::font(this, -2), p.muted));
		slang_ui::draw_text(gc.get(), source.detail, source_x, y, source_width);
		y += GetCharHeight() + FromDIP(6);

		for (auto& channel: source.channels) {
			channel.y = y + row / 2.0;
			gc->SetFont(gc->CreateFont(slang_ui::font(this, -1), p.text));
			slang_ui::draw_text(
				gc.get(), channel.name, source_x + FromDIP(8), y + FromDIP(2), source_width - FromDIP(24)
				);
			gc->SetBrush(wxBrush(channel.destinations.empty() ? p.border : p.accent));
			gc->SetPen(*wxTRANSPARENT_PEN);
			gc->DrawEllipse(source_x + source_width - dot * 2, channel.y - dot, dot * 2, dot * 2);
			y += row;
		}
		y += FromDIP(6);
	}
	auto const source_bottom = y;

	/* ---- DCP output column --------------------------------------------- */

	auto const meter_width = std::max(FromDIP(48), static_cast<int>(output_width * 0.42));
	auto const label_width = FromDIP(30);

	y = body_y;
	for (auto& output: _outputs) {
		output.y = y + row / 2.0;

		auto const natural = to_dbfs(output.peak);
		auto const resulting = output.peak > 0 ? natural + _gain_db : natural;

		gc->SetFont(gc->CreateFont(slang_ui::font(this, -1, output.live), output.live ? p.text : p.muted));
		slang_ui::draw_text(gc.get(), output.name, output_x + FromDIP(10), y + FromDIP(2), label_width);

		wxRect const meter(
			static_cast<int>(output_x + label_width + FromDIP(14)),
			static_cast<int>(output.y - FromDIP(5)),
			meter_width,
			FromDIP(10)
			);
		draw_meter(gc.get(), meter, _have_measurement ? resulting : -std::numeric_limits<double>::infinity(), output.live);

		gc->SetFont(gc->CreateFont(slang_ui::font(this, -2), output.live ? p.muted : slang_ui::mix(p.muted, p.card, 0.5)));
		auto const level = !_have_measurement
			? (output.live ? wxString(_("not measured")) : wxString(_("silent")))
			: format_dbfs(output.peak > 0 ? resulting : -std::numeric_limits<double>::infinity());
		slang_ui::draw_text(
			gc.get(), level, meter.GetRight() + FromDIP(8), y + FromDIP(2),
			output_x + output_width - meter.GetRight() - FromDIP(8)
			);

		gc->SetBrush(wxBrush(output.live ? p.accent : p.border));
		gc->SetPen(*wxTRANSPARENT_PEN);
		gc->DrawEllipse(output_x, output.y - dot, dot * 2, dot * 2);

		y += row;
	}
	auto const output_bottom = y;

	/* ---- processing column --------------------------------------------- */

	auto const stack_bottom = std::max(source_bottom, output_bottom);
	auto const mix_in_x = mix_x;
	auto const mix_out_x = mix_x + mix_width;

	/* The mix box: what the source channels are combined into. */
	auto mix_box_height = GetCharHeight() + FromDIP(16);
	if (!_processor_lines.empty()) {
		mix_box_height += static_cast<int>(_processor_lines.size()) * (GetCharHeight() + FromDIP(2));
	}
	wxRect const mix_box(
		static_cast<int>(mix_x), static_cast<int>(body_y),
		static_cast<int>(mix_width), mix_box_height
		);
	slang_ui::rounded_rect(gc.get(), mix_box, FromDIP(8), p.accent_soft, slang_ui::mix(p.accent, p.card, 0.4));

	{
		auto text_y = static_cast<double>(mix_box.y + FromDIP(8));
		gc->SetFont(gc->CreateFont(slang_ui::font(this, -1, true), p.text));
		slang_ui::draw_text(
			gc.get(),
			_processor_name.IsEmpty() ? _("Direct (no mixing)") : _processor_name,
			mix_box.x + FromDIP(10), text_y, mix_box.width - FromDIP(20)
			);
		text_y += GetCharHeight() + FromDIP(4);
		gc->SetFont(gc->CreateFont(slang_ui::font(this, -2), p.muted));
		for (auto const& line: _processor_lines) {
			slang_ui::draw_text(gc.get(), line, mix_box.x + FromDIP(10), text_y, mix_box.width - FromDIP(20));
			text_y += GetCharHeight() + FromDIP(2);
		}
	}

	/* Pin the mapping columns down the left edge of the mix box. */
	for (size_t m = 0; m < _mix_inputs.size(); ++m) {
		_mix_inputs[m].y = mix_box.y + mix_box.height * (m + 1.0) / (_mix_inputs.size() + 1.0);
	}

	/* The gain box, below it. */
	auto const gain_top = mix_box.GetBottom() + FromDIP(14);
	wxRect const gain_box(
		static_cast<int>(mix_x), gain_top,
		static_cast<int>(mix_width),
		GetCharHeight() * 3 + FromDIP(22)
		);
	slang_ui::rounded_rect(gc.get(), gain_box, FromDIP(8), p.card_sunken, p.border);

	{
		auto text_y = static_cast<double>(gain_box.y + FromDIP(8));
		gc->SetFont(gc->CreateFont(slang_ui::font(this, -1, true), p.text));
		slang_ui::draw_text(gc.get(), _("Automatic gain"), gain_box.x + FromDIP(10), text_y, gain_box.width - FromDIP(20));
		text_y += GetCharHeight() + FromDIP(4);

		gc->SetFont(gc->CreateFont(slang_ui::font(this, -2), p.muted));
		if (_analysing) {
			slang_ui::draw_text(
				gc.get(),
				_progress
					? wxString::Format(_("Measuring the mix on the GPU... %d%%"), static_cast<int>(*_progress * 100))
					: wxString(_("Measuring the mix on the GPU...")),
				gain_box.x + FromDIP(10), text_y, gain_box.width - FromDIP(20)
				);
			text_y += GetCharHeight() + FromDIP(4);

			wxRect const bar(
				gain_box.x + FromDIP(10), static_cast<int>(text_y) + FromDIP(2),
				gain_box.width - FromDIP(20), FromDIP(6)
				);
			slang_ui::rounded_rect(gc.get(), bar, bar.height / 2.0, p.border);
			if (_progress) {
				auto const width = static_cast<int>(std::lround(bar.width * std::min(1.0f, std::max(0.0f, *_progress))));
				if (width > 2) {
					slang_ui::rounded_rect(gc.get(), wxRect(bar.x, bar.y, width, bar.height), bar.height / 2.0, p.accent);
				}
			}
		} else if (!_have_measurement) {
			slang_ui::draw_text(
				gc.get(), _("Not measured yet"), gain_box.x + FromDIP(10), text_y, gain_box.width - FromDIP(20)
				);
		} else {
			auto const natural = _natural_peak_dbfs.get_value_or(-std::numeric_limits<double>::infinity());
			auto const resulting = natural + _gain_db;
			gc->SetFont(gc->CreateFont(slang_ui::font(this, -2), p.muted));
			slang_ui::draw_text(
				gc.get(),
				std::isfinite(natural)
					? wxString::Format(_("Mix peak %+.1f dB to %+.1f dB"), natural, resulting)
					: wxString(_("The mix is silent")),
				gain_box.x + FromDIP(10), text_y, gain_box.width - FromDIP(20)
				);
			text_y += GetCharHeight() + FromDIP(4);

			auto const capped = _gain_db > 0
				&& (natural + _gain_db) < SlangAudioAnalyseJob::TARGET_PEAK_DBFS - 0.05;
			wxString summary;
			if (!std::isfinite(natural)) {
				summary = _("No gain applied");
			} else if (std::abs(_gain_db) < 0.05) {
				summary = _("Already at the -3.5 dB target");
			} else if (capped) {
				/* Only name a cap when there IS one: the shipped policy boosts
				 * as far as the target needs, and "capped at inf dB" would be
				 * nonsense on a mix that landed short for any other reason. */
				summary = std::isfinite(SlangAudioAnalyseJob::MAX_BOOST_DB)
					? wxString::Format(
						_("%+.1f dB applied (boost capped at %.0f dB)"),
						_gain_db, SlangAudioAnalyseJob::MAX_BOOST_DB
						)
					: wxString::Format(_("%+.1f dB applied"), _gain_db);
			} else {
				summary = wxString::Format(_("%+.1f dB applied to reach the -3.5 dB target"), _gain_db);
			}
			gc->SetFont(gc->CreateFont(slang_ui::font(this, -2, true), _gain_db < 0 ? p.warning : p.success));
			slang_ui::draw_text(gc.get(), summary, gain_box.x + FromDIP(10), text_y, gain_box.width - FromDIP(20));
		}
	}

	/* ---- links ---------------------------------------------------------- */

	for (auto const& source: _sources) {
		for (auto const& channel: source.channels) {
			for (auto destination: channel.destinations) {
				draw_link(
					gc.get(),
					source_x + source_width, channel.y,
					mix_in_x, _mix_inputs[destination].y,
					true
					);
			}
		}
	}

	for (size_t m = 0; m < _mix_inputs.size(); ++m) {
		bool fed = false;
		for (auto const& source: _sources) {
			for (auto const& channel: source.channels) {
				if (std::find(channel.destinations.begin(), channel.destinations.end(), static_cast<int>(m)) != channel.destinations.end()) {
					fed = true;
				}
			}
		}
		if (!fed) {
			continue;
		}
		for (auto destination: processor_destinations(static_cast<int>(m))) {
			if (destination < static_cast<int>(_outputs.size())) {
				draw_link(gc.get(), mix_out_x, _mix_inputs[m].y, output_x, _outputs[destination].y, true);
			}
		}
	}

	/* Mapping pins on the mix box edges, drawn last so the links tuck under. */
	gc->SetPen(*wxTRANSPARENT_PEN);
	for (auto const& input: _mix_inputs) {
		gc->SetBrush(wxBrush(p.accent));
		gc->DrawEllipse(mix_in_x - dot, input.y - dot, dot * 2, dot * 2);
		gc->DrawEllipse(mix_out_x - dot, input.y - dot, dot * 2, dot * 2);
	}

	/* ---- footnote ------------------------------------------------------- */

	if (!_measurement_note.IsEmpty()) {
		gc->SetFont(gc->CreateFont(slang_ui::font(this, -2), p.muted));
		slang_ui::draw_text(
			gc.get(), _measurement_note, source_x,
			std::max(static_cast<double>(gain_box.GetBottom() + FromDIP(10)), stack_bottom + FromDIP(2)),
			usable
			);
	}
}

#endif
