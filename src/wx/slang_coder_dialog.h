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


/** @file  src/wx/slang_coder_dialog.h
 *  @brief SlangCoderDialog: the "which JPEG 2000 coder?" chooser shown at the
 *  start of a GPU export (Jobs -> Make DCP using GPU), while the GPU audio
 *  analysis runs in the background.  HT (fast, the default) vs MQ (highest
 *  PSNR) with a concise line and a fuller strengths/weaknesses paragraph each.
 */

#pragma once

#include "wx_util.h"
#include <wx/wx.h>
#include <string>


class SlangCoderDialog : public wxDialog
{
public:
	/** @param current_coder "ht" or "mq" - the option to pre-select.
	 *  @param audio_running true when the GPU audio-analysis job is churning
	 *  in the background behind this dialog (adds the explanatory footnote).
	 *  @param bit_rate_mbps if > 0, the DCP video bit rate that was set
	 *  automatically to match the source video (adds a note); 0 disables it.
	 *  @param bit_rate_changed true if that automatic value differs from the
	 *  film's previous bit rate (so the note says "adjusted" rather than
	 *  "already matches").
	 */
	SlangCoderDialog(
		wxWindow* parent, std::string current_coder, bool audio_running,
		int bit_rate_mbps, bool bit_rate_changed
		)
		: wxDialog(parent, wxID_ANY, _("Make DCP using GPU"))
	{
		/* The detail paragraphs are indented under their radio button; wrap
		 * them a little narrower than the full dialog body. */
		int const body_wrap = 540;
		int const indent = 24;
		int const detail_wrap = body_wrap - indent;

		auto overall = new wxBoxSizer(wxVERTICAL);
		SetSizer(overall);

		auto sizer = new wxBoxSizer(wxVERTICAL);

		auto heading = new wxStaticText(this, wxID_ANY, _("Choose the JPEG 2000 encoder for this DCP"));
		auto heading_font = heading->GetFont();
		heading_font.SetWeight(wxFONTWEIGHT_BOLD);
		heading_font.SetPointSize(heading_font.GetPointSize() + 1);
		heading->SetFont(heading_font);
		sizer->Add(heading, 0, wxBOTTOM, DCPOMATIC_SIZER_GAP);

		auto intro = new wxStaticText(
			this, wxID_ANY,
			_("Both produce a fully DCI-compliant DCP. You can change this any time in "
			  "Preferences -> GPU (Slang).")
			);
		intro->Wrap(body_wrap);
		sizer->Add(intro, 0, wxBOTTOM, DCPOMATIC_SIZER_GAP * 2);

		/* Report the automatic source-matched bit rate (match_source_bitrate,
		 * on by default) so the user sees the DCP's bandwidth was set for them
		 * rather than silently changed. */
		if (bit_rate_mbps > 0) {
			auto bit_rate = new wxStaticText(
				this, wxID_ANY,
				bit_rate_changed
				? wxString::Format(
					_("The DCP's video bit rate was automatically adjusted to %d Mbit/s to match the source video."),
					bit_rate_mbps)
				: wxString::Format(
					_("The DCP's video bit rate already matches the source video (%d Mbit/s)."),
					bit_rate_mbps)
				);
			auto bit_rate_font = bit_rate->GetFont();
			bit_rate_font.SetWeight(wxFONTWEIGHT_BOLD);
			bit_rate->SetFont(bit_rate_font);
			bit_rate->Wrap(body_wrap);
			sizer->Add(bit_rate, 0, wxBOTTOM, DCPOMATIC_SIZER_GAP * 2);
		}

		/* Create both radio buttons consecutively (wxRB_GROUP on the first) so
		 * they form a single exclusive group; the descriptive labels added by
		 * add_option() afterwards do not affect the grouping. */
		_ht = new wxRadioButton(
			this, wxID_ANY, _("HT - HTJ2K (High-Throughput)"),
			wxDefaultPosition, wxDefaultSize, wxRB_GROUP
			);
		_mq = new wxRadioButton(this, wxID_ANY, _("MQ - classic JPEG 2000"));

		add_option(
			sizer, _ht, indent, detail_wrap,
			_("Recommended - the fastest export, and tuned to match or beat MQ on how the picture looks."),
			_("The modern High-Throughput block coder (JPEG 2000 Part 15). On the GPU it "
			  "encodes 4K frames roughly 3x faster than MQ, so exports finish much sooner, "
			  "and it still fills the whole DCI bit-rate budget. Its raw PSNR is about 0.4 dB "
			  "below MQ, but with the perceptual tuning applied here it equals or beats MQ on "
			  "the measures that track what the eye actually sees - fine detail, smooth "
			  "gradients (banding) and Butteraugli. Decoded by OpenJPEG 2.5+ and modern "
			  "digital-cinema players; only a few older Part-1-only tools cannot read it.")
			);

		add_option(
			sizer, _mq, indent, detail_wrap,
			_("Highest PSNR and the widest compatibility - but slower to encode."),
			_("The original JPEG 2000 (Part 1) arithmetic coder that every DCP player has read "
			  "for twenty years. It gives the highest PSNR, which means the decoded frames are "
			  "numerically the closest to your source: fine gradients and detail come back with "
			  "the least deviation and the lowest chance of a visible compression artefact on "
			  "demanding shots (at DCI bit rates the margin over HT is small and mostly "
			  "sub-perceptual). The trade-off is speed - about 3x slower than HT on the "
			  "GPU. Choose it for maximum fidelity, or for guaranteed playback on the oldest "
			  "equipment.")
			);

		if (current_coder == "mq") {
			_mq->SetValue(true);
		} else {
			_ht->SetValue(true);
		}

		if (audio_running) {
			auto note = new wxStaticText(
				this, wxID_ANY,
				_("Meanwhile the audio is being analysed on the GPU in the background. The DCP "
				  "will start once you have chosen and that analysis has finished.")
				);
			auto note_font = note->GetFont();
			note_font.SetStyle(wxFONTSTYLE_ITALIC);
			note->SetFont(note_font);
			note->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_GRAYTEXT));
			note->Wrap(body_wrap);
			sizer->Add(note, 0, wxTOP, DCPOMATIC_SIZER_GAP);
		}

		overall->Add(sizer, 1, wxEXPAND | wxALL, DCPOMATIC_DIALOG_BORDER);

		if (auto buttons = CreateSeparatedButtonSizer(wxOK | wxCANCEL)) {
			if (auto ok = FindWindow(wxID_OK)) {
				ok->SetLabel(_("Make DCP"));
			}
			overall->Add(buttons, wxSizerFlags().Expand().DoubleBorder());
		}

		overall->Layout();
		Fit();
		CentreOnParent();
	}

	/** @return the chosen coder, "ht" or "mq". */
	std::string coder() const
	{
		return _mq->GetValue() ? "mq" : "ht";
	}

private:
	void add_option(wxSizer* sizer, wxRadioButton* radio, int indent, int wrap, wxString concise, wxString detail)
	{
		sizer->Add(radio, 0, wxTOP, DCPOMATIC_SIZER_GAP);

		auto concise_text = new wxStaticText(this, wxID_ANY, concise);
		auto concise_font = concise_text->GetFont();
		concise_font.SetWeight(wxFONTWEIGHT_BOLD);
		concise_text->SetFont(concise_font);
		concise_text->Wrap(wrap);
		sizer->Add(concise_text, 0, wxLEFT, indent);

		auto detail_text = new wxStaticText(this, wxID_ANY, detail);
		detail_text->Wrap(wrap);
		detail_text->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_GRAYTEXT));
		sizer->Add(detail_text, 0, wxLEFT | wxBOTTOM, indent);
	}

	wxRadioButton* _ht = nullptr;
	wxRadioButton* _mq = nullptr;
};
