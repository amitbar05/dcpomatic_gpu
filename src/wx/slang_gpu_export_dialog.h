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


/** @file  src/wx/slang_gpu_export_dialog.h
 *  @brief SlangGPUExportDialog: the confirmation shown at the start of a GPU
 *  export (Jobs -> Make DCP using GPU), listing what the export is about to
 *  change about the project before it changes it.
 *
 *  This file was slang_coder_dialog.h until 2026-07-31, when the HT (HTJ2K,
 *  JPEG 2000 Part 15) coder was removed from the integration -- Part 15 is not
 *  what a DCI DCP is specified to carry -- leaving MQ as the only coder and the
 *  picker with nothing to pick.  The dialog itself stays, because picking a
 *  coder was never all it did: it is the ONLY point at which the user confirms
 *  an action that then rewrites the film.  Jobs -> Make DCP using GPU installs
 *  the smart-centre audio processor, widens the film to 6 channels, resets every
 *  content's AudioMapping and rewrites the DCP's video bit rate -- and nothing
 *  undoes any of that if the export is abandoned afterwards.  Deleting the
 *  dialog with the radio buttons would have made all of it happen from a menu
 *  click with no prompt and no disclosure.
 *
 *  (The 2026-07-25 review round moved the dialog to BEFORE those mutations for
 *  the same reason -- see the "ASK BEFORE MUTATING" comment at the call site.)
 */

#pragma once

#include "wx_util.h"
#include <wx/wx.h>


class SlangGPUExportDialog : public wxDialog
{
public:
	/** @param audio_will_be_analysed true when confirming this dialog will run
	 *  the GPU audio-analysis pre-pass (adds the explanatory footnote).  Future
	 *  tense on purpose: the dialog is shown BEFORE anything is started or
	 *  changed, so that cancelling leaves the film exactly as it was found.
	 *  @param audio_will_be_upmixed true when confirming will give the film the
	 *  smart-centre processor and widen it to 5.1 (mono/stereo sources only).
	 *  @param bit_rate_mbps if > 0, the DCP video bit rate that WILL be set to
	 *  match the source video when this dialog is confirmed (adds a note);
	 *  0 disables it.
	 *  @param bit_rate_changed true if that value differs from the film's
	 *  current bit rate (so the note says it will be adjusted rather than that
	 *  it already matches).
	 */
	SlangGPUExportDialog(
		wxWindow* parent, bool audio_will_be_analysed, bool audio_will_be_upmixed,
		int bit_rate_mbps, bool bit_rate_changed
		)
		: wxDialog(parent, wxID_ANY, _("Make DCP using GPU"))
	{
		int const body_wrap = 540;

		auto overall = new wxBoxSizer(wxVERTICAL);
		SetSizer(overall);

		auto sizer = new wxBoxSizer(wxVERTICAL);

		auto heading = new wxStaticText(this, wxID_ANY, _("Make this DCP with the GPU encoder"));
		auto heading_font = heading->GetFont();
		heading_font.SetWeight(wxFONTWEIGHT_BOLD);
		heading_font.SetPointSize(heading_font.GetPointSize() + 1);
		heading->SetFont(heading_font);
		sizer->Add(heading, 0, wxBOTTOM, DCPOMATIC_SIZER_GAP);

		auto intro = new wxStaticText(
			this, wxID_ANY,
			_("The picture will be encoded on the GPU as classic JPEG 2000 (Part 1) - the "
			  "essence a DCP is specified to carry, and what every cinema server reads. "
			  "The GPU encoder can be turned on and off in Preferences -> GPU (Slang).")
			);
		intro->Wrap(body_wrap);
		sizer->Add(intro, 0, wxBOTTOM, DCPOMATIC_SIZER_GAP * 2);

		/* What confirming will CHANGE about the project.  Everything listed
		 * here is applied only after this dialog returns wxID_OK, and none of
		 * it is undone if the export is later cancelled -- which is exactly why
		 * it is disclosed before rather than reported after. */
		if (bit_rate_mbps > 0 || audio_will_be_upmixed) {
			auto changes_heading = new wxStaticText(this, wxID_ANY, _("This will change the project:"));
			auto changes_font = changes_heading->GetFont();
			changes_font.SetWeight(wxFONTWEIGHT_BOLD);
			changes_heading->SetFont(changes_font);
			sizer->Add(changes_heading, 0, wxBOTTOM, DCPOMATIC_SIZER_GAP);

			/* Report the automatic source-matched bit rate
			 * (match_source_bitrate, on by default) so the user sees the DCP's
			 * bandwidth was set for them rather than silently changed. */
			if (bit_rate_mbps > 0) {
				add_change(
					sizer, body_wrap,
					bit_rate_changed
					? wxString::Format(
						_("The DCP's video bit rate will be set to %d Mbit/s to match the source video."),
						bit_rate_mbps)
					: wxString::Format(
						_("The DCP's video bit rate already matches the source video (%d Mbit/s)."),
						bit_rate_mbps)
					);
			}

			if (audio_will_be_upmixed) {
				add_change(
					sizer, body_wrap,
					_("The sound will be mixed to L, C and R by the smart-centre processor "
					  "and the DCP widened to 5.1 channels. This resets how each source's "
					  "channels are routed; you can change it afterwards in the DCP's Audio tab.")
					);
			}

			sizer->AddSpacer(DCPOMATIC_SIZER_GAP);
		}

		if (audio_will_be_analysed) {
			auto note = new wxStaticText(
				this, wxID_ANY,
				_("The audio will be analysed on the GPU and its level set automatically; "
				  "the DCP starts when that has finished.")
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

private:
	void add_change(wxSizer* sizer, int wrap, wxString text)
	{
		auto item = new wxStaticText(this, wxID_ANY, char_to_wx("•  ") + text);
		item->Wrap(wrap);
		sizer->Add(item, 0, wxLEFT | wxBOTTOM, 12);
	}
};
