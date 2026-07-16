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


/** Preferences page for the Slang/Vulkan GPU J2K encoder (frame-server
 *  integration): enable the GPU export, pick the Tier-1 coder (HT default
 *  vs MQ), point at the frame server's socket, and control the audio
 *  automation (GPU auto-gain to just under -3 dBFS; smart-centre upmix for
 *  mono/stereo sources).  Mirrors the Grok GPUPage next door. */

#pragma once

#include <wx/choice.h>


class SlangGPUPage : public dcpomatic::preferences::Page
{
public:
	SlangGPUPage(wxSize panel_size, int border)
		: Page(panel_size, border)
	{}

	wxString GetName() const override
	{
		return _("GPU (Slang)");
	}

private:
	void setup() override
	{
		_enable = new CheckBox(_panel, _("Export using the GPU (Slang encoder)"));
		_panel->GetSizer()->Add(_enable, 0, wxALL | wxEXPAND, _border);

		wxFlexGridSizer* table = new wxFlexGridSizer(2, DCPOMATIC_SIZER_X_GAP, DCPOMATIC_SIZER_Y_GAP);
		table->AddGrowableCol(1, 1);
		_panel->GetSizer()->Add(table, 1, wxALL | wxEXPAND, _border);

		add_label_to_sizer(table, _panel, _("J2K coder"), true, 0, wxLEFT | wxRIGHT | wxALIGN_CENTRE_VERTICAL);
		_coder = new wxChoice(_panel, wxID_ANY);
		_coder->Append(_("HT — HTJ2K, fastest (default)"));
		_coder->Append(_("MQ — classic JPEG2000, highest quality"));
		table->Add(_coder, 1, wxEXPAND);

		add_label_to_sizer(table, _panel, _("Frame server socket"), true, 0, wxLEFT | wxRIGHT | wxALIGN_CENTRE_VERTICAL);
		_socket = new wxTextCtrl(_panel, wxID_ANY);
		table->Add(_socket, 1, wxEXPAND | wxALL);

		_auto_gain = new CheckBox(_panel, _("Analyse audio on the GPU and reduce gain to just under -3dB"));
		_panel->GetSizer()->Add(_auto_gain, 0, wxALL | wxEXPAND, _border);

		_smart_center = new CheckBox(_panel, _("Mix mono/stereo sources to L/C/R with a smart centre"));
		_panel->GetSizer()->Add(_smart_center, 0, wxALL | wxEXPAND, _border);

		_match_source_bitrate = new CheckBox(_panel, _("Set the DCP's video bit rate from the source video's (scaled for its codec's efficiency)"));
		_panel->GetSizer()->Add(_match_source_bitrate, 0, wxALL | wxEXPAND, _border);

		_enable->bind(&SlangGPUPage::changed, this);
		_coder->Bind(wxEVT_CHOICE, boost::bind(&SlangGPUPage::changed, this));
		_socket->Bind(wxEVT_TEXT, boost::bind(&SlangGPUPage::changed, this));
		_auto_gain->bind(&SlangGPUPage::changed, this);
		_smart_center->bind(&SlangGPUPage::changed, this);
		_match_source_bitrate->bind(&SlangGPUPage::changed, this);

		setup_sensitivity();
	}

	void setup_sensitivity()
	{
		auto const slang = Config::instance()->slang();
		_coder->Enable(slang.enable);
		_socket->Enable(slang.enable);
		_auto_gain->Enable(slang.enable);
		_smart_center->Enable(slang.enable);
		_match_source_bitrate->Enable(slang.enable);
	}

	void config_changed() override
	{
		auto const slang = Config::instance()->slang();

		checked_set(_enable, slang.enable);
		_coder->SetSelection(slang.coder == "mq" ? 1 : 0);
		checked_set(_socket, slang.socket);
		checked_set(_auto_gain, slang.auto_gain);
		checked_set(_smart_center, slang.smart_center);
		checked_set(_match_source_bitrate, slang.match_source_bitrate);

		setup_sensitivity();
	}

	void changed()
	{
		auto slang = Config::instance()->slang();
		slang.enable = _enable->GetValue();
		slang.coder = _coder->GetSelection() == 1 ? "mq" : "ht";
		slang.socket = wx_to_std(_socket->GetValue());
		slang.auto_gain = _auto_gain->GetValue();
		slang.smart_center = _smart_center->GetValue();
		slang.match_source_bitrate = _match_source_bitrate->GetValue();
		Config::instance()->set_slang(slang);

		setup_sensitivity();
	}

	CheckBox* _enable = nullptr;
	wxChoice* _coder = nullptr;
	wxTextCtrl* _socket = nullptr;
	CheckBox* _auto_gain = nullptr;
	CheckBox* _smart_center = nullptr;
	CheckBox* _match_source_bitrate = nullptr;
};
