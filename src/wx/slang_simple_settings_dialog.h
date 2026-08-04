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


/** @file  src/wx/slang_simple_settings_dialog.h
 *  @brief SlangSimpleSettingsDialog: the simplified interface's Settings screen.
 */


#ifndef DCPOMATIC_SLANG_SIMPLE_SETTINGS_DIALOG_H
#define DCPOMATIC_SLANG_SIMPLE_SETTINGS_DIALOG_H


#ifdef DCPOMATIC_SLANG


#include <dcp/language_tag.h>
#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
#include <wx/wx.h>
LIBDCP_ENABLE_WARNINGS
#include <boost/optional.hpp>
#include <memory>
#include <string>


class CheckBox;
class Film;
class LanguageTagWidget;
class RegionSubtagWidget;
class SlangCard;
class wxSpinCtrl;


/** @class SlangSimpleSettingsDialog
 *  @brief Everything a DCP needs that is the same for every DCP you make, asked
 *  once.
 *
 *  The simplified interface runs with no menu bar, so Preferences is not
 *  reachable from it at all -- and the things Preferences holds are not
 *  optional decoration: an ISDCF Studio and Facility code, the issuer and
 *  creator strings that go into every CPL and PKL, the release territory, and
 *  the whole GPU encoder configuration.  Without this screen the simplified
 *  interface could only ever make packages carrying the "no registered code"
 *  sentinels and DCP-o-matic's own issuer, and the user's own answers had to be
 *  typed in the full interface -- which is exactly the interface they chose not
 *  to use.
 *
 *  It is deliberately NOT a second Preferences: it holds the settings that
 *  change what the DCP IS, and nothing else.  Everything here is also in the
 *  full interface's Preferences (this writes the same Config), so the two can
 *  never disagree about where a value lives.
 *
 *  Save/Cancel rather than Preferences' apply-as-you-type, for two reasons.
 *  The ISDCF codes are grammar-checked (a code that does not fit its field is
 *  silently DROPPED from the DCP's name, giving an 11-part name a QC tool
 *  rejects), and half of a code is not a code -- typing "AB" on the way to
 *  "ABCD" must not be persisted and stamped into any film created in between.
 */
class SlangSimpleSettingsDialog : public wxDialog
{
public:
	/** @param film the project that is open, if any.  A setting the user
	 *  CHANGES here is applied to it as well as stored as the default for new
	 *  projects: the alternative is a screen that appears to do nothing until
	 *  the next project, which is not what "Settings" means to the person who
	 *  just typed their studio code in.
	 */
	SlangSimpleSettingsDialog(wxWindow* parent, std::shared_ptr<Film> film);

	/** @return true if anything the open film shows was changed, so the caller
	 *  knows whether to save and redraw it. */
	bool film_changed() const {
		return _film_changed;
	}

private:
	SlangCard* add_card(wxSizer* sizer, wxString title, wxString subtitle);
	void build_identity(wxSizer* sizer);
	void build_defaults(wxSizer* sizer);
	void build_picture(wxSizer* sizer);
	void build_gpu(wxSizer* sizer);

	void setup_sensitivity();
	void save();

	/** Validate one ISDCF code field against the DCNC character rule for it.
	 *  @param value what the user typed (already trimmed).
	 *  @param min_length minimum number of characters the field allows.
	 *  @param max_length maximum number of characters the field allows.
	 *  @return the upper-cased code, or boost::none if the field is empty
	 *  (which is a legitimate answer: it means "no registered code").
	 *  Throws std::runtime_error, with a message naming the rule, if what was
	 *  typed cannot be a code.
	 */
	static boost::optional<std::string> validate_isdcf_code(
		std::string value, char const* field, size_t min_length, size_t max_length
		);

	std::shared_ptr<Film> _film;
	bool _film_changed = false;

	wxTextCtrl* _studio = nullptr;
	wxTextCtrl* _facility = nullptr;
	wxTextCtrl* _issuer = nullptr;
	wxTextCtrl* _creator = nullptr;

	CheckBox* _enable_audio_language = nullptr;
	LanguageTagWidget* _audio_language = nullptr;
	CheckBox* _enable_territory = nullptr;
	RegionSubtagWidget* _territory = nullptr;

	CheckBox* _match_source_bitrate = nullptr;
	wxSpinCtrl* _maximum_bit_rate = nullptr;
	wxStaticText* _bit_rate_note = nullptr;

	CheckBox* _auto_gain = nullptr;
	CheckBox* _smart_center = nullptr;
	wxTextCtrl* _socket = nullptr;
};


#endif

#endif
