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
#include "language_tag_widget.h"
#include "region_subtag_widget.h"
#include "slang_simple_settings_dialog.h"
#include "slang_ui_theme.h"
#include "wx_util.h"
#include "lib/config.h"
#include "lib/film.h"
#include "lib/video_encoding.h"
#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
#include <wx/spinctrl.h>
LIBDCP_ENABLE_WARNINGS
#include <fmt/format.h>
#include <boost/algorithm/string.hpp>
#include <boost/bind/bind.hpp>
#include <algorithm>
#include <stdexcept>


#if BOOST_VERSION >= 106100
using namespace boost::placeholders;
#endif


using std::shared_ptr;
using std::string;
using boost::optional;


/** Width the cards are laid out at; the dialog sizes itself to this. */
static int const DIALOG_WIDTH = 660;
static int const MARGIN = 16;
static int const CARD_GAP = 12;

/** The DCI ceiling (SMPTE ST 429-2 / the DCI spec), and the point above which
 *  QC tools start commenting on a package that is still perfectly legal.
 *
 *  libdcp's own verifier reports any frame above 230 Mb/s as
 *  NEARLY_INVALID_PICTURE_FRAME_SIZE_IN_BYTES, and Clairmeta warns when the
 *  whole asset averages above 245 Mb/s.  Neither makes the DCP invalid -- 250
 *  is the limit and a DCP at 250 meets it -- but a delivery that draws two
 *  advisories is a delivery someone has to explain, so the number is worth
 *  saying out loud next to the control rather than leaving to be discovered
 *  from a QC report.
 */
static int const DCI_MAXIMUM_BIT_RATE_MBPS = 250;
static int const QC_QUIET_BIT_RATE_MBPS = 230;

/** What Film::Film() actually gives a new project: min(this, the ceiling).
 *  Kept in step with src/lib/film.cc by hand -- there is no shared constant to
 *  reach for -- and only ever used to SAY what will happen, never to set it. */
static int const DCPOMATIC_DEFAULT_BIT_RATE_MBPS = 150;


SlangSimpleSettingsDialog::SlangSimpleSettingsDialog(wxWindow* parent, shared_ptr<Film> film)
	: wxDialog(parent, wxID_ANY, _("Settings"))
	, _film(film)
{
	auto const p = slang_ui::palette();
	SetBackgroundColour(p.page);

	auto outer = new wxBoxSizer(wxVERTICAL);

	auto intro = new wxStaticText(
		this, wxID_ANY,
		_("These are the answers that are the same for every DCP you make. "
		  "Set them once; new projects start with them.")
		);
	intro->SetFont(slang_ui::font(this, -1));
	intro->SetForegroundColour(p.muted);
	intro->Wrap(FromDIP(DIALOG_WIDTH));
	outer->Add(intro, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, FromDIP(MARGIN));

	auto column = new wxBoxSizer(wxVERTICAL);
	outer->Add(column, 1, wxEXPAND | wxALL, FromDIP(MARGIN));

	build_identity(column);
	column->AddSpacer(FromDIP(CARD_GAP));
	build_defaults(column);
	column->AddSpacer(FromDIP(CARD_GAP));
	build_picture(column);
	column->AddSpacer(FromDIP(CARD_GAP));
	build_gpu(column);

	/* wxStdDialogButtonSizer rather than two hand-placed buttons: it puts OK and
	 * Cancel in the platform's own order, and binds Escape to Cancel. */
	auto buttons = CreateStdDialogButtonSizer(wxOK | wxCANCEL);
	if (auto ok = FindWindow(wxID_OK)) {
		static_cast<wxButton*>(ok)->SetLabel(_("Save"));
	}
	outer->Add(buttons, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, FromDIP(MARGIN));

	/* Validation lives in save(), and a failed validation must NOT close the
	 * dialog -- so the OK button is intercepted here rather than letting the
	 * default handler end the modal loop.  EndModal is called only once every
	 * field has been accepted. */
	Bind(wxEVT_BUTTON, [this](wxCommandEvent& ev) {
		if (ev.GetId() != wxID_OK) {
			ev.Skip();
			return;
		}
		try {
			save();
		} catch (std::exception& e) {
			error_dialog(this, std_to_wx(e.what()));
			return;
		}
		EndModal(wxID_OK);
	});

	SetSizerAndFit(outer);
	SetMinSize(GetSize());
	CentreOnParent();

	setup_sensitivity();
}


SlangCard*
SlangSimpleSettingsDialog::add_card(wxSizer* sizer, wxString title, wxString subtitle)
{
	auto card = new SlangCard(this, title, subtitle);
	card->SetMinSize(wxSize(FromDIP(DIALOG_WIDTH), -1));
	sizer->Add(card, 0, wxEXPAND);
	return card;
}


/** A two-column table inside a card, sized so the value column takes the slack. */
static wxFlexGridSizer*
card_table(SlangCard* card)
{
	auto table = new wxFlexGridSizer(2, DCPOMATIC_SIZER_X_GAP, DCPOMATIC_SIZER_Y_GAP);
	table->AddGrowableCol(1, 1);
	card->body()->Add(table, 0, wxEXPAND);
	return table;
}


static wxStaticText*
card_label(SlangCard* card, wxFlexGridSizer* table, wxString text)
{
	auto label = new wxStaticText(card, wxID_ANY, text);
	label->SetFont(slang_ui::font(card, -1));
	label->SetForegroundColour(slang_ui::palette().muted);
	table->Add(label, 0, wxALIGN_CENTRE_VERTICAL);
	return label;
}


/** A muted note under a card's controls, wrapped to the dialog's width. */
static wxStaticText*
card_note(SlangCard* card, wxString text)
{
	auto note = new wxStaticText(card, wxID_ANY, text);
	note->SetFont(slang_ui::font(card, -2));
	note->SetForegroundColour(slang_ui::palette().muted);
	note->Wrap(card->FromDIP(DIALOG_WIDTH) - card->FromDIP(MARGIN) * 3);
	card->body()->Add(note, 0, wxEXPAND | wxTOP, card->FromDIP(6));
	return note;
}


/** What the sentence under the bit-rate box says, for a given state of the
 *  "match the source" box and a given ceiling.  One function so the note shown
 *  and the note the dialog is SIZED for cannot drift apart -- see where it is
 *  called from in build_picture().
 */
static wxString
bit_rate_note_text(bool match_source, int mbps)
{
	wxString note;
	if (match_source) {
		note = wxString::Format(
			_("The source's own bit rate is used, scaled for its codec, and never allowed above "
			  "%d Mb/s."),
			mbps
			);
	} else {
		/* This box is a CEILING (Config::maximum_video_bit_rate), not the rate.
		 * It used to say "Every DCP is made at %d Mb/s." here, which was simply
		 * untrue: with matching off nothing writes the film's bit rate at all,
		 * so a new project gets Film::Film()'s own min(150 Mb/s, this ceiling)
		 * and an existing one keeps what it had.  At the shipped 250 the screen
		 * therefore claimed 250 while the Video card next door -- reading the
		 * same film -- correctly showed 150.  Two screens of one interface
		 * disagreeing about the mastering rate is exactly the failure this
		 * project keeps finding, so the sentence now says what actually
		 * happens. */
		note = wxString::Format(
			_("Projects are made at %d Mb/s -- DCP-o-matic's own default -- or at this rate if it "
			  "is lower, since this is a limit rather than the rate itself."),
			std::min(mbps, DCPOMATIC_DEFAULT_BIT_RATE_MBPS)
			);
	}

	if (mbps > DCI_MAXIMUM_BIT_RATE_MBPS) {
		/* Above the DCI limit entirely.  Reachable only because Preferences ->
		 * Non-standard offers 250-1000 for work that is not a DCI DCP; this
		 * dialog neither offers nor silently removes it (see build_picture()). */
		note += char_to_wx(" ");
		note += wxString::Format(
			_("%d Mb/s is above the DCI limit of %d Mb/s: a package made at this rate is not a "
			  "DCI-compliant DCP. It was set in Preferences -> Non-standard."),
			mbps, DCI_MAXIMUM_BIT_RATE_MBPS
			);
	} else if (mbps > QC_QUIET_BIT_RATE_MBPS) {
		/* Not an error: 250 is the DCI limit and a DCP at 250 meets it.  But
		 * both verification tools this project checks against comment above
		 * 230, so say it here instead of letting a QC report be where it comes
		 * up. */
		note += char_to_wx(" ");
		note += wxString::Format(
			_("%d Mb/s is the DCI limit; above %d Mb/s verification tools report the picture as "
			  "close to it, which is legal but often queried."),
			DCI_MAXIMUM_BIT_RATE_MBPS, QC_QUIET_BIT_RATE_MBPS
			);
	}

	return note;
}


void
SlangSimpleSettingsDialog::build_identity(wxSizer* sizer)
{
	auto card = add_card(
		sizer, _("You"), _("Who made this DCP - it is written into every package.")
		);
	auto table = card_table(card);

	card_label(card, table, _("Studio code"));
	_studio = new wxTextCtrl(card, wxID_ANY);
	_studio->SetToolTip(
		_("Your ISDCF-registered studio code, e.g. 'ABC'. Two to four letters or digits.")
		);
	table->Add(_studio, 1, wxEXPAND);

	card_label(card, table, _("Facility code"));
	_facility = new wxTextCtrl(card, wxID_ANY);
	_facility->SetToolTip(
		_("Your ISDCF-registered facility code, e.g. 'ABC'. Two or three letters or digits.")
		);
	table->Add(_facility, 1, wxEXPAND);

	card_label(card, table, _("Issuer"));
	_issuer = new wxTextCtrl(card, wxID_ANY);
	_issuer->SetToolTip(_("Written into the DCP's CPL and PKL as who issued the package."));
	table->Add(_issuer, 1, wxEXPAND);

	card_label(card, table, _("Creator"));
	_creator = new wxTextCtrl(card, wxID_ANY);
	_creator->SetToolTip(_("Written into the DCP's CPL and PKL as what created the package."));
	table->Add(_creator, 1, wxEXPAND);

	/* Both codes are real ISDCF-administered registries -- you apply for one and
	 * wait -- so "leave it empty" has to be an obviously acceptable answer here,
	 * and it must be clear what an empty field ships as.  The convention's own
	 * sentinels are NULL (studio) and NUL (facility); inventing a plausible
	 * three-letter code instead is not informality, it is a claim to a code
	 * somebody else may hold. */
	card_note(
		card,
		_("Studio and facility codes are issued by the ISDCF; leave them empty if you have not "
		  "registered one and the DCP's name will say NULL and NUL, which is what the naming "
		  "convention asks for.")
		);

	auto config = Config::instance();
	checked_set(_studio, config->default_studio().get_value_or(""));
	checked_set(_facility, config->default_facility().get_value_or(""));
	checked_set(_issuer, config->dcp_issuer());
	checked_set(_creator, config->dcp_creator());
}


void
SlangSimpleSettingsDialog::build_defaults(wxSizer* sizer)
{
	auto config = Config::instance();

	auto card = add_card(
		sizer, _("New projects"), _("What a project starts with when you add a video.")
		);
	auto table = card_table(card);

	_enable_audio_language = new CheckBox(card, _("Spoken language"));
	_enable_audio_language->SetForegroundColour(slang_ui::palette().muted);
	_enable_audio_language->SetFont(slang_ui::font(card, -1));
	table->Add(_enable_audio_language, 0, wxALIGN_CENTRE_VERTICAL);
	_audio_language = new LanguageTagWidget(
		card, _("The language the soundtrack is spoken in"), config->default_audio_language()
		);
	table->Add(_audio_language->sizer(), 1, wxEXPAND);

	_enable_territory = new CheckBox(card, _("Release territory"));
	_enable_territory->SetForegroundColour(slang_ui::palette().muted);
	_enable_territory->SetFont(slang_ui::font(card, -1));
	table->Add(_enable_territory, 0, wxALIGN_CENTRE_VERTICAL);
	_territory = new RegionSubtagWidget(
		card, _("The territory the DCP is released in"), config->default_territory()
		);
	table->Add(_territory->sizer(), 1, wxEXPAND);

	/* Left unticked, the name carries the convention's INT-TD / INT-TL
	 * ("international, texted / textless"), which Film::isdcf_name() works out
	 * live from whether the film has subtitles.  That is a real answer, not a
	 * gap -- so the note says so rather than nagging for a territory. */
	card_note(
		card,
		_("Without a territory the DCP's name says INT-TD or INT-TL - international, with or "
		  "without subtitles - which the naming convention allows.")
		);

	_enable_audio_language->bind(&SlangSimpleSettingsDialog::setup_sensitivity, this);
	_enable_territory->bind(&SlangSimpleSettingsDialog::setup_sensitivity, this);

	checked_set(_enable_audio_language, static_cast<bool>(config->default_audio_language()));
	checked_set(_enable_territory, static_cast<bool>(config->default_territory()));
}


void
SlangSimpleSettingsDialog::build_picture(wxSizer* sizer)
{
	auto config = Config::instance();
	auto const slang = config->slang();

	auto card = add_card(
		sizer, _("Picture"), _("How much data the picture is given.")
		);

	_match_source_bitrate = new CheckBox(card, _("Set the bit rate from the source video"));
	_match_source_bitrate->SetToolTip(
		_("Probe what the source video was encoded at, scale it for the codec it used, and "
		  "give the DCP a matching bit rate. Turn this off to always use the rate below.")
		);
	card->body()->Add(_match_source_bitrate, 0, wxEXPAND);

	auto table = card_table(card);
	card_label(card, table, _("Highest bit rate"));
	auto rate_row = new wxBoxSizer(wxHORIZONTAL);
	_maximum_bit_rate = new wxSpinCtrl(card);
	/* 10 Mb/s is create_cli's own floor for a hand-set rate; below it a 4K DCP
	 * is not worth making.  The DCI limit is the top THIS screen offers -- but
	 * NOT necessarily the top the config holds: Preferences -> Non-standard
	 * offers 250-1000 Mb/s for work that is not a DCI DCP.
	 *
	 * The range therefore stretches to whatever is already configured.  With a
	 * flat SetRange(10, 250), wxSpinCtrl::SetValue CLAMPS -- so a config holding
	 * 500 displayed as 250 and was written back as 250 by ANY Save, including
	 * one that only touched the socket.  A settings screen must not destroy a
	 * value it does not offer to set; it can decline to offer it, and say so
	 * (see bit_rate_note_text()), but it may not quietly lower it. */
	_maximum_bit_rate->SetRange(
		10,
		std::max(
			DCI_MAXIMUM_BIT_RATE_MBPS,
			static_cast<int>(config->maximum_video_bit_rate(VideoEncoding::JPEG2000) / 1000000)
			)
		);
	rate_row->Add(_maximum_bit_rate, 0, wxALIGN_CENTRE_VERTICAL);
	auto unit = new wxStaticText(card, wxID_ANY, _("Mb/s"));
	unit->SetFont(slang_ui::font(card, -1));
	unit->SetForegroundColour(slang_ui::palette().muted);
	rate_row->Add(unit, 0, wxALIGN_CENTRE_VERTICAL | wxLEFT, card->FromDIP(6));
	table->Add(rate_row, 1, wxEXPAND);

	/* Built with the LONGEST wording the note can ever take, so the dialog's
	 * SetSizerAndFit() reserves room for it; setup_sensitivity() then swaps in
	 * whichever version currently applies.  Built empty (or short), the dialog
	 * would be fitted to a two-line note and clip the moment the user raised the
	 * rate past the QC advisory threshold and the third line appeared -- and a
	 * warning nobody can read is worse than no warning. */
	_bit_rate_note = card_note(card, bit_rate_note_text(true, DCI_MAXIMUM_BIT_RATE_MBPS));

	_match_source_bitrate->bind(&SlangSimpleSettingsDialog::setup_sensitivity, this);
	_maximum_bit_rate->Bind(
		wxEVT_SPINCTRL, boost::bind(&SlangSimpleSettingsDialog::setup_sensitivity, this)
		);

	checked_set(_match_source_bitrate, slang.match_source_bitrate);
	checked_set(
		_maximum_bit_rate,
		static_cast<int>(config->maximum_video_bit_rate(VideoEncoding::JPEG2000) / 1000000)
		);
}


void
SlangSimpleSettingsDialog::build_gpu(wxSizer* sizer)
{
	auto const slang = Config::instance()->slang();

	auto card = add_card(
		sizer, _("Sound and the GPU encoder"), _("What happens to your film on the way to the DCP.")
		);

	_auto_gain = new CheckBox(card, _("Measure the sound on the GPU and level it for the cinema"));
	_auto_gain->SetToolTip(_("Bring the loudest channel to just under -3.5 dBFS."));
	card->body()->Add(_auto_gain, 0, wxEXPAND);

	_smart_center = new CheckBox(card, _("Give mono and stereo films a centre channel"));
	_smart_center->SetToolTip(
		_("Extract a centre from what the left and right channels have in common, and take it "
		  "back out of them, so the dialogue comes from the centre speaker.")
		);
	card->body()->Add(_smart_center, 0, wxEXPAND | wxTOP, card->FromDIP(4));

	/* There is deliberately NO "encode on the GPU" switch here.
	 *
	 * A draft of this dialog had one, and it was a control that could not
	 * express "off": this screen's Create DCP is wired straight to
	 * jobs_make_dcp_gpu_with_options() (dcpomatic.cc), whose export path
	 * unconditionally does `slang.enable = true; config->set_slang(slang)`
	 * before it starts -- so unticking the box survived only until the next
	 * export, which silently re-ticked it.  The full interface's Preferences
	 * page keeps the switch and it is meaningful there, because that interface
	 * has a plain "Make DCP" that honours it; the simplified one has a single
	 * button and that button IS the GPU export.
	 *
	 * A switch that reverts itself is worse than no switch, so the note says
	 * plainly what this interface does and where the real setting lives. */
	auto table = card_table(card);
	card_label(card, table, _("Frame server socket"));
	_socket = new wxTextCtrl(card, wxID_ANY);
	_socket->SetToolTip(_("The Unix socket the GPU frame server is listening on."));
	table->Add(_socket, 1, wxEXPAND);

	card_note(
		card,
		_("This interface always encodes the picture on the GPU; that is what it is for. To make "
		  "a DCP on the CPU instead, use Advanced... and Preferences.")
		);

	checked_set(_auto_gain, slang.auto_gain);
	checked_set(_smart_center, slang.smart_center);
	checked_set(_socket, slang.socket);
}


void
SlangSimpleSettingsDialog::setup_sensitivity()
{
	/* The rate box stays live either way, because it is a CEILING either way --
	 * this comment used to say it "is the rate itself" with matching off, which
	 * is what the note under it wrongly claimed too; see bit_rate_note_text().
	 * Only the sentence changes, and it changes to describe the consequence,
	 * not to give the same number a second meaning. */
	_bit_rate_note->SetLabel(
		bit_rate_note_text(_match_source_bitrate->GetValue(), _maximum_bit_rate->GetValue())
		);
	_bit_rate_note->Wrap(FromDIP(DIALOG_WIDTH) - FromDIP(MARGIN) * 3);

	_audio_language->enable(_enable_audio_language->GetValue());
	_territory->enable(_enable_territory->GetValue());

	Layout();
}


optional<string>
SlangSimpleSettingsDialog::validate_isdcf_code(string value, char const* field, size_t min_length, size_t max_length)
{
	boost::algorithm::trim(value);
	if (value.empty()) {
		/* Not an error: it means "I have no registered code", and Film::Film()
		 * turns that into the convention's own NULL / NUL sentinel. */
		return {};
	}

	auto const legal = std::all_of(value.begin(), value.end(), [](char c) {
		return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
	});

	/* Refused rather than silently trimmed or padded.  A code that does not fit
	 * its field is not written into a wrong DCP name -- Film::isdcf_name() drops
	 * the part ENTIRELY, so the package gets an 11-part name whose remaining
	 * fields a QC tool then reads against the wrong positions.  A message naming
	 * the rule is much cheaper than that. */
	if (!legal || value.size() < min_length || value.size() > max_length) {
		throw std::runtime_error(
			fmt::format(
				wx_to_std(
					_("{} must be {} to {} letters or digits (you typed \"{}\"). "
					  "Leave it empty if you have no registered code.")
					),
				field, min_length, max_length, value
				)
			);
	}

	boost::algorithm::to_upper(value);
	return value;
}


void
SlangSimpleSettingsDialog::save()
{
	auto config = Config::instance();

	/* Validate EVERYTHING before writing ANYTHING.  A half-applied save leaves
	 * the config holding a new studio code and the old facility one, with no
	 * indication of which of the two the error message was about. */
	auto const studio = validate_isdcf_code(wx_to_std(_studio->GetValue()), "Studio code", 2, 4);
	auto const facility = validate_isdcf_code(wx_to_std(_facility->GetValue()), "Facility code", 2, 3);

	/* A ticked box with nothing chosen behind it.  Neither inventing a value
	 * (a territory is a claim about where the DCP is released -- guessing one is
	 * not a convenience) nor silently ignoring the tick (the box would simply
	 * come back unticked next time, with no explanation) is acceptable, so say
	 * what is missing and let the user answer it. */
	if (_enable_audio_language->GetValue() && !_audio_language->get()) {
		throw std::runtime_error(
			wx_to_std(_("Press Edit... beside \"Spoken language\" to choose one, or untick it."))
			);
	}
	if (_enable_territory->GetValue() && !_territory->get()) {
		throw std::runtime_error(
			wx_to_std(_("Press Edit... beside \"Release territory\" to choose one, or untick it."))
			);
	}

	auto const old_studio = config->default_studio();
	auto const old_facility = config->default_facility();

	if (studio) {
		config->set_default_studio(*studio);
	} else {
		config->unset_default_studio();
	}
	if (facility) {
		config->set_default_facility(*facility);
	} else {
		config->unset_default_facility();
	}

	config->set_dcp_issuer(wx_to_std(_issuer->GetValue()));
	config->set_dcp_creator(wx_to_std(_creator->GetValue()));

	/* Both are guaranteed to hold a value by the checks at the top of this
	 * function, so the ticked branch never has to invent one. */
	if (_enable_audio_language->GetValue()) {
		config->set_default_audio_language(*_audio_language->get());
	} else {
		config->unset_default_audio_language();
	}

	if (_enable_territory->GetValue()) {
		config->set_default_territory(*_territory->get());
	} else {
		config->unset_default_territory();
	}

	/* Only when it actually moved, for the same reason studio and facility are
	 * diffed below: Save is equally how the socket or a sound checkbox gets
	 * written, and an unconditional write here would push a clamped value back
	 * into the config for a field the user never touched. */
	auto const maximum_bit_rate = static_cast<int64_t>(_maximum_bit_rate->GetValue()) * 1000000;
	if (maximum_bit_rate != config->maximum_video_bit_rate(VideoEncoding::JPEG2000)) {
		config->set_maximum_video_bit_rate(VideoEncoding::JPEG2000, maximum_bit_rate);
	}

	auto slang = config->slang();
	slang.match_source_bitrate = _match_source_bitrate->GetValue();
	slang.auto_gain = _auto_gain->GetValue();
	slang.smart_center = _smart_center->GetValue();
	slang.socket = wx_to_std(_socket->GetValue());
	config->set_slang(slang);

	if (!_film) {
		return;
	}

	/* Apply to the project that is open -- but only what the user actually
	 * CHANGED here.  Studio and facility are stamped onto a Film at
	 * construction, so a film made before this dialog was ever opened keeps the
	 * sentinel for ever otherwise, and "I typed my studio code and the DCP still
	 * says NULL" is the whole failure this screen exists to prevent.
	 *
	 * Only the changed ones, because Save is also how the socket or the bit rate
	 * gets set: re-stamping every field on every Save would quietly overwrite a
	 * per-film studio code somebody set in the full interface, for a project
	 * this dialog was not asked to touch.
	 */
	if (studio != old_studio) {
		_film->set_studio(studio.get_value_or("NULL"));
		_film_changed = true;
	}
	if (facility != old_facility) {
		_film->set_facility(facility.get_value_or("NUL"));
		_film_changed = true;
	}
}

#endif
