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


/** @file  src/wx/slang_ui_theme.h
 *  @brief Small look-and-feel toolkit shared by the simplified interface.
 *
 *  wxWidgets' native controls carry a lot of platform chrome that reads as
 *  "settings dialog"; the simplified screen wants flat cards, one obvious
 *  primary action and a drop target.  Rather than skinning the whole app this
 *  header provides just the handful of owner-drawn pieces that screen needs:
 *
 *    - slang_ui::palette()  — one light and one dark set of tokens, picked from
 *                             DCP-o-matic's own gui_is_dark(), so the simplified
 *                             screen follows the user's system theme.
 *    - SlangCard            — a rounded panel with a numbered step badge, title
 *                             and subtitle; its body() sizer holds the content.
 *    - SlangFlatButton      — a flat button (primary/secondary/ghost/danger)
 *                             that emits the ordinary wxEVT_BUTTON, so callers
 *                             bind it exactly like a wxButton.
 *    - SlangDropArea        — a dashed "drop a file here, or click to choose"
 *                             target, used for the video and subtitle steps.
 *
 *  Everything is header-only and implicitly inline (member functions defined
 *  in-class), so it can be included by more than one translation unit.
 *  Header-only also mirrors slang_gpu_config_panel.h / slang_gpu_export_dialog.h.
 */


#ifndef DCPOMATIC_SLANG_UI_THEME_H
#define DCPOMATIC_SLANG_UI_THEME_H


#ifdef DCPOMATIC_SLANG


#include "wx_util.h"
#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
#include <wx/dcbuffer.h>
#include <wx/dnd.h>
#include <wx/graphics.h>
#include <wx/wx.h>
LIBDCP_ENABLE_WARNINGS
#include <boost/filesystem.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>


namespace slang_ui {


/** Colour tokens for one appearance (light or dark). */
struct Palette
{
	wxColour page;          ///< the screen behind the cards
	wxColour card;          ///< card fill
	wxColour card_sunken;   ///< inset areas inside a card (drop zones, meters)
	wxColour border;        ///< hairline card/inset border
	wxColour text;          ///< primary text
	wxColour muted;         ///< secondary text
	wxColour accent;        ///< brand/primary action
	wxColour accent_hover;
	wxColour accent_soft;   ///< accent at low opacity, for badges/fills
	wxColour on_accent;     ///< text on top of accent
	wxColour success;
	wxColour warning;
	wxColour danger;
};


inline Palette
palette()
{
	if (gui_is_dark()) {
		Palette p;
		p.page = wxColour(24, 27, 34);
		p.card = wxColour(33, 38, 47);
		p.card_sunken = wxColour(27, 31, 39);
		p.border = wxColour(53, 61, 74);
		p.text = wxColour(231, 237, 245);
		p.muted = wxColour(148, 163, 184);
		p.accent = wxColour(89, 143, 247);
		p.accent_hover = wxColour(122, 165, 250);
		p.accent_soft = wxColour(43, 60, 92);
		p.on_accent = wxColour(12, 16, 24);
		p.success = wxColour(74, 202, 137);
		p.warning = wxColour(233, 176, 82);
		p.danger = wxColour(244, 116, 116);
		return p;
	}

	Palette p;
	p.page = wxColour(244, 246, 250);
	p.card = wxColour(255, 255, 255);
	p.card_sunken = wxColour(246, 248, 251);
	p.border = wxColour(223, 228, 236);
	p.text = wxColour(23, 32, 42);
	p.muted = wxColour(100, 116, 139);
	p.accent = wxColour(45, 106, 227);
	p.accent_hover = wxColour(30, 84, 195);
	p.accent_soft = wxColour(226, 235, 253);
	p.on_accent = wxColour(255, 255, 255);
	p.success = wxColour(21, 128, 61);
	p.warning = wxColour(180, 105, 10);
	p.danger = wxColour(200, 45, 45);
	return p;
}


/** Blend @ref a over @ref b by @ref t (0 = all b, 1 = all a). */
inline wxColour
mix(wxColour const& a, wxColour const& b, double t)
{
	auto const lerp = [t](unsigned char x, unsigned char y) {
		return static_cast<unsigned char>(std::lround(y + (x - y) * t));
	};
	return wxColour(lerp(a.Red(), b.Red()), lerp(a.Green(), b.Green()), lerp(a.Blue(), b.Blue()));
}


/** A font derived from the window's own, so we inherit the user's UI font and
 *  its size; @ref point_delta nudges the size and @ref bold picks the weight.
 */
inline wxFont
font(wxWindow const* window, int point_delta = 0, bool bold = false)
{
	auto f = window->GetFont();
	f.SetPointSize(std::max(6, f.GetPointSize() + point_delta));
	f.SetWeight(bold ? wxFONTWEIGHT_BOLD : wxFONTWEIGHT_NORMAL);
	return f;
}


inline void
rounded_rect(wxGraphicsContext* gc, wxRect const& rect, double radius, wxColour const& fill)
{
	gc->SetBrush(wxBrush(fill));
	gc->SetPen(*wxTRANSPARENT_PEN);
	gc->DrawRoundedRectangle(rect.x, rect.y, rect.width, rect.height, radius);
}


inline void
rounded_rect(wxGraphicsContext* gc, wxRect const& rect, double radius, wxColour const& fill, wxColour const& border, double border_width = 1)
{
	gc->SetBrush(wxBrush(fill));
	gc->SetPen(wxPen(border, border_width));
	/* Inset by half the pen width so the stroke lands inside the rectangle
	 * rather than straddling its edge (which looks blurred). */
	auto const h = border_width / 2;
	gc->DrawRoundedRectangle(rect.x + h, rect.y + h, rect.width - border_width, rect.height - border_width, radius);
}


/** Draw text truncated with an ellipsis so it never spills out of @ref max_width. */
inline void
draw_text(wxGraphicsContext* gc, wxString const& text, double x, double y, double max_width)
{
	wxDouble width, height, descent, leading;
	gc->GetTextExtent(text, &width, &height, &descent, &leading);
	if (width <= max_width || text.IsEmpty()) {
		gc->DrawText(text, x, y);
		return;
	}

	auto shortened = text;
	while (!shortened.IsEmpty()) {
		shortened.RemoveLast();
		gc->GetTextExtent(shortened + wxString::FromUTF8("…"), &width, &height, &descent, &leading);
		if (width <= max_width) {
			break;
		}
	}
	gc->DrawText(shortened + wxString::FromUTF8("…"), x, y);
}


}


/** @class SlangFlatButton
 *  @brief A flat, rounded, owner-drawn button that emits wxEVT_BUTTON.
 *
 *  Keyboard-operable (space/enter) and focusable, so it is not a mouse-only
 *  control.  Sized from its label; pass @ref wide to make it fill its slot.
 */
class SlangFlatButton : public wxWindow
{
public:
	enum class Kind
	{
		PRIMARY,   ///< the one obvious action on the screen
		SECONDARY, ///< outlined
		GHOST,     ///< text only, for tertiary actions
		DANGER     ///< outlined, destructive
	};

	SlangFlatButton(wxWindow* parent, wxString label, Kind kind = Kind::SECONDARY)
		: wxWindow(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxWANTS_CHARS)
		, _label(label)
		, _kind(kind)
	{
		SetBackgroundStyle(wxBG_STYLE_PAINT);
		Bind(wxEVT_PAINT, [this](wxPaintEvent&) { paint(); });
		Bind(wxEVT_ENTER_WINDOW, [this](wxMouseEvent&) { set_hover(true); });
		Bind(wxEVT_LEAVE_WINDOW, [this](wxMouseEvent&) { set_hover(false); });
		Bind(wxEVT_LEFT_DOWN, [this](wxMouseEvent&) { left_down(); });
		Bind(wxEVT_LEFT_UP, [this](wxMouseEvent& ev) { left_up(ev); });
		Bind(wxEVT_MOUSE_CAPTURE_LOST, [this](wxMouseCaptureLostEvent&) { capture_lost(); });
		Bind(wxEVT_KEY_DOWN, [this](wxKeyEvent& ev) { key_down(ev); });
		Bind(wxEVT_SET_FOCUS, [this](wxFocusEvent& ev) { refresh_focus(true); ev.Skip(); });
		Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& ev) { refresh_focus(false); ev.Skip(); });
		SetCursor(wxCursor(wxCURSOR_HAND));
	}

	bool AcceptsFocus() const override
	{
		return IsThisEnabled() && IsShown();
	}

	bool AcceptsFocusFromKeyboard() const override
	{
		return AcceptsFocus();
	}

	void set_label_text(wxString label)
	{
		if (_label == label) {
			return;
		}
		_label = label;
		InvalidateBestSize();
		Refresh();
	}

	wxString label_text() const
	{
		return _label;
	}

	void set_kind(Kind kind)
	{
		if (_kind == kind) {
			return;
		}
		_kind = kind;
		Refresh();
	}

	/** Reserve room for a longer label than the one currently shown, so the
	 *  button does not jump around as its text changes. */
	void set_minimum_label(wxString label)
	{
		_minimum_label = label;
		InvalidateBestSize();
	}

	bool Enable(bool enable = true) override
	{
		auto const changed = wxWindow::Enable(enable);
		if (changed) {
			_hover = false;
			_pressed = false;
			Refresh();
		}
		return changed;
	}

	/** Convenience for the common "call this when clicked" case. */
	void on_click(std::function<void ()> handler)
	{
		Bind(wxEVT_BUTTON, [handler](wxCommandEvent&) { handler(); });
	}

protected:
	wxSize DoGetBestSize() const override
	{
		wxClientDC dc(const_cast<SlangFlatButton*>(this));
		dc.SetFont(slang_ui::font(this, 0, _kind == Kind::PRIMARY));
		auto size = dc.GetTextExtent(_label);
		if (!_minimum_label.IsEmpty()) {
			auto const other = dc.GetTextExtent(_minimum_label);
			size.SetWidth(std::max(size.GetWidth(), other.GetWidth()));
		}
		auto const pad_x = FromDIP(_kind == Kind::GHOST ? 10 : 18);
		auto const pad_y = FromDIP(_kind == Kind::PRIMARY ? 12 : 8);
		return wxSize(size.GetWidth() + pad_x * 2, size.GetHeight() + pad_y * 2);
	}

private:
	void set_hover(bool hover)
	{
		if (!IsThisEnabled() || _hover == hover) {
			return;
		}
		_hover = hover;
		Refresh();
	}

	void left_down()
	{
		if (!IsThisEnabled()) {
			return;
		}
		_pressed = true;
		SetFocus();
		if (!HasCapture()) {
			CaptureMouse();
		}
		Refresh();
	}

	void left_up(wxMouseEvent& ev)
	{
		if (HasCapture()) {
			ReleaseMouse();
		}
		auto const was_pressed = _pressed;
		_pressed = false;
		Refresh();
		if (was_pressed && IsThisEnabled() && wxRect(GetSize()).Contains(ev.GetPosition())) {
			send_click();
		}
	}

	void capture_lost()
	{
		_pressed = false;
		Refresh();
	}

	void key_down(wxKeyEvent& ev)
	{
		if (IsThisEnabled() && (ev.GetKeyCode() == WXK_SPACE || ev.GetKeyCode() == WXK_RETURN || ev.GetKeyCode() == WXK_NUMPAD_ENTER)) {
			send_click();
			return;
		}
		ev.Skip();
	}

	void refresh_focus(bool focused)
	{
		_focused = focused;
		Refresh();
	}

	void send_click()
	{
		wxCommandEvent ev(wxEVT_BUTTON, GetId());
		ev.SetEventObject(this);
		ProcessWindowEvent(ev);
	}

	void paint()
	{
		wxAutoBufferedPaintDC dc(this);
		dc.SetBackground(wxBrush(GetParent()->GetBackgroundColour()));
		dc.Clear();

		std::unique_ptr<wxGraphicsContext> gc(wxGraphicsContext::Create(dc));
		if (!gc) {
			return;
		}
		gc->SetAntialiasMode(wxANTIALIAS_DEFAULT);

		auto const p = slang_ui::palette();
		auto const enabled = IsThisEnabled();
		wxRect const rect(GetSize());
		auto const radius = FromDIP(8);

		wxColour fill = p.card;
		wxColour edge = p.border;
		wxColour label = p.text;

		switch (_kind) {
		case Kind::PRIMARY:
			fill = _pressed ? slang_ui::mix(p.accent_hover, p.accent, 0.7) : (_hover ? p.accent_hover : p.accent);
			edge = fill;
			label = p.on_accent;
			break;
		case Kind::SECONDARY:
			fill = _pressed ? p.accent_soft : (_hover ? slang_ui::mix(p.accent_soft, p.card, 0.55) : p.card);
			edge = _hover ? p.accent : p.border;
			label = p.text;
			break;
		case Kind::GHOST: {
			/* Sit on whatever we are placed on (page or card) so a ghost
			 * button never punches a differently-coloured hole in a card. */
			auto const behind = GetParent()->GetBackgroundColour();
			fill = _hover ? slang_ui::mix(p.accent_soft, behind, 0.6) : behind;
			edge = fill;
			label = _hover ? p.accent : p.muted;
			break;
		}
		case Kind::DANGER:
			fill = _hover ? slang_ui::mix(p.danger, p.card, 0.12) : p.card;
			edge = p.danger;
			label = p.danger;
			break;
		}

		if (!enabled) {
			fill = slang_ui::mix(fill, p.page, 0.45);
			edge = slang_ui::mix(edge, p.page, 0.5);
			label = slang_ui::mix(label, p.page, 0.55);
		}

		slang_ui::rounded_rect(gc.get(), rect, radius, fill, edge);

		if (_focused && enabled) {
			auto const inset = FromDIP(3);
			gc->SetBrush(*wxTRANSPARENT_BRUSH);
			gc->SetPen(wxPen(_kind == Kind::PRIMARY ? p.on_accent : p.accent, 1, wxPENSTYLE_DOT));
			gc->DrawRoundedRectangle(
				rect.x + inset, rect.y + inset, rect.width - inset * 2, rect.height - inset * 2, std::max(2, radius - inset)
				);
		}

		gc->SetFont(gc->CreateFont(slang_ui::font(this, 0, _kind == Kind::PRIMARY), label));
		wxDouble width, height, descent, leading;
		gc->GetTextExtent(_label, &width, &height, &descent, &leading);
		gc->DrawText(_label, (rect.width - width) / 2, (rect.height - height) / 2);
	}

	wxString _label;
	wxString _minimum_label;
	Kind _kind;
	bool _hover = false;
	bool _pressed = false;
	bool _focused = false;
};


/** @class SlangCard
 *  @brief A rounded panel with a numbered step badge, a title, a subtitle and a
 *  body sizer for the step's own controls.
 *
 *  The badge/title/subtitle are painted rather than built from wxStaticTexts so
 *  they can be restyled (accent when the step is the next one to do, muted when
 *  it is complete) without fighting native control background colours -- a
 *  recurring source of grey rectangles on GTK.  The body's own children are
 *  ordinary widgets in body().
 */
class SlangCard : public wxPanel
{
public:
	SlangCard(wxWindow* parent, wxString title, wxString subtitle = {}, int step = 0)
		: wxPanel(parent, wxID_ANY)
		, _title(title)
		, _subtitle(subtitle)
		, _step(step)
	{
		SetBackgroundStyle(wxBG_STYLE_PAINT);
		SetBackgroundColour(slang_ui::palette().card);

		_sizer = new wxBoxSizer(wxVERTICAL);
		_body = new wxBoxSizer(wxVERTICAL);
		_sizer->AddSpacer(header_height());
		_sizer->Add(_body, 1, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, FromDIP(PADDING));
		SetSizer(_sizer);

		Bind(wxEVT_PAINT, [this](wxPaintEvent&) { paint(); });
	}

	/** @return the sizer holding this card's contents. */
	wxSizer* body() const
	{
		return _body;
	}

	void set_title(wxString title)
	{
		if (_title != title) {
			_title = title;
			Refresh();
		}
	}

	void set_subtitle(wxString subtitle)
	{
		if (_subtitle != subtitle) {
			_subtitle = subtitle;
			Refresh();
		}
	}

	/** Mark the step done (tick in the badge) or not. */
	void set_done(bool done)
	{
		if (_done != done) {
			_done = done;
			Refresh();
		}
	}

	/** Highlight this card as the step the user should do next. */
	void set_active(bool active)
	{
		if (_active != active) {
			_active = active;
			Refresh();
		}
	}

private:
	static int constexpr PADDING = 16;

	int badge_size() const
	{
		return _step > 0 ? FromDIP(26) : 0;
	}

	int header_height() const
	{
		auto const lines = _subtitle.IsEmpty() ? 1 : 2;
		return FromDIP(PADDING) + GetCharHeight() * lines + (lines == 2 ? FromDIP(3) : 0) + FromDIP(10);
	}

	void paint()
	{
		wxAutoBufferedPaintDC dc(this);
		dc.SetBackground(wxBrush(GetParent()->GetBackgroundColour()));
		dc.Clear();

		std::unique_ptr<wxGraphicsContext> gc(wxGraphicsContext::Create(dc));
		if (!gc) {
			return;
		}
		gc->SetAntialiasMode(wxANTIALIAS_DEFAULT);

		auto const p = slang_ui::palette();
		wxRect const rect(GetSize());
		auto const radius = FromDIP(12);

		slang_ui::rounded_rect(
			gc.get(), rect, radius, p.card, _active ? p.accent : p.border, _active ? 2 : 1
			);

		auto const pad = FromDIP(PADDING);
		auto x = static_cast<double>(pad);
		auto const top = static_cast<double>(pad);

		if (_step > 0) {
			auto const size = badge_size();
			wxRect const badge(pad, static_cast<int>(top), size, size);
			auto const fill = _done ? p.success : (_active ? p.accent : p.accent_soft);
			auto const label_colour = (_done || _active) ? p.on_accent : p.accent;
			slang_ui::rounded_rect(gc.get(), badge, size / 2.0, fill);

			auto const text = _done ? wxString::FromUTF8("✓") : wxString::Format(char_to_wx("%d"), _step);
			gc->SetFont(gc->CreateFont(slang_ui::font(this, -1, true), label_colour));
			wxDouble width, height, descent, leading;
			gc->GetTextExtent(text, &width, &height, &descent, &leading);
			gc->DrawText(text, badge.x + (size - width) / 2, badge.y + (size - height) / 2);

			x += size + FromDIP(10);
		}

		auto const available = rect.width - x - pad;
		gc->SetFont(gc->CreateFont(slang_ui::font(this, 1, true), p.text));
		slang_ui::draw_text(gc.get(), _title, x, top + FromDIP(1), available);

		if (!_subtitle.IsEmpty()) {
			gc->SetFont(gc->CreateFont(slang_ui::font(this, -1), p.muted));
			slang_ui::draw_text(gc.get(), _subtitle, x, top + GetCharHeight() + FromDIP(4), available);
		}
	}

	wxBoxSizer* _sizer = nullptr;
	wxBoxSizer* _body = nullptr;
	wxString _title;
	wxString _subtitle;
	int _step = 0;
	bool _done = false;
	bool _active = false;
};


/** @class SlangDropArea
 *  @brief A dashed "drop files here, or click to choose" target.
 *
 *  Accepts both a click (which calls the same handler as a drop, with no paths)
 *  and a real file drop.  The owner decides what to do with the paths.
 */
class SlangDropArea : public wxWindow
{
public:
	using Handler = std::function<void (std::vector<boost::filesystem::path>)>;

	SlangDropArea(wxWindow* parent, wxString prompt, wxString hint, Handler handler)
		: wxWindow(parent, wxID_ANY)
		, _prompt(prompt)
		, _hint(hint)
		, _handler(handler)
	{
		SetBackgroundStyle(wxBG_STYLE_PAINT);
		SetDropTarget(new Target(this));
		Bind(wxEVT_PAINT, [this](wxPaintEvent&) { paint(); });
		Bind(wxEVT_ENTER_WINDOW, [this](wxMouseEvent&) { set_hover(true); });
		Bind(wxEVT_LEAVE_WINDOW, [this](wxMouseEvent&) { set_hover(false); });
		Bind(wxEVT_LEFT_UP, [this](wxMouseEvent&) { clicked(); });
		SetCursor(wxCursor(wxCURSOR_HAND));
	}

	void set_prompt(wxString prompt)
	{
		if (_prompt != prompt) {
			_prompt = prompt;
			Refresh();
		}
	}

	void set_hint(wxString hint)
	{
		if (_hint != hint) {
			_hint = hint;
			Refresh();
		}
	}

protected:
	wxSize DoGetBestSize() const override
	{
		return wxSize(FromDIP(320), GetCharHeight() * 2 + FromDIP(52));
	}

private:
	class Target : public wxFileDropTarget
	{
	public:
		explicit Target(SlangDropArea* area)
			: _area(area)
		{}

		bool OnDropFiles(wxCoord, wxCoord, wxArrayString const& filenames) override
		{
			std::vector<boost::filesystem::path> paths;
			for (auto const& name: filenames) {
				paths.push_back(wx_to_std(name));
			}
			_area->set_dragging(false);
			if (paths.empty()) {
				return false;
			}
			/* A disabled area must refuse a DROP too, not just a click.  A drop
			 * target is a separate object from the window's mouse handling, so
			 * Enable(false) greys the area and stops left_down() without saying
			 * anything to wxFileDropTarget -- and this handler adds content to
			 * the film, which is exactly what being disabled during an export
			 * means it must not do. */
			if (!_area->IsThisEnabled()) {
				return false;
			}
			_area->_handler(paths);
			return true;
		}

		wxDragResult OnDragOver(wxCoord, wxCoord, wxDragResult def) override
		{
			if (!_area->IsThisEnabled()) {
				return wxDragNone;
			}
			_area->set_dragging(true);
			return def;
		}

		void OnLeave() override
		{
			_area->set_dragging(false);
		}

	private:
		SlangDropArea* _area;
	};

	void set_hover(bool hover)
	{
		if (_hover != hover) {
			_hover = hover;
			Refresh();
		}
	}

	void set_dragging(bool dragging)
	{
		if (_dragging != dragging) {
			_dragging = dragging;
			Refresh();
		}
	}

	void clicked()
	{
		if (IsThisEnabled()) {
			_handler({});
		}
	}

	void paint()
	{
		wxAutoBufferedPaintDC dc(this);
		dc.SetBackground(wxBrush(GetParent()->GetBackgroundColour()));
		dc.Clear();

		std::unique_ptr<wxGraphicsContext> gc(wxGraphicsContext::Create(dc));
		if (!gc) {
			return;
		}
		gc->SetAntialiasMode(wxANTIALIAS_DEFAULT);

		auto const p = slang_ui::palette();
		wxRect const rect(GetSize());
		auto const active = _dragging || _hover;
		auto const fill = _dragging ? p.accent_soft : (_hover ? slang_ui::mix(p.accent_soft, p.card_sunken, 0.4) : p.card_sunken);

		gc->SetBrush(wxBrush(fill));
		gc->SetPen(wxPen(active ? p.accent : p.border, _dragging ? 2 : 1, wxPENSTYLE_SHORT_DASH));
		gc->DrawRoundedRectangle(1, 1, rect.width - 2, rect.height - 2, FromDIP(10));

		wxDouble width, height, descent, leading;
		auto const centre_x = rect.width / 2.0;
		auto const total = GetCharHeight() * 2 + FromDIP(6);
		auto y = (rect.height - total) / 2.0;

		gc->SetFont(gc->CreateFont(slang_ui::font(this, 0, true), active ? p.accent : p.text));
		gc->GetTextExtent(_prompt, &width, &height, &descent, &leading);
		gc->DrawText(_prompt, centre_x - width / 2, y);
		y += height + FromDIP(4);

		gc->SetFont(gc->CreateFont(slang_ui::font(this, -1), p.muted));
		gc->GetTextExtent(_hint, &width, &height, &descent, &leading);
		gc->DrawText(_hint, centre_x - width / 2, y);
	}

	wxString _prompt;
	wxString _hint;
	Handler _handler;
	bool _hover = false;
	bool _dragging = false;
};


#endif

#endif
