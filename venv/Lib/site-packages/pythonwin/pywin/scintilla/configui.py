import win32api
import win32con
import win32ui
from pywin.mfc import dialog

# Used to indicate that style should use default color
from win32con import CLR_INVALID

from . import scintillacon

######################################################
# Property Page for syntax formatting options

# The standard 16 color VGA palette should always be possible
paletteVGA = (
    ("Black", win32api.RGB(0, 0, 0)),
    ("Navy", win32api.RGB(0, 0, 128)),
    ("Green", win32api.RGB(0, 128, 0)),
    ("Cyan", win32api.RGB(0, 128, 128)),
    ("Maroon", win32api.RGB(128, 0, 0)),
    ("Purple", win32api.RGB(128, 0, 128)),
    ("Olive", win32api.RGB(128, 128, 0)),
    ("Gray", win32api.RGB(128, 128, 128)),
    ("Silver", win32api.RGB(192, 192, 192)),
    ("Blue", win32api.RGB(0, 0, 255)),
    ("Lime", win32api.RGB(0, 255, 0)),
    ("Aqua", win32api.RGB(0, 255, 255)),
    ("Red", win32api.RGB(255, 0, 0)),
    ("Fuchsia", win32api.RGB(255, 0, 255)),
    ("Yellow", win32api.RGB(255, 255, 0)),
    ("White", win32api.RGB(255, 255, 255)),
    # and a few others will generally be possible.
    ("DarkGrey", win32api.RGB(64, 64, 64)),
    ("PurpleBlue", win32api.RGB(64, 64, 192)),
    ("DarkGreen", win32api.RGB(0, 96, 0)),
    ("DarkOlive", win32api.RGB(128, 128, 64)),
    ("MediumBlue", win32api.RGB(0, 0, 192)),
    ("DarkNavy", win32api.RGB(0, 0, 96)),
    ("Magenta", win32api.RGB(96, 0, 96)),
    ("OffWhite", win32api.RGB(255, 255, 220)),
    ("LightPurple", win32api.RGB(220, 220, 255)),
    ("<Default>", win32con.CLR_INVALID),
)


class ScintillaFormatPropertyPage(dialog.PropertyPage):
    def __init__(self, scintillaClass=None, caption=0):
        self.scintillaClass = scintillaClass
        dialog.PropertyPage.__init__(self, win32ui.IDD_PP_FORMAT, caption=caption)

    def OnInitDialog(self):
        try:
            if self.scintillaClass is None:
                from . import control

                sc = control.CScintillaEdit
            else:
                sc = self.scintillaClass

            self.scintilla = sc()
            style = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.ES_MULTILINE
            # Convert the rect size
            rect = self.MapDialogRect((5, 5, 120, 75))
            self.scintilla.CreateWindow(style, rect, self, 111)
            self.HookNotify(self.OnBraceMatch, scintillacon.SCN_CHECKBRACE)
            self.scintilla.HookKeyStroke(self.OnEsc, 27)
            self.scintilla.SCISetViewWS(1)
            self.pos_bstart = self.pos_bend = self.pos_bbad = 0

            colorizer = self.scintilla._GetColorizer()
            text = colorizer.GetSampleText()
            items = text.split("|", 2)
            pos = len(items[0])
            self.scintilla.SCIAddText("".join(items))
            self.scintilla.SetSel(pos, pos)
            self.scintilla.ApplyFormattingStyles()
            self.styles = self.scintilla._GetColorizer().styles

            self.cbo = self.GetDlgItem(win32ui.IDC_COMBO1)
            for c in paletteVGA:
                self.cbo.AddString(c[0])

            self.cboBoldItalic = self.GetDlgItem(win32ui.IDC_COMBO2)
            for item in ("Bold Italic", "Bold", "Italic", "Regular"):
                self.cboBoldItalic.InsertString(0, item)

            self.butIsDefault = self.GetDlgItem(win32ui.IDC_CHECK1)
            self.butIsDefaultBackground = self.GetDlgItem(win32ui.IDC_CHECK2)
            self.listbox = self.GetDlgItem(win32ui.IDC_LIST1)
            self.HookCommand(self.OnListCommand, win32ui.IDC_LIST1)
            names = list(self.styles.keys())
            names.sort()
            for name in names:
                if self.styles[name].aliased is None:
                    self.listbox.AddString(name)
            self.listbox.SetCurSel(0)

            idc = win32ui.IDC_RADIO1
            if not self.scintilla._GetColorizer().bUseFixed:
                idc = win32ui.IDC_RADIO2
            self.GetDlgItem(idc).SetCheck(1)
            self.UpdateUIForStyle(self.styles[names[0]])

            self.scintilla.HookFormatter(self)
            self.HookCommand(self.OnButDefaultFixedFont, win32ui.IDC_BUTTON1)
            self.HookCommand(self.OnButDefaultPropFont, win32ui.IDC_BUTTON2)
            self.HookCommand(self.OnButThisFont, win32ui.IDC_BUTTON3)
            self.HookCommand(self.OnButUseDefaultFont, win32ui.IDC_CHECK1)
            self.HookCommand(self.OnButThisBackground, win32ui.IDC_BUTTON4)
            self.HookCommand(self.OnButUseDefaultBackground, win32ui.IDC_CHECK2)
            self.HookCommand(self.OnStyleUIChanged, win32ui.IDC_COMBO1)
            self.HookCommand(self.OnStyleUIChanged, win32ui.IDC_COMBO2)
            self.HookCommand(self.OnButFixedOrDefault, win32ui.IDC_RADIO1)
            self.HookCommand(self.OnButFixedOrDefault, win32ui.IDC_RADIO2)
        except:
            import traceback

            traceback.print_exc()

    def OnEsc(self, ch):
        self.GetParent().EndDialog(win32con.IDCANCEL)

    def OnBraceMatch(self, std, extra):
        import pywin.scintilla.view

        pywin.scintilla.view.DoBraceMatch(self.scintilla)

    def GetSelectedStyle(self):
        return self.styles[self.listbox.GetText(self.listbox.GetCurSel())]

    def _DoButDefaultFont(self, extra_flags, attr):
        baseFormat = getattr(self.scintilla._GetColorizer(), attr)
        flags = (
            extra_flags
            | win32con.CF_SCREENFONTS
            | win32con.CF_EFFECTS
            | win32con.CF_FORCEFONTEXIST
        )
        d = win32ui.CreateFontDialog(baseFormat, flags, None, self)
        if d.DoModal() == win32con.IDOK:
            setattr(self.scintilla._GetColorizer(), attr, d.GetCharFormat())
            self.OnStyleUIChanged(0, win32con.BN_CLICKED)

    def OnButDefaultFixedFont(self, id, code):
        if code == win32con.BN_CLICKED:
            self._DoButDefaultFont(win32con.CF_FIXEDPITCHONLY, "baseFormatFixed")
            return 1

    def OnButDefaultPropFont(self, id, code):
        if code == win32con.BN_CLICKED:
            self._DoButDefaultFont(win32con.CF_SCALABLEONLY, "baseFormatProp")
            return 1

    def OnButFixedOrDefault(self, id, code):
        if code == win32con.BN_CLICKED:
            bUseFixed = id == win32ui.IDC_RADIO1
            self.GetDlgItem(win32ui.IDC_RADIO1).GetCheck() != 0
            self.scintilla._GetColorizer().bUseFixed = bUseFixed
            self.scintilla.ApplyFormattingStyles(0)
            return 1

    def OnButThisFont(self, id, code):
        if code == win32con.BN_CLICKED:
            flags = (
                win32con.CF_SCREENFONTS
                | win32con.CF_EFFECTS
                | win32con.CF_FORCEFONTEXIST
            )
            style = self.GetSelectedStyle()
            # If the selected style is based on the default, we need to apply
            # the default to it.
            def_format = self.scintilla._GetColorizer().GetDefaultFormat()
            format = style.GetCompleteFormat(def_format)
            d = win32ui.CreateFontDialog(format, flags, None, self)
            if d.DoModal() == win32con.IDOK:
                style.format = d.GetCharFormat()
                self.scintilla.ApplyFormattingStyles(0)
            return 1

    def OnButUseDefaultFont(self, id, code):
        if code == win32con.BN_CLICKED:
            isDef = self.butIsDefault.GetCheck()
            self.GetDlgItem(win32ui.IDC_BUTTON3).EnableWindow(not isDef)
            if isDef:  # Being reset to the default font.
                style = self.GetSelectedStyle()
                style.ForceAgainstDefault()
                self.UpdateUIForStyle(style)
                self.scintilla.ApplyFormattingStyles(0)
            else:
                # User wants to override default -
                # do nothing!
                pass

    def OnButThisBackground(self, id, code):
        if code == win32con.BN_CLICKED:
            style = self.GetSelectedStyle()
            bg = win32api.RGB(0xFF, 0xFF, 0xFF)
            if style.background != CLR_INVALID:
                bg = style.background
            d = win32ui.CreateColorDialog(bg, 0, self)
            if d.DoModal() == win32con.IDOK:
                style.background = d.GetColor()
                self.scintilla.ApplyFormattingStyles(0)
            return 1

    def OnButUseDefaultBackground(self, id, code):
        if code == win32con.BN_CLICKED:
            isDef = self.butIsDefaultBackground.GetCheck()
            self.GetDlgItem(win32ui.IDC_BUTTON4).EnableWindow(not isDef)
            if isDef:  # Being reset to the default color
                style = self.GetSelectedStyle()
                style.background = style.default_background
                self.UpdateUIForStyle(style)
                self.scintilla.ApplyFormattingStyles(0)
            else:
                # User wants to override default -
                # do nothing!
                pass

    def OnListCommand(self, id, code):
        if code == win32con.LBN_SELCHANGE:
            style = self.GetSelectedStyle()
            self.UpdateUIForStyle(style)
        return 1

    def UpdateUIForStyle(self, style):
        format = style.format
        sel = 0
        for c in paletteVGA:
            if format[4] == c[1]:
                # 				print "Style", style.name, "is", c[0]
                break
            sel = sel + 1
        else:
            sel = -1
        self.cbo.SetCurSel(sel)
        self.butIsDefault.SetCheck(style.IsBasedOnDefault())
        self.GetDlgItem(win32ui.IDC_BUTTON3).EnableWindow(not style.IsBasedOnDefault())

        self.butIsDefaultBackground.SetCheck(
            style.background == style.default_background
        )
        self.GetDlgItem(win32ui.IDC_BUTTON4).EnableWindow(
            style.background != style.default_background
        )

        bold = format[1] & win32con.CFE_BOLD != 0
        italic = format[1] & win32con.CFE_ITALIC != 0
        self.cboBoldItalic.SetCurSel(bold * 2 + italic)

    def OnStyleUIChanged(self, id, code):
        if code in [win32con.BN_CLICKED, win32con.CBN_SELCHANGE]:
            style = self.GetSelectedStyle()
            self.ApplyUIFormatToStyle(style)
            self.scintilla.ApplyFormattingStyles(0)
            return 0
        return 1

    def ApplyUIFormatToStyle(self, style):
        format = style.format
        color = paletteVGA[self.cbo.GetCurSel()]
        effect = 0
        sel = self.cboBoldItalic.GetCurSel()
        if sel == 0:
            effect = 0
        elif sel == 1:
            effect = win32con.CFE_ITALIC
        elif sel == 2:
            effect = win32con.CFE_BOLD
        else:
            effect = win32con.CFE_BOLD | win32con.CFE_ITALIC
        maskFlags = (
            format[0] | win32con.CFM_COLOR | win32con.CFM_BOLD | win32con.CFM_ITALIC
        )
        style.format = (
            maskFlags,
            effect,
            style.format[2],
            style.format[3],
            color[1],
        ) + style.format[5:]

    def OnOK(self):
        self.scintilla._GetColorizer().SavePreferences()
        return 1


def test():
    page = ColorEditorPropertyPage()
    sheet = pywin.mfc.dialog.PropertySheet("Test")
    sheet.AddPage(page)
    sheet.CreateWindow()
