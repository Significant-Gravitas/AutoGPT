import pywin.scintilla.config
import win32api
import win32con
import win32ui
from pywin.framework.editor import (
    DeleteEditorOption,
    GetEditorFontOption,
    GetEditorOption,
    SetEditorFontOption,
    SetEditorOption,
    defaultCharacterFormat,
    editorTemplate,
)
from pywin.mfc import dialog

from . import document

# The standard 16 color VGA palette should always be possible
paletteVGA = (
    ("Black", 0, 0, 0),
    ("Navy", 0, 0, 128),
    ("Green", 0, 128, 0),
    ("Cyan", 0, 128, 128),
    ("Maroon", 128, 0, 0),
    ("Purple", 128, 0, 128),
    ("Olive", 128, 128, 0),
    ("Gray", 128, 128, 128),
    ("Silver", 192, 192, 192),
    ("Blue", 0, 0, 255),
    ("Lime", 0, 255, 0),
    ("Aqua", 0, 255, 255),
    ("Red", 255, 0, 0),
    ("Fuchsia", 255, 0, 255),
    ("Yellow", 255, 255, 0),
    ("White", 255, 255, 255),
)


######################################################
#
# Property Page for editor options
#
class EditorPropertyPage(dialog.PropertyPage):
    def __init__(self):
        dialog.PropertyPage.__init__(self, win32ui.IDD_PP_EDITOR)
        self.autooptions = []
        self._AddEditorOption(win32ui.IDC_AUTO_RELOAD, "i", "Auto Reload", 1)
        self._AddEditorOption(
            win32ui.IDC_COMBO1, "i", "Backup Type", document.BAK_DOT_BAK_BAK_DIR
        )
        self._AddEditorOption(
            win32ui.IDC_AUTOCOMPLETE, "i", "Autocomplete Attributes", 1
        )
        self._AddEditorOption(win32ui.IDC_CALLTIPS, "i", "Show Call Tips", 1)
        self._AddEditorOption(
            win32ui.IDC_MARGIN_LINENUMBER, "i", "Line Number Margin Width", 0
        )
        self._AddEditorOption(win32ui.IDC_RADIO1, "i", "MarkersInMargin", None)
        self._AddEditorOption(
            win32ui.IDC_MARGIN_MARKER, "i", "Marker Margin Width", None
        )
        self["Marker Margin Width"] = GetEditorOption("Marker Margin Width", 16)

        # Folding
        self._AddEditorOption(win32ui.IDC_MARGIN_FOLD, "i", "Fold Margin Width", 12)
        self._AddEditorOption(win32ui.IDC_FOLD_ENABLE, "i", "Enable Folding", 1)
        self._AddEditorOption(win32ui.IDC_FOLD_ON_OPEN, "i", "Fold On Open", 0)
        self._AddEditorOption(win32ui.IDC_FOLD_SHOW_LINES, "i", "Fold Lines", 1)

        # Right edge.
        self._AddEditorOption(
            win32ui.IDC_RIGHTEDGE_ENABLE, "i", "Right Edge Enabled", 0
        )
        self._AddEditorOption(
            win32ui.IDC_RIGHTEDGE_COLUMN, "i", "Right Edge Column", 75
        )

        # Source control, etc
        self.AddDDX(win32ui.IDC_VSS_INTEGRATE, "bVSS")
        self.AddDDX(win32ui.IDC_KEYBOARD_CONFIG, "Configs", "l")
        self["Configs"] = pywin.scintilla.config.find_config_files()

    def _AddEditorOption(self, idd, typ, optionName, defaultVal):
        self.AddDDX(idd, optionName, typ)
        # some options are "derived" - ie, can be implied from others
        # (eg, "view markers in background" is implied from "markerMarginWidth==0"
        # So we don't actually store these values, but they do still get DDX support.
        if defaultVal is not None:
            self[optionName] = GetEditorOption(optionName, defaultVal)
            self.autooptions.append((optionName, defaultVal))

    def OnInitDialog(self):
        for name, val in self.autooptions:
            self[name] = GetEditorOption(name, val)

        # Note that these MUST be in the same order as the BAK constants.
        cbo = self.GetDlgItem(win32ui.IDC_COMBO1)
        cbo.AddString("None")
        cbo.AddString(".BAK File")
        cbo.AddString("TEMP dir")
        cbo.AddString("Own dir")

        # Source Safe
        bVSS = (
            GetEditorOption("Source Control Module", "") == "pywin.framework.editor.vss"
        )
        self["bVSS"] = bVSS

        edit = self.GetDlgItem(win32ui.IDC_RIGHTEDGE_SAMPLE)
        edit.SetWindowText("Sample Color")

        rc = dialog.PropertyPage.OnInitDialog(self)

        try:
            self.GetDlgItem(win32ui.IDC_KEYBOARD_CONFIG).SelectString(
                -1, GetEditorOption("Keyboard Config", "default")
            )
        except win32ui.error:
            import traceback

            traceback.print_exc()
            pass

        self.HookCommand(self.OnButSimple, win32ui.IDC_FOLD_ENABLE)
        self.HookCommand(self.OnButSimple, win32ui.IDC_RADIO1)
        self.HookCommand(self.OnButSimple, win32ui.IDC_RADIO2)
        self.HookCommand(self.OnButSimple, win32ui.IDC_RIGHTEDGE_ENABLE)
        self.HookCommand(self.OnButEdgeColor, win32ui.IDC_RIGHTEDGE_DEFINE)

        butMarginEnabled = self["Marker Margin Width"] > 0
        self.GetDlgItem(win32ui.IDC_RADIO1).SetCheck(butMarginEnabled)
        self.GetDlgItem(win32ui.IDC_RADIO2).SetCheck(not butMarginEnabled)

        self.edgeColor = self.initialEdgeColor = GetEditorOption(
            "Right Edge Color", win32api.RGB(0xEF, 0xEF, 0xEF)
        )
        for spinner_id in (win32ui.IDC_SPIN1, win32ui.IDC_SPIN2, win32ui.IDC_SPIN3):
            spinner = self.GetDlgItem(spinner_id)
            spinner.SetRange(0, 100)
        self.UpdateUIForState()

        return rc

    def OnButSimple(self, id, code):
        if code == win32con.BN_CLICKED:
            self.UpdateUIForState()

    def OnButEdgeColor(self, id, code):
        if code == win32con.BN_CLICKED:
            d = win32ui.CreateColorDialog(self.edgeColor, 0, self)
            # Ensure the current color is a custom color (as it may not be in the swatch)
            # plus some other nice gray scales.
            ccs = [self.edgeColor]
            for c in range(0xEF, 0x4F, -0x10):
                ccs.append(win32api.RGB(c, c, c))
            d.SetCustomColors(ccs)
            if d.DoModal() == win32con.IDOK:
                self.edgeColor = d.GetColor()
                self.UpdateUIForState()

    def UpdateUIForState(self):
        folding = self.GetDlgItem(win32ui.IDC_FOLD_ENABLE).GetCheck()
        self.GetDlgItem(win32ui.IDC_FOLD_ON_OPEN).EnableWindow(folding)
        self.GetDlgItem(win32ui.IDC_FOLD_SHOW_LINES).EnableWindow(folding)

        widthEnabled = self.GetDlgItem(win32ui.IDC_RADIO1).GetCheck()
        self.GetDlgItem(win32ui.IDC_MARGIN_MARKER).EnableWindow(widthEnabled)
        self.UpdateData()  # Ensure self[] is up to date with the control data.
        if widthEnabled and self["Marker Margin Width"] == 0:
            self["Marker Margin Width"] = 16
            self.UpdateData(0)  # Ensure control up to date with self[]

        # Right edge
        edgeEnabled = self.GetDlgItem(win32ui.IDC_RIGHTEDGE_ENABLE).GetCheck()
        self.GetDlgItem(win32ui.IDC_RIGHTEDGE_COLUMN).EnableWindow(edgeEnabled)
        self.GetDlgItem(win32ui.IDC_RIGHTEDGE_SAMPLE).EnableWindow(edgeEnabled)
        self.GetDlgItem(win32ui.IDC_RIGHTEDGE_DEFINE).EnableWindow(edgeEnabled)

        edit = self.GetDlgItem(win32ui.IDC_RIGHTEDGE_SAMPLE)
        edit.SetBackgroundColor(0, self.edgeColor)

    def OnOK(self):
        for name, defVal in self.autooptions:
            SetEditorOption(name, self[name])
        # Margin width gets handled differently.
        if self["MarkersInMargin"] == 0:
            SetEditorOption("Marker Margin Width", self["Marker Margin Width"])
        else:
            SetEditorOption("Marker Margin Width", 0)
        if self.edgeColor != self.initialEdgeColor:
            SetEditorOption("Right Edge Color", self.edgeColor)
        if self["bVSS"]:
            SetEditorOption("Source Control Module", "pywin.framework.editor.vss")
        else:
            if (
                GetEditorOption("Source Control Module", "")
                == "pywin.framework.editor.vss"
            ):
                SetEditorOption("Source Control Module", "")
        # Keyboard config
        configname = self.GetDlgItem(win32ui.IDC_KEYBOARD_CONFIG).GetWindowText()
        if configname:
            if configname == "default":
                DeleteEditorOption("Keyboard Config")
            else:
                SetEditorOption("Keyboard Config", configname)

            import pywin.scintilla.view

            pywin.scintilla.view.LoadConfiguration()

        # Now tell all views we have changed.
        ##		for doc in editorTemplate.GetDocumentList():
        ##			for view in doc.GetAllViews():
        ##				try:
        ##					fn = view.OnConfigChange
        ##				except AttributeError:
        ##					continue
        ##				fn()
        return 1


class EditorWhitespacePropertyPage(dialog.PropertyPage):
    def __init__(self):
        dialog.PropertyPage.__init__(self, win32ui.IDD_PP_TABS)
        self.autooptions = []
        self._AddEditorOption(win32ui.IDC_TAB_SIZE, "i", "Tab Size", 4)
        self._AddEditorOption(win32ui.IDC_INDENT_SIZE, "i", "Indent Size", 4)
        self._AddEditorOption(win32ui.IDC_USE_SMART_TABS, "i", "Smart Tabs", 1)
        self._AddEditorOption(win32ui.IDC_VIEW_WHITESPACE, "i", "View Whitespace", 0)
        self._AddEditorOption(win32ui.IDC_VIEW_EOL, "i", "View EOL", 0)
        self._AddEditorOption(
            win32ui.IDC_VIEW_INDENTATIONGUIDES, "i", "View Indentation Guides", 0
        )

    def _AddEditorOption(self, idd, typ, optionName, defaultVal):
        self.AddDDX(idd, optionName, typ)
        self[optionName] = GetEditorOption(optionName, defaultVal)
        self.autooptions.append((optionName, defaultVal))

    def OnInitDialog(self):
        for name, val in self.autooptions:
            self[name] = GetEditorOption(name, val)

        rc = dialog.PropertyPage.OnInitDialog(self)

        idc = win32ui.IDC_TABTIMMY_NONE
        if GetEditorOption("Use Tab Timmy", 1):
            idc = win32ui.IDC_TABTIMMY_IND
        self.GetDlgItem(idc).SetCheck(1)

        idc = win32ui.IDC_RADIO1
        if GetEditorOption("Use Tabs", 0):
            idc = win32ui.IDC_USE_TABS
        self.GetDlgItem(idc).SetCheck(1)

        tt_color = GetEditorOption("Tab Timmy Color", win32api.RGB(0xFF, 0, 0))
        self.cbo = self.GetDlgItem(win32ui.IDC_COMBO1)
        for c in paletteVGA:
            self.cbo.AddString(c[0])
        sel = 0
        for c in paletteVGA:
            if tt_color == win32api.RGB(c[1], c[2], c[3]):
                break
            sel = sel + 1
        else:
            sel = -1
        self.cbo.SetCurSel(sel)
        self.HookCommand(self.OnButSimple, win32ui.IDC_TABTIMMY_NONE)
        self.HookCommand(self.OnButSimple, win32ui.IDC_TABTIMMY_IND)
        self.HookCommand(self.OnButSimple, win32ui.IDC_TABTIMMY_BG)
        # Set ranges for the spinners.
        for spinner_id in [win32ui.IDC_SPIN1, win32ui.IDC_SPIN2]:
            spinner = self.GetDlgItem(spinner_id)
            spinner.SetRange(1, 16)
        return rc

    def OnButSimple(self, id, code):
        if code == win32con.BN_CLICKED:
            self.UpdateUIForState()

    def UpdateUIForState(self):
        timmy = self.GetDlgItem(win32ui.IDC_TABTIMMY_NONE).GetCheck()
        self.GetDlgItem(win32ui.IDC_COMBO1).EnableWindow(not timmy)

    def OnOK(self):
        for name, defVal in self.autooptions:
            SetEditorOption(name, self[name])

        SetEditorOption("Use Tabs", self.GetDlgItem(win32ui.IDC_USE_TABS).GetCheck())

        SetEditorOption(
            "Use Tab Timmy", self.GetDlgItem(win32ui.IDC_TABTIMMY_IND).GetCheck()
        )
        c = paletteVGA[self.cbo.GetCurSel()]
        SetEditorOption("Tab Timmy Color", win32api.RGB(c[1], c[2], c[3]))

        return 1


def testpp():
    ps = dialog.PropertySheet("Editor Options")
    ps.AddPage(EditorWhitespacePropertyPage())
    ps.DoModal()


if __name__ == "__main__":
    testpp()
