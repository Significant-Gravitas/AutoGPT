# The property page to define generic IDE options for Pythonwin

import win32con
import win32ui
from pywin.framework import interact
from pywin.mfc import dialog

buttonControlMap = {
    win32ui.IDC_BUTTON1: win32ui.IDC_EDIT1,
    win32ui.IDC_BUTTON2: win32ui.IDC_EDIT2,
    win32ui.IDC_BUTTON3: win32ui.IDC_EDIT3,
}


class OptionsPropPage(dialog.PropertyPage):
    def __init__(self):
        dialog.PropertyPage.__init__(self, win32ui.IDD_PP_IDE)
        self.AddDDX(win32ui.IDC_CHECK1, "bShowAtStartup")
        self.AddDDX(win32ui.IDC_CHECK2, "bDocking")
        self.AddDDX(win32ui.IDC_EDIT4, "MRUSize", "i")

    def OnInitDialog(self):
        edit = self.GetDlgItem(win32ui.IDC_EDIT1)
        format = eval(
            win32ui.GetProfileVal(
                interact.sectionProfile,
                interact.STYLE_INTERACTIVE_PROMPT,
                str(interact.formatInput),
            )
        )
        edit.SetDefaultCharFormat(format)
        edit.SetWindowText("Input Text")

        edit = self.GetDlgItem(win32ui.IDC_EDIT2)
        format = eval(
            win32ui.GetProfileVal(
                interact.sectionProfile,
                interact.STYLE_INTERACTIVE_OUTPUT,
                str(interact.formatOutput),
            )
        )
        edit.SetDefaultCharFormat(format)
        edit.SetWindowText("Output Text")

        edit = self.GetDlgItem(win32ui.IDC_EDIT3)
        format = eval(
            win32ui.GetProfileVal(
                interact.sectionProfile,
                interact.STYLE_INTERACTIVE_ERROR,
                str(interact.formatOutputError),
            )
        )
        edit.SetDefaultCharFormat(format)
        edit.SetWindowText("Error Text")

        self["bShowAtStartup"] = interact.LoadPreference("Show at startup", 1)
        self["bDocking"] = interact.LoadPreference("Docking", 0)
        self["MRUSize"] = win32ui.GetProfileVal("Settings", "Recent File List Size", 10)

        # Hook the button clicks.
        self.HookCommand(self.HandleCharFormatChange, win32ui.IDC_BUTTON1)
        self.HookCommand(self.HandleCharFormatChange, win32ui.IDC_BUTTON2)
        self.HookCommand(self.HandleCharFormatChange, win32ui.IDC_BUTTON3)

        # Ensure the spin control remains in range.
        spinner = self.GetDlgItem(win32ui.IDC_SPIN1)
        spinner.SetRange(1, 16)

        return dialog.PropertyPage.OnInitDialog(self)

    # Called to save away the new format tuple for the specified item.
    def HandleCharFormatChange(self, id, code):
        if code == win32con.BN_CLICKED:
            editId = buttonControlMap.get(id)
            assert editId is not None, "Format button has no associated edit control"
            editControl = self.GetDlgItem(editId)
            existingFormat = editControl.GetDefaultCharFormat()
            flags = win32con.CF_SCREENFONTS
            d = win32ui.CreateFontDialog(existingFormat, flags, None, self)
            if d.DoModal() == win32con.IDOK:
                cf = d.GetCharFormat()
                editControl.SetDefaultCharFormat(cf)
                self.SetModified(1)
            return 0  # We handled this fully!

    def OnOK(self):
        # Handle the edit controls - get all the fonts, put them back into interact, then
        # get interact to save its stuff!
        controlAttrs = [
            (win32ui.IDC_EDIT1, interact.STYLE_INTERACTIVE_PROMPT),
            (win32ui.IDC_EDIT2, interact.STYLE_INTERACTIVE_OUTPUT),
            (win32ui.IDC_EDIT3, interact.STYLE_INTERACTIVE_ERROR),
        ]
        for id, key in controlAttrs:
            control = self.GetDlgItem(id)
            fmt = control.GetDefaultCharFormat()
            win32ui.WriteProfileVal(interact.sectionProfile, key, str(fmt))

        # Save the other interactive window options.
        interact.SavePreference("Show at startup", self["bShowAtStartup"])
        interact.SavePreference("Docking", self["bDocking"])

        # And the other options.
        win32ui.WriteProfileVal("Settings", "Recent File List Size", self["MRUSize"])

        return 1

    def ChangeFormat(self, fmtAttribute, fmt):
        dlg = win32ui.CreateFontDialog(fmt)
        if dlg.DoModal() != win32con.IDOK:
            return None
        return dlg.GetCharFormat()

    def OnFormatTitle(self, command, code):
        fmt = self.GetFormat(interact.formatTitle)
        if fmt:
            formatTitle = fmt
            SaveFontPreferences()

    def OnFormatInput(self, command, code):
        global formatInput
        fmt = self.GetFormat(formatInput)
        if fmt:
            formatInput = fmt
            SaveFontPreferences()

    def OnFormatOutput(self, command, code):
        global formatOutput
        fmt = self.GetFormat(formatOutput)
        if fmt:
            formatOutput = fmt
            SaveFontPreferences()

    def OnFormatError(self, command, code):
        global formatOutputError
        fmt = self.GetFormat(formatOutputError)
        if fmt:
            formatOutputError = fmt
            SaveFontPreferences()
