import win32ui
from pywin.mfc import dialog

from . import dbgcon


class DebuggerOptionsPropPage(dialog.PropertyPage):
    def __init__(self):
        dialog.PropertyPage.__init__(self, win32ui.IDD_PP_DEBUGGER)

    def OnInitDialog(self):
        options = self.options = dbgcon.LoadDebuggerOptions()
        self.AddDDX(win32ui.IDC_CHECK1, dbgcon.OPT_HIDE)
        self[dbgcon.OPT_STOP_EXCEPTIONS] = options[dbgcon.OPT_STOP_EXCEPTIONS]
        self.AddDDX(win32ui.IDC_CHECK2, dbgcon.OPT_STOP_EXCEPTIONS)
        self[dbgcon.OPT_HIDE] = options[dbgcon.OPT_HIDE]
        return dialog.PropertyPage.OnInitDialog(self)

    def OnOK(self):
        self.UpdateData()
        dirty = 0
        for key, val in list(self.items()):
            if key in self.options:
                if self.options[key] != val:
                    self.options[key] = val
                    dirty = 1
        if dirty:
            dbgcon.SaveDebuggerOptions(self.options)
        # If there is a debugger open, then set its options.
        import pywin.debugger

        if pywin.debugger.currentDebugger is not None:
            pywin.debugger.currentDebugger.options = self.options
        return 1
