# find.py - Find and Replace
import afxres
import win32api
import win32con
import win32ui
from pywin.framework import scriptutils
from pywin.mfc import dialog

FOUND_NOTHING = 0
FOUND_NORMAL = 1
FOUND_LOOPED_BACK = 2
FOUND_NEXT_FILE = 3


class SearchParams:
    def __init__(self, other=None):
        if other is None:
            self.__dict__["findText"] = ""
            self.__dict__["replaceText"] = ""
            self.__dict__["matchCase"] = 0
            self.__dict__["matchWords"] = 0
            self.__dict__["acrossFiles"] = 0
            self.__dict__["remember"] = 1
            self.__dict__["sel"] = (-1, -1)
            self.__dict__["keepDialogOpen"] = 0
        else:
            self.__dict__.update(other.__dict__)

    # Helper so we cant misspell attributes :-)
    def __setattr__(self, attr, val):
        if not hasattr(self, attr):
            raise AttributeError(attr)
        self.__dict__[attr] = val


curDialog = None
lastSearch = defaultSearch = SearchParams()
searchHistory = []


def ShowFindDialog():
    _ShowDialog(FindDialog)


def ShowReplaceDialog():
    _ShowDialog(ReplaceDialog)


def _ShowDialog(dlgClass):
    global curDialog
    if curDialog is not None:
        if curDialog.__class__ != dlgClass:
            curDialog.DestroyWindow()
            curDialog = None
        else:
            curDialog.SetFocus()
    if curDialog is None:
        curDialog = dlgClass()
        curDialog.CreateWindow()


def FindNext():
    params = SearchParams(lastSearch)
    params.sel = (-1, -1)
    if not params.findText:
        ShowFindDialog()
    else:
        return _FindIt(None, params)


def _GetControl(control=None):
    if control is None:
        control = scriptutils.GetActiveEditControl()
    return control


def _FindIt(control, searchParams):
    global lastSearch, defaultSearch
    control = _GetControl(control)
    if control is None:
        return FOUND_NOTHING

    # Move to the next char, so we find the next one.
    flags = 0
    if searchParams.matchWords:
        flags = flags | win32con.FR_WHOLEWORD
    if searchParams.matchCase:
        flags = flags | win32con.FR_MATCHCASE
    if searchParams.sel == (-1, -1):
        sel = control.GetSel()
        # If the position is the same as we found last time,
        # then we assume it is a "FindNext"
        if sel == lastSearch.sel:
            sel = sel[0] + 1, sel[0] + 1
    else:
        sel = searchParams.sel

    if sel[0] == sel[1]:
        sel = sel[0], control.GetTextLength()

    rc = FOUND_NOTHING
    # (Old edit control will fail here!)
    posFind, foundSel = control.FindText(flags, sel, searchParams.findText)
    lastSearch = SearchParams(searchParams)
    if posFind >= 0:
        rc = FOUND_NORMAL
        lineno = control.LineFromChar(posFind)
        control.SCIEnsureVisible(lineno)
        control.SetSel(foundSel)
        control.SetFocus()
        win32ui.SetStatusText(win32ui.LoadString(afxres.AFX_IDS_IDLEMESSAGE))
    if rc == FOUND_NOTHING and lastSearch.acrossFiles:
        # Loop around all documents.  First find this document.
        try:
            try:
                doc = control.GetDocument()
            except AttributeError:
                try:
                    doc = control.GetParent().GetDocument()
                except AttributeError:
                    print("Cant find a document for the control!")
                    doc = None
            if doc is not None:
                template = doc.GetDocTemplate()
                alldocs = template.GetDocumentList()
                mypos = lookpos = alldocs.index(doc)
                while 1:
                    lookpos = (lookpos + 1) % len(alldocs)
                    if lookpos == mypos:
                        break
                    view = alldocs[lookpos].GetFirstView()
                    posFind, foundSel = view.FindText(
                        flags, (0, view.GetTextLength()), searchParams.findText
                    )
                    if posFind >= 0:
                        nChars = foundSel[1] - foundSel[0]
                        lineNo = view.LineFromChar(posFind)  # zero based.
                        lineStart = view.LineIndex(lineNo)
                        colNo = posFind - lineStart  # zero based.
                        scriptutils.JumpToDocument(
                            alldocs[lookpos].GetPathName(),
                            lineNo + 1,
                            colNo + 1,
                            nChars,
                        )
                        rc = FOUND_NEXT_FILE
                        break
        except win32ui.error:
            pass
    if rc == FOUND_NOTHING:
        # Loop around this control - attempt to find from the start of the control.
        posFind, foundSel = control.FindText(
            flags, (0, sel[0] - 1), searchParams.findText
        )
        if posFind >= 0:
            control.SCIEnsureVisible(control.LineFromChar(foundSel[0]))
            control.SetSel(foundSel)
            control.SetFocus()
            win32ui.SetStatusText("Not found! Searching from the top of the file.")
            rc = FOUND_LOOPED_BACK
        else:
            lastSearch.sel = -1, -1
            win32ui.SetStatusText("Can not find '%s'" % searchParams.findText)

    if rc != FOUND_NOTHING:
        lastSearch.sel = foundSel

    if lastSearch.remember:
        defaultSearch = lastSearch

        # track search history
        try:
            ix = searchHistory.index(searchParams.findText)
        except ValueError:
            if len(searchHistory) > 50:
                searchHistory[50:] = []
        else:
            del searchHistory[ix]
        searchHistory.insert(0, searchParams.findText)

    return rc


def _ReplaceIt(control):
    control = _GetControl(control)
    statusText = "Can not find '%s'." % lastSearch.findText
    rc = FOUND_NOTHING
    if control is not None and lastSearch.sel != (-1, -1):
        control.ReplaceSel(lastSearch.replaceText)
        rc = FindNext()
        if rc != FOUND_NOTHING:
            statusText = win32ui.LoadString(afxres.AFX_IDS_IDLEMESSAGE)
    win32ui.SetStatusText(statusText)
    return rc


class FindReplaceDialog(dialog.Dialog):
    def __init__(self):
        dialog.Dialog.__init__(self, self._GetDialogTemplate())
        self.HookCommand(self.OnFindNext, 109)

    def OnInitDialog(self):
        self.editFindText = self.GetDlgItem(102)
        self.butMatchWords = self.GetDlgItem(105)
        self.butMatchCase = self.GetDlgItem(107)
        self.butKeepDialogOpen = self.GetDlgItem(115)
        self.butAcrossFiles = self.GetDlgItem(116)
        self.butRemember = self.GetDlgItem(117)

        self.editFindText.SetWindowText(defaultSearch.findText)
        control = _GetControl()
        # _GetControl only gets normal MDI windows; if the interactive
        # window is docked and no document open, we get None.
        if control:
            # If we have a selection, default to that.
            sel = control.GetSelText()
            if len(sel) != 0:
                self.editFindText.SetWindowText(sel)
                if defaultSearch.remember:
                    defaultSearch.findText = sel
        for hist in searchHistory:
            self.editFindText.AddString(hist)

        if hasattr(self.editFindText, "SetEditSel"):
            self.editFindText.SetEditSel(0, -1)
        else:
            self.editFindText.SetSel(0, -1)
        self.butMatchWords.SetCheck(defaultSearch.matchWords)
        self.butMatchCase.SetCheck(defaultSearch.matchCase)
        self.butKeepDialogOpen.SetCheck(defaultSearch.keepDialogOpen)
        self.butAcrossFiles.SetCheck(defaultSearch.acrossFiles)
        self.butRemember.SetCheck(defaultSearch.remember)
        return dialog.Dialog.OnInitDialog(self)

    def OnDestroy(self, msg):
        global curDialog
        curDialog = None
        return dialog.Dialog.OnDestroy(self, msg)

    def DoFindNext(self):
        params = SearchParams()
        params.findText = self.editFindText.GetWindowText()
        params.matchCase = self.butMatchCase.GetCheck()
        params.matchWords = self.butMatchWords.GetCheck()
        params.acrossFiles = self.butAcrossFiles.GetCheck()
        params.remember = self.butRemember.GetCheck()
        return _FindIt(None, params)

    def OnFindNext(self, id, code):
        if code != 0:  # BN_CLICKED
            # 3d controls (python.exe + start_pythonwin.pyw) send
            # other notification codes
            return 1  #
        if not self.editFindText.GetWindowText():
            win32api.MessageBeep()
            return 1
        if self.DoFindNext() != FOUND_NOTHING:
            if not self.butKeepDialogOpen.GetCheck():
                self.DestroyWindow()


class FindDialog(FindReplaceDialog):
    def _GetDialogTemplate(self):
        style = (
            win32con.DS_MODALFRAME
            | win32con.WS_POPUP
            | win32con.WS_VISIBLE
            | win32con.WS_CAPTION
            | win32con.WS_SYSMENU
            | win32con.DS_SETFONT
        )
        visible = win32con.WS_CHILD | win32con.WS_VISIBLE
        dt = [
            ["Find", (0, 2, 240, 75), style, None, (8, "MS Sans Serif")],
            ["Static", "Fi&nd What:", 101, (5, 8, 40, 10), visible],
            [
                "ComboBox",
                "",
                102,
                (50, 7, 120, 120),
                visible
                | win32con.WS_BORDER
                | win32con.WS_TABSTOP
                | win32con.WS_VSCROLL
                | win32con.CBS_DROPDOWN
                | win32con.CBS_AUTOHSCROLL,
            ],
            [
                "Button",
                "Match &whole word only",
                105,
                (5, 23, 100, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "Match &case",
                107,
                (5, 33, 100, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "Keep &dialog open",
                115,
                (5, 43, 100, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "Across &open files",
                116,
                (5, 52, 100, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "&Remember as default search",
                117,
                (5, 61, 150, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "&Find Next",
                109,
                (185, 5, 50, 14),
                visible | win32con.BS_DEFPUSHBUTTON | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "Cancel",
                win32con.IDCANCEL,
                (185, 23, 50, 14),
                visible | win32con.WS_TABSTOP,
            ],
        ]
        return dt


class ReplaceDialog(FindReplaceDialog):
    def _GetDialogTemplate(self):
        style = (
            win32con.DS_MODALFRAME
            | win32con.WS_POPUP
            | win32con.WS_VISIBLE
            | win32con.WS_CAPTION
            | win32con.WS_SYSMENU
            | win32con.DS_SETFONT
        )
        visible = win32con.WS_CHILD | win32con.WS_VISIBLE
        dt = [
            ["Replace", (0, 2, 240, 95), style, 0, (8, "MS Sans Serif")],
            ["Static", "Fi&nd What:", 101, (5, 8, 40, 10), visible],
            [
                "ComboBox",
                "",
                102,
                (60, 7, 110, 120),
                visible
                | win32con.WS_BORDER
                | win32con.WS_TABSTOP
                | win32con.WS_VSCROLL
                | win32con.CBS_DROPDOWN
                | win32con.CBS_AUTOHSCROLL,
            ],
            ["Static", "Re&place with:", 103, (5, 25, 50, 10), visible],
            [
                "ComboBox",
                "",
                104,
                (60, 24, 110, 120),
                visible
                | win32con.WS_BORDER
                | win32con.WS_TABSTOP
                | win32con.WS_VSCROLL
                | win32con.CBS_DROPDOWN
                | win32con.CBS_AUTOHSCROLL,
            ],
            [
                "Button",
                "Match &whole word only",
                105,
                (5, 42, 100, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "Match &case",
                107,
                (5, 52, 100, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "Keep &dialog open",
                115,
                (5, 62, 100, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "Across &open files",
                116,
                (5, 72, 100, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "&Remember as default search",
                117,
                (5, 81, 150, 10),
                visible | win32con.BS_AUTOCHECKBOX | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "&Find Next",
                109,
                (185, 5, 50, 14),
                visible | win32con.BS_DEFPUSHBUTTON | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "&Replace",
                110,
                (185, 23, 50, 14),
                visible | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "Replace &All",
                111,
                (185, 41, 50, 14),
                visible | win32con.WS_TABSTOP,
            ],
            [
                "Button",
                "Cancel",
                win32con.IDCANCEL,
                (185, 59, 50, 14),
                visible | win32con.WS_TABSTOP,
            ],
        ]
        return dt

    def OnInitDialog(self):
        rc = FindReplaceDialog.OnInitDialog(self)
        self.HookCommand(self.OnReplace, 110)
        self.HookCommand(self.OnReplaceAll, 111)
        self.HookMessage(self.OnActivate, win32con.WM_ACTIVATE)
        self.editReplaceText = self.GetDlgItem(104)
        self.editReplaceText.SetWindowText(lastSearch.replaceText)
        if hasattr(self.editReplaceText, "SetEditSel"):
            self.editReplaceText.SetEditSel(0, -1)
        else:
            self.editReplaceText.SetSel(0, -1)
        self.butReplace = self.GetDlgItem(110)
        self.butReplaceAll = self.GetDlgItem(111)
        self.CheckButtonStates()
        return rc  # 0 when focus set

    def CheckButtonStates(self):
        # We can do a "Replace" or "Replace All" if the current selection
        # is the same as the search text.
        ft = self.editFindText.GetWindowText()
        control = _GetControl()
        # 		bCanReplace = len(ft)>0 and control.GetSelText() == ft
        bCanReplace = control is not None and lastSearch.sel == control.GetSel()
        self.butReplace.EnableWindow(bCanReplace)

    # 		self.butReplaceAll.EnableWindow(bCanReplace)

    def OnActivate(self, msg):
        wparam = msg[2]
        fActive = win32api.LOWORD(wparam)
        if fActive != win32con.WA_INACTIVE:
            self.CheckButtonStates()

    def OnFindNext(self, id, code):
        if code != 0:
            return 1
        self.DoFindNext()
        self.CheckButtonStates()

    def OnReplace(self, id, code):
        if code != 0:
            return 1
        lastSearch.replaceText = self.editReplaceText.GetWindowText()
        _ReplaceIt(None)

    def OnReplaceAll(self, id, code):
        if code != 0:
            return 1
        control = _GetControl(None)
        if control is not None:
            control.SetSel(0)
            num = 0
            if self.DoFindNext() == FOUND_NORMAL:
                num = 1
                lastSearch.replaceText = self.editReplaceText.GetWindowText()
                while _ReplaceIt(control) == FOUND_NORMAL:
                    num = num + 1

            win32ui.SetStatusText("Replaced %d occurrences" % num)
            if num > 0 and not self.butKeepDialogOpen.GetCheck():
                self.DestroyWindow()


if __name__ == "__main__":
    ShowFindDialog()
