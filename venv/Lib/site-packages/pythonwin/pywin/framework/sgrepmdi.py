# SGrepMDI is by Gordon McMillan (gmcm@hypernet.com)
# It does basically what Find In Files does in MSVC with a couple enhancements.
# - It saves any directories in the app's ini file (if you want to get rid
# 	of them you'll have to edit the file)
# - "Directories" can be directories,
#  -	semicolon separated lists of "directories",
#  -	environment variables that evaluate to "directories",
#  -	registry path names that evaluate to "directories",
#  -	all of which is recursive, so you can mix them all up.
# - It is MDI, so you can 'nest' greps and return to earlier ones,
# 	(ie, have multiple results open at the same time)
# - Like FIF, double clicking a line opens an editor and takes you to the line.
# - You can highlight text, right click and start a new grep with the selected
# 	text as search pattern and same directories etc as before.
# - You can save grep parameters (so you don't lose your hardearned pattern)
# 	from File|Save
# - You can save grep results by right clicking in the result window.
# Hats off to Mark Hammond for providing an environment where I could cobble
# something like this together in a couple evenings!

import glob
import os
import re

import win32api
import win32con
import win32ui
from pywin.mfc import dialog, docview, window

from . import scriptutils


def getsubdirs(d):
    dlist = []
    flist = glob.glob(d + "\\*")
    for f in flist:
        if os.path.isdir(f):
            dlist.append(f)
            dlist = dlist + getsubdirs(f)
    return dlist


class dirpath:
    def __init__(self, str, recurse=0):
        dp = str.split(";")
        dirs = {}
        for d in dp:
            if os.path.isdir(d):
                d = d.lower()
                if d not in dirs:
                    dirs[d] = None
                    if recurse:
                        subdirs = getsubdirs(d)
                        for sd in subdirs:
                            sd = sd.lower()
                            if sd not in dirs:
                                dirs[sd] = None
            elif os.path.isfile(d):
                pass
            else:
                x = None
                if d in os.environ:
                    x = dirpath(os.environ[d])
                elif d[:5] == "HKEY_":
                    keystr = d.split("\\")
                    try:
                        root = eval("win32con." + keystr[0])
                    except:
                        win32ui.MessageBox(
                            "Can't interpret registry key name '%s'" % keystr[0]
                        )
                    try:
                        subkey = "\\".join(keystr[1:])
                        val = win32api.RegQueryValue(root, subkey)
                        if val:
                            x = dirpath(val)
                        else:
                            win32ui.MessageBox(
                                "Registry path '%s' did not return a path entry" % d
                            )
                    except:
                        win32ui.MessageBox(
                            "Can't interpret registry key value: %s" % keystr[1:]
                        )
                else:
                    win32ui.MessageBox("Directory '%s' not found" % d)
                if x:
                    for xd in x:
                        if xd not in dirs:
                            dirs[xd] = None
                            if recurse:
                                subdirs = getsubdirs(xd)
                                for sd in subdirs:
                                    sd = sd.lower()
                                    if sd not in dirs:
                                        dirs[sd] = None
        self.dirs = []
        for d in list(dirs.keys()):
            self.dirs.append(d)

    def __getitem__(self, key):
        return self.dirs[key]

    def __len__(self):
        return len(self.dirs)

    def __setitem__(self, key, value):
        self.dirs[key] = value

    def __delitem__(self, key):
        del self.dirs[key]

    def __getslice__(self, lo, hi):
        return self.dirs[lo:hi]

    def __setslice__(self, lo, hi, seq):
        self.dirs[lo:hi] = seq

    def __delslice__(self, lo, hi):
        del self.dirs[lo:hi]

    def __add__(self, other):
        if type(other) == type(self) or type(other) == type([]):
            return self.dirs + other.dirs

    def __radd__(self, other):
        if type(other) == type(self) or type(other) == type([]):
            return other.dirs + self.dirs


# Group(1) is the filename, group(2) is the lineno.
# regexGrepResult=regex.compile("^\\([a-zA-Z]:.*\\)(\\([0-9]+\\))")

regexGrep = re.compile(r"^([a-zA-Z]:[^(]*)\(([0-9]+)\)")

# these are the atom numbers defined by Windows for basic dialog controls

BUTTON = 0x80
EDIT = 0x81
STATIC = 0x82
LISTBOX = 0x83
SCROLLBAR = 0x84
COMBOBOX = 0x85


class GrepTemplate(docview.RichEditDocTemplate):
    def __init__(self):
        docview.RichEditDocTemplate.__init__(
            self, win32ui.IDR_TEXTTYPE, GrepDocument, GrepFrame, GrepView
        )
        self.SetDocStrings("\nGrep\nGrep\nGrep params (*.grep)\n.grep\n\n\n")
        win32ui.GetApp().AddDocTemplate(self)
        self.docparams = None

    def MatchDocType(self, fileName, fileType):
        doc = self.FindOpenDocument(fileName)
        if doc:
            return doc
        ext = os.path.splitext(fileName)[1].lower()
        if ext == ".grep":
            return win32ui.CDocTemplate_Confidence_yesAttemptNative
        return win32ui.CDocTemplate_Confidence_noAttempt

    def setParams(self, params):
        self.docparams = params

    def readParams(self):
        tmp = self.docparams
        self.docparams = None
        return tmp


class GrepFrame(window.MDIChildWnd):
    # The template and doc params will one day be removed.
    def __init__(self, wnd=None):
        window.MDIChildWnd.__init__(self, wnd)


class GrepDocument(docview.RichEditDoc):
    def __init__(self, template):
        docview.RichEditDoc.__init__(self, template)
        self.dirpattern = ""
        self.filpattern = ""
        self.greppattern = ""
        self.casesensitive = 1
        self.recurse = 1
        self.verbose = 0

    def OnOpenDocument(self, fnm):
        # this bizarre stuff with params is so right clicking in a result window
        # and starting a new grep can communicate the default parameters to the
        # new grep.
        try:
            params = open(fnm, "r").read()
        except:
            params = None
        self.setInitParams(params)
        return self.OnNewDocument()

    def OnCloseDocument(self):
        try:
            win32ui.GetApp().DeleteIdleHandler(self.SearchFile)
        except:
            pass
        return self._obj_.OnCloseDocument()

    def saveInitParams(self):
        # Only save the flags, not the text boxes.
        paramstr = "\t%s\t\t%d\t%d" % (
            self.filpattern,
            self.casesensitive,
            self.recurse,
        )
        win32ui.WriteProfileVal("Grep", "Params", paramstr)

    def setInitParams(self, paramstr):
        if paramstr is None:
            paramstr = win32ui.GetProfileVal("Grep", "Params", "\t\t\t1\t0\t0")
        params = paramstr.split("\t")
        if len(params) < 3:
            params = params + [""] * (3 - len(params))
        if len(params) < 6:
            params = params + [0] * (6 - len(params))
        self.dirpattern = params[0]
        self.filpattern = params[1]
        self.greppattern = params[2]
        self.casesensitive = int(params[3])
        self.recurse = int(params[4])
        self.verbose = int(params[5])
        # setup some reasonable defaults.
        if not self.dirpattern:
            try:
                editor = win32ui.GetMainFrame().MDIGetActive()[0].GetEditorView()
                self.dirpattern = os.path.abspath(
                    os.path.dirname(editor.GetDocument().GetPathName())
                )
            except (AttributeError, win32ui.error):
                self.dirpattern = os.getcwd()
        if not self.filpattern:
            self.filpattern = "*.py"

    def OnNewDocument(self):
        if self.dirpattern == "":
            self.setInitParams(greptemplate.readParams())
        d = GrepDialog(
            self.dirpattern,
            self.filpattern,
            self.greppattern,
            self.casesensitive,
            self.recurse,
            self.verbose,
        )
        if d.DoModal() == win32con.IDOK:
            self.dirpattern = d["dirpattern"]
            self.filpattern = d["filpattern"]
            self.greppattern = d["greppattern"]
            self.casesensitive = d["casesensitive"]
            self.recurse = d["recursive"]
            self.verbose = d["verbose"]
            self.doSearch()
            self.saveInitParams()
            return 1
        return 0  # cancelled - return zero to stop frame creation.

    def doSearch(self):
        self.dp = dirpath(self.dirpattern, self.recurse)
        self.SetTitle("Grep for %s in %s" % (self.greppattern, self.filpattern))
        # self.text = []
        self.GetFirstView().Append("#Search " + self.dirpattern + "\n")
        if self.verbose:
            self.GetFirstView().Append("#   =" + repr(self.dp.dirs) + "\n")
        self.GetFirstView().Append("# Files " + self.filpattern + "\n")
        self.GetFirstView().Append("#   For " + self.greppattern + "\n")
        self.fplist = self.filpattern.split(";")
        if self.casesensitive:
            self.pat = re.compile(self.greppattern)
        else:
            self.pat = re.compile(self.greppattern, re.IGNORECASE)
        win32ui.SetStatusText("Searching.  Please wait...", 0)
        self.dpndx = self.fpndx = 0
        self.fndx = -1
        if not self.dp:
            self.GetFirstView().Append(
                "# ERROR: '%s' does not resolve to any search locations"
                % self.dirpattern
            )
            self.SetModifiedFlag(0)
        else:
            self.flist = glob.glob(self.dp[0] + "\\" + self.fplist[0])
            win32ui.GetApp().AddIdleHandler(self.SearchFile)

    def SearchFile(self, handler, count):
        self.fndx = self.fndx + 1
        if self.fndx < len(self.flist):
            f = self.flist[self.fndx]
            if self.verbose:
                self.GetFirstView().Append("# .." + f + "\n")
            # Directories may match the file type pattern, and files may be removed
            #  while grep is running
            if os.path.isfile(f):
                win32ui.SetStatusText("Searching " + f, 0)
                lines = open(f, "r").readlines()
                for i in range(len(lines)):
                    line = lines[i]
                    if self.pat.search(line) != None:
                        self.GetFirstView().Append(f + "(" + repr(i + 1) + ") " + line)
        else:
            self.fndx = -1
            self.fpndx = self.fpndx + 1
            if self.fpndx < len(self.fplist):
                self.flist = glob.glob(
                    self.dp[self.dpndx] + "\\" + self.fplist[self.fpndx]
                )
            else:
                self.fpndx = 0
                self.dpndx = self.dpndx + 1
                if self.dpndx < len(self.dp):
                    self.flist = glob.glob(
                        self.dp[self.dpndx] + "\\" + self.fplist[self.fpndx]
                    )
                else:
                    win32ui.SetStatusText("Search complete.", 0)
                    self.SetModifiedFlag(0)  # default to not modified.
                    try:
                        win32ui.GetApp().DeleteIdleHandler(self.SearchFile)
                    except:
                        pass
                    return 0
        return 1

    def GetParams(self):
        return (
            self.dirpattern
            + "\t"
            + self.filpattern
            + "\t"
            + self.greppattern
            + "\t"
            + repr(self.casesensitive)
            + "\t"
            + repr(self.recurse)
            + "\t"
            + repr(self.verbose)
        )

    def OnSaveDocument(self, filename):
        # 		print 'OnSaveDocument() filename=',filename
        savefile = open(filename, "wb")
        txt = self.GetParams() + "\n"
        # 		print 'writing',txt
        savefile.write(txt)
        savefile.close()
        self.SetModifiedFlag(0)
        return 1


ID_OPEN_FILE = 0xE400
ID_GREP = 0xE401
ID_SAVERESULTS = 0x402
ID_TRYAGAIN = 0x403


class GrepView(docview.RichEditView):
    def __init__(self, doc):
        docview.RichEditView.__init__(self, doc)
        self.SetWordWrap(win32ui.CRichEditView_WrapNone)
        self.HookHandlers()

    def OnInitialUpdate(self):
        rc = self._obj_.OnInitialUpdate()
        format = (-402653169, 0, 200, 0, 0, 0, 49, "Courier New")
        self.SetDefaultCharFormat(format)
        return rc

    def HookHandlers(self):
        self.HookMessage(self.OnRClick, win32con.WM_RBUTTONDOWN)
        self.HookCommand(self.OnCmdOpenFile, ID_OPEN_FILE)
        self.HookCommand(self.OnCmdGrep, ID_GREP)
        self.HookCommand(self.OnCmdSave, ID_SAVERESULTS)
        self.HookCommand(self.OnTryAgain, ID_TRYAGAIN)
        self.HookMessage(self.OnLDblClick, win32con.WM_LBUTTONDBLCLK)

    def OnLDblClick(self, params):
        line = self.GetLine()
        regexGrepResult = regexGrep.match(line)
        if regexGrepResult:
            fname = regexGrepResult.group(1)
            line = int(regexGrepResult.group(2))
            scriptutils.JumpToDocument(fname, line)
            return 0  # dont pass on
        return 1  # pass it on by default.

    def OnRClick(self, params):
        menu = win32ui.CreatePopupMenu()
        flags = win32con.MF_STRING | win32con.MF_ENABLED
        lineno = self._obj_.LineFromChar(-1)  # selection or current line
        line = self._obj_.GetLine(lineno)
        regexGrepResult = regexGrep.match(line)
        if regexGrepResult:
            self.fnm = regexGrepResult.group(1)
            self.lnnum = int(regexGrepResult.group(2))
            menu.AppendMenu(flags, ID_OPEN_FILE, "&Open " + self.fnm)
            menu.AppendMenu(win32con.MF_SEPARATOR)
        menu.AppendMenu(flags, ID_TRYAGAIN, "&Try Again")
        charstart, charend = self._obj_.GetSel()
        if charstart != charend:
            linestart = self._obj_.LineIndex(lineno)
            self.sel = line[charstart - linestart : charend - linestart]
            menu.AppendMenu(flags, ID_GREP, "&Grep for " + self.sel)
            menu.AppendMenu(win32con.MF_SEPARATOR)
        menu.AppendMenu(flags, win32ui.ID_EDIT_CUT, "Cu&t")
        menu.AppendMenu(flags, win32ui.ID_EDIT_COPY, "&Copy")
        menu.AppendMenu(flags, win32ui.ID_EDIT_PASTE, "&Paste")
        menu.AppendMenu(flags, win32con.MF_SEPARATOR)
        menu.AppendMenu(flags, win32ui.ID_EDIT_SELECT_ALL, "&Select all")
        menu.AppendMenu(flags, win32con.MF_SEPARATOR)
        menu.AppendMenu(flags, ID_SAVERESULTS, "Sa&ve results")
        menu.TrackPopupMenu(params[5])
        return 0

    def OnCmdOpenFile(self, cmd, code):
        doc = win32ui.GetApp().OpenDocumentFile(self.fnm)
        if doc:
            vw = doc.GetFirstView()
            # hope you have an editor that implements GotoLine()!
            try:
                vw.GotoLine(int(self.lnnum))
            except:
                pass
        return 0

    def OnCmdGrep(self, cmd, code):
        if code != 0:
            return 1
        curparamsstr = self.GetDocument().GetParams()
        params = curparamsstr.split("\t")
        params[2] = self.sel
        greptemplate.setParams("\t".join(params))
        greptemplate.OpenDocumentFile()
        return 0

    def OnTryAgain(self, cmd, code):
        if code != 0:
            return 1
        greptemplate.setParams(self.GetDocument().GetParams())
        greptemplate.OpenDocumentFile()
        return 0

    def OnCmdSave(self, cmd, code):
        if code != 0:
            return 1
        flags = win32con.OFN_OVERWRITEPROMPT
        dlg = win32ui.CreateFileDialog(
            0, None, None, flags, "Text Files (*.txt)|*.txt||", self
        )
        dlg.SetOFNTitle("Save Results As")
        if dlg.DoModal() == win32con.IDOK:
            pn = dlg.GetPathName()
            self._obj_.SaveTextFile(pn)
        return 0

    def Append(self, strng):
        numlines = self.GetLineCount()
        endpos = self.LineIndex(numlines - 1) + len(self.GetLine(numlines - 1))
        self.SetSel(endpos, endpos)
        self.ReplaceSel(strng)


class GrepDialog(dialog.Dialog):
    def __init__(self, dp, fp, gp, cs, r, v):
        style = (
            win32con.DS_MODALFRAME
            | win32con.WS_POPUP
            | win32con.WS_VISIBLE
            | win32con.WS_CAPTION
            | win32con.WS_SYSMENU
            | win32con.DS_SETFONT
        )
        CS = win32con.WS_CHILD | win32con.WS_VISIBLE
        tmp = [
            ["Grep", (0, 0, 210, 90), style, None, (8, "MS Sans Serif")],
        ]
        tmp.append([STATIC, "Grep For:", -1, (7, 7, 50, 9), CS])
        tmp.append(
            [
                EDIT,
                gp,
                101,
                (52, 7, 144, 11),
                CS | win32con.WS_TABSTOP | win32con.ES_AUTOHSCROLL | win32con.WS_BORDER,
            ]
        )
        tmp.append([STATIC, "Directories:", -1, (7, 20, 50, 9), CS])
        tmp.append(
            [
                EDIT,
                dp,
                102,
                (52, 20, 128, 11),
                CS | win32con.WS_TABSTOP | win32con.ES_AUTOHSCROLL | win32con.WS_BORDER,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "...",
                110,
                (182, 20, 16, 11),
                CS | win32con.BS_PUSHBUTTON | win32con.WS_TABSTOP,
            ]
        )
        tmp.append([STATIC, "File types:", -1, (7, 33, 50, 9), CS])
        tmp.append(
            [
                EDIT,
                fp,
                103,
                (52, 33, 128, 11),
                CS | win32con.WS_TABSTOP | win32con.ES_AUTOHSCROLL | win32con.WS_BORDER,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "...",
                111,
                (182, 33, 16, 11),
                CS | win32con.BS_PUSHBUTTON | win32con.WS_TABSTOP,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "Case sensitive",
                104,
                (7, 45, 72, 9),
                CS
                | win32con.BS_AUTOCHECKBOX
                | win32con.BS_LEFTTEXT
                | win32con.WS_TABSTOP,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "Subdirectories",
                105,
                (7, 56, 72, 9),
                CS
                | win32con.BS_AUTOCHECKBOX
                | win32con.BS_LEFTTEXT
                | win32con.WS_TABSTOP,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "Verbose",
                106,
                (7, 67, 72, 9),
                CS
                | win32con.BS_AUTOCHECKBOX
                | win32con.BS_LEFTTEXT
                | win32con.WS_TABSTOP,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "OK",
                win32con.IDOK,
                (166, 53, 32, 12),
                CS | win32con.BS_DEFPUSHBUTTON | win32con.WS_TABSTOP,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "Cancel",
                win32con.IDCANCEL,
                (166, 67, 32, 12),
                CS | win32con.BS_PUSHBUTTON | win32con.WS_TABSTOP,
            ]
        )
        dialog.Dialog.__init__(self, tmp)
        self.AddDDX(101, "greppattern")
        self.AddDDX(102, "dirpattern")
        self.AddDDX(103, "filpattern")
        self.AddDDX(104, "casesensitive")
        self.AddDDX(105, "recursive")
        self.AddDDX(106, "verbose")
        self._obj_.data["greppattern"] = gp
        self._obj_.data["dirpattern"] = dp
        self._obj_.data["filpattern"] = fp
        self._obj_.data["casesensitive"] = cs
        self._obj_.data["recursive"] = r
        self._obj_.data["verbose"] = v
        self.HookCommand(self.OnMoreDirectories, 110)
        self.HookCommand(self.OnMoreFiles, 111)

    def OnMoreDirectories(self, cmd, code):
        if code != 0:
            return 1
        self.getMore("Grep\\Directories", "dirpattern")

    def OnMoreFiles(self, cmd, code):
        if code != 0:
            return 1
        self.getMore("Grep\\File Types", "filpattern")

    def getMore(self, section, key):
        self.UpdateData(1)
        # get the items out of the ini file
        ini = win32ui.GetProfileFileName()
        secitems = win32api.GetProfileSection(section, ini)
        items = []
        for secitem in secitems:
            items.append(secitem.split("=")[1])
        dlg = GrepParamsDialog(items)
        if dlg.DoModal() == win32con.IDOK:
            itemstr = ";".join(dlg.getItems())
            self._obj_.data[key] = itemstr
            # update the ini file with dlg.getNew()
            i = 0
            newitems = dlg.getNew()
            if newitems:
                items = items + newitems
                for item in items:
                    win32api.WriteProfileVal(section, repr(i), item, ini)
                    i = i + 1
            self.UpdateData(0)

    def OnOK(self):
        self.UpdateData(1)
        for id, name in (
            (101, "greppattern"),
            (102, "dirpattern"),
            (103, "filpattern"),
        ):
            if not self[name]:
                self.GetDlgItem(id).SetFocus()
                win32api.MessageBeep()
                win32ui.SetStatusText("Please enter a value")
                return
        self._obj_.OnOK()


class GrepParamsDialog(dialog.Dialog):
    def __init__(self, items):
        self.items = items
        self.newitems = []
        style = (
            win32con.DS_MODALFRAME
            | win32con.WS_POPUP
            | win32con.WS_VISIBLE
            | win32con.WS_CAPTION
            | win32con.WS_SYSMENU
            | win32con.DS_SETFONT
        )
        CS = win32con.WS_CHILD | win32con.WS_VISIBLE
        tmp = [
            ["Grep Parameters", (0, 0, 205, 100), style, None, (8, "MS Sans Serif")],
        ]
        tmp.append(
            [
                LISTBOX,
                "",
                107,
                (7, 7, 150, 72),
                CS
                | win32con.LBS_MULTIPLESEL
                | win32con.LBS_STANDARD
                | win32con.LBS_HASSTRINGS
                | win32con.WS_TABSTOP
                | win32con.LBS_NOTIFY,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "OK",
                win32con.IDOK,
                (167, 7, 32, 12),
                CS | win32con.BS_DEFPUSHBUTTON | win32con.WS_TABSTOP,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "Cancel",
                win32con.IDCANCEL,
                (167, 23, 32, 12),
                CS | win32con.BS_PUSHBUTTON | win32con.WS_TABSTOP,
            ]
        )
        tmp.append([STATIC, "New:", -1, (2, 83, 15, 12), CS])
        tmp.append(
            [
                EDIT,
                "",
                108,
                (18, 83, 139, 12),
                CS | win32con.WS_TABSTOP | win32con.ES_AUTOHSCROLL | win32con.WS_BORDER,
            ]
        )
        tmp.append(
            [
                BUTTON,
                "Add",
                109,
                (167, 83, 32, 12),
                CS | win32con.BS_PUSHBUTTON | win32con.WS_TABSTOP,
            ]
        )
        dialog.Dialog.__init__(self, tmp)
        self.HookCommand(self.OnAddItem, 109)
        self.HookCommand(self.OnListDoubleClick, 107)

    def OnInitDialog(self):
        lb = self.GetDlgItem(107)
        for item in self.items:
            lb.AddString(item)
        return self._obj_.OnInitDialog()

    def OnAddItem(self, cmd, code):
        if code != 0:
            return 1
        eb = self.GetDlgItem(108)
        item = eb.GetLine(0)
        self.newitems.append(item)
        lb = self.GetDlgItem(107)
        i = lb.AddString(item)
        lb.SetSel(i, 1)
        return 1

    def OnListDoubleClick(self, cmd, code):
        if code == win32con.LBN_DBLCLK:
            self.OnOK()
            return 1

    def OnOK(self):
        lb = self.GetDlgItem(107)
        self.selections = lb.GetSelTextItems()
        self._obj_.OnOK()

    def getItems(self):
        return self.selections

    def getNew(self):
        return self.newitems


try:
    win32ui.GetApp().RemoveDocTemplate(greptemplate)
except NameError:
    pass

greptemplate = GrepTemplate()
