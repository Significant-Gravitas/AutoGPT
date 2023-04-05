######################################################################
##
## The Pychecker MDI Plug-In UserModule for Pythonwin
##
## contributed by Robert Kiendl
##
## Style is similar to (and inherited) from the SGrepMDI UserModule
##
## Usage:
##
## Start Pychecker on current file: Menu/File/New../Pychecker.
## Use it: Jump to Pychecker warning source lines by double-click.
## Auto-add "#$pycheck_no" / "#$pycheck_no=specific-re-pattern" tags
## to source lines by context/right-mouse-click on warning lines.
##
## It requires pychecker installed and the pychecker.bat to be on
## the PATH. Example pychecker.bat:
##
##   REM pychecker.bat
##   C:\bin\python.exe C:\PYTHON23\Lib\site-packages\pychecker\checker.py %1 %2 %3 %4 %5 %6 %7 %8 %9
##
## Adding it as default module in PythonWin:
##
##  +++ ./intpyapp.py	2006-10-02 17:59:32.974161600 +0200
##  @@ -272,7 +282,7 @@
## 	def LoadUserModules(self, moduleNames = None):
## 		# Load the users modules.
## 		if moduleNames is None:
##  -			default = "sgrepmdi"
##  +			default = "sgrepmdi,mdi_pychecker"
## 			moduleNames=win32ui.GetProfileVal('Python','Startup Modules',default)
## 		self.DoLoadModules(moduleNames)
##
######################################################################

import glob
import os
import re
import sys
import time

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
        for d in dirs.keys():
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
# regexGrep=re.compile(r"^([a-zA-Z]:[^(]*)\((\d+)\)")
regexGrep = re.compile(r"^(..[^\(:]+)?[\(:](\d+)[\):]:?\s*(.*)")

# these are the atom numbers defined by Windows for basic dialog controls

BUTTON = 0x80
EDIT = 0x81
STATIC = 0x82
LISTBOX = 0x83
SCROLLBAR = 0x84
COMBOBOX = 0x85


class TheTemplate(docview.RichEditDocTemplate):
    def __init__(self):
        docview.RichEditDocTemplate.__init__(
            self, win32ui.IDR_TEXTTYPE, TheDocument, TheFrame, TheView
        )
        self.SetDocStrings(
            "\nPychecker\nPychecker\nPychecker params (*.pychecker)\n.pychecker\n\n\n"
        )
        win32ui.GetApp().AddDocTemplate(self)
        self.docparams = None

    def MatchDocType(self, fileName, fileType):
        doc = self.FindOpenDocument(fileName)
        if doc:
            return doc
        ext = os.path.splitext(fileName)[1].lower()
        if ext == ".pychecker":
            return win32ui.CDocTemplate_Confidence_yesAttemptNative
        return win32ui.CDocTemplate_Confidence_noAttempt

    def setParams(self, params):
        self.docparams = params

    def readParams(self):
        tmp = self.docparams
        self.docparams = None
        return tmp


class TheFrame(window.MDIChildWnd):
    # The template and doc params will one day be removed.
    def __init__(self, wnd=None):
        window.MDIChildWnd.__init__(self, wnd)


class TheDocument(docview.RichEditDoc):
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
            win32ui.GetApp().DeleteIdleHandler(self.idleHandler)
        except:
            pass
        return self._obj_.OnCloseDocument()

    def saveInitParams(self):
        # Only save the flags, not the text boxes.
        paramstr = "\t\t\t%d\t%d" % (self.casesensitive, self.recurse)
        win32ui.WriteProfileVal("Pychecker", "Params", paramstr)

    def setInitParams(self, paramstr):
        if paramstr is None:
            paramstr = win32ui.GetProfileVal("Pychecker", "Params", "\t\t\t1\t0\t0")
        params = paramstr.split("\t")
        if len(params) < 3:
            params = params + [""] * (3 - len(params))
        if len(params) < 6:
            params = params + [0] * (6 - len(params))
        self.dirpattern = params[0]
        self.filpattern = params[1]
        self.greppattern = params[2] or "-#1000 --only"
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
            try:
                editor = win32ui.GetMainFrame().MDIGetActive()[0].GetEditorView()
                self.filpattern = editor.GetDocument().GetPathName()
            except AttributeError:
                self.filpattern = "*.py"

    def OnNewDocument(self):
        if self.dirpattern == "":
            self.setInitParams(greptemplate.readParams())
        d = TheDialog(
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
            # self.casesensitive = d['casesensitive']
            # self.recurse = d['recursive']
            # self.verbose = d['verbose']
            self.doSearch()
            self.saveInitParams()
            return 1
        return 0  # cancelled - return zero to stop frame creation.

    def doSearch(self):
        self.dp = dirpath(self.dirpattern, self.recurse)
        self.SetTitle(
            "Pychecker Run '%s' (options: %s)" % (self.filpattern, self.greppattern)
        )
        # self.text = []
        self.GetFirstView().Append(
            "#Pychecker Run in " + self.dirpattern + "  %s\n" % time.asctime()
        )
        if self.verbose:
            self.GetFirstView().Append("#   =" + repr(self.dp.dirs) + "\n")
        self.GetFirstView().Append("# Files   " + self.filpattern + "\n")
        self.GetFirstView().Append("# Options " + self.greppattern + "\n")
        self.fplist = self.filpattern.split(";")
        self.GetFirstView().Append(
            "# Running...  ( double click on result lines in order to jump to the source code ) \n"
        )
        win32ui.SetStatusText("Pychecker running.  Please wait...", 0)
        self.dpndx = self.fpndx = 0
        self.fndx = -1
        if not self.dp:
            self.GetFirstView().Append(
                "# ERROR: '%s' does not resolve to any search locations"
                % self.dirpattern
            )
            self.SetModifiedFlag(0)
        else:
            ##self.flist = glob.glob(self.dp[0]+'\\'+self.fplist[0])
            import operator

            self.flist = reduce(operator.add, list(map(glob.glob, self.fplist)))
            # import pywin.debugger;pywin.debugger.set_trace()
            self.startPycheckerRun()

    def idleHandler(self, handler, count):
        import time

        time.sleep(0.001)
        if self.result != None:
            win32ui.GetApp().DeleteIdleHandler(self.idleHandler)
            return 0
        return 1  # more

    def startPycheckerRun(self):
        self.result = None
        old = win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_APPSTARTING))
        win32ui.GetApp().AddIdleHandler(self.idleHandler)
        import _thread

        _thread.start_new(self.threadPycheckerRun, ())
        ##win32api.SetCursor(old)

    def threadPycheckerRun(self):
        result = ""
        rc = -1
        try:
            options = self.greppattern
            files = " ".join(self.flist)
            # Recently MarkH has failed to run pychecker without it having
            # been explicitly installed - so we assume it is and locate it
            # from its default location.
            # Step1 - get python.exe
            py = os.path.join(sys.prefix, "python.exe")
            if not os.path.isfile(py):
                if "64 bit" in sys.version:
                    py = os.path.join(sys.prefix, "PCBuild", "amd64", "python.exe")
                else:
                    py = os.path.join(sys.prefix, "PCBuild", "python.exe")
            try:
                py = win32api.GetShortPathName(py)
            except win32api.error:
                py = ""
            # Find checker.py
            import sysconfig

            pychecker = os.path.join(
                sysconfig.get_paths()["purelib"], "pychecker", "checker.py"
            )
            if not os.path.isfile(py):
                result = "Can't find python.exe!\n"
            elif not os.path.isfile(pychecker):
                result = (
                    "Can't find checker.py - please install pychecker "
                    "(or run 'setup.py install' if you have the source version)\n"
                )
            else:
                cmd = '%s "%s" %s %s 2>&1' % (py, pychecker, options, files)
                ##fin,fout,ferr=os.popen3(cmd)
                ##result=ferr.read()+fout.read()
                result = os.popen(cmd).read()
                ##rc=f.close()
            self.GetFirstView().Append(result)
        finally:
            self.result = result
            print("== Pychecker run finished ==")
            self.GetFirstView().Append("\n" + "== Pychecker run finished ==")
            self.SetModifiedFlag(0)

    def _inactive_idleHandler(self, handler, count):
        self.fndx = self.fndx + 1
        if self.fndx < len(self.flist):
            f = self.flist[self.fndx]
            if self.verbose:
                self.GetFirstView().Append("# .." + f + "\n")
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
                        win32ui.GetApp().DeleteIdleHandler(self.idleHandler)
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
        #       print 'OnSaveDocument() filename=',filename
        savefile = open(filename, "wb")
        txt = self.GetParams() + "\n"
        #       print 'writing',txt
        savefile.write(txt)
        savefile.close()
        self.SetModifiedFlag(0)
        return 1


ID_OPEN_FILE = 0xE500
ID_PYCHECKER = 0xE501
ID_SAVERESULTS = 0x502
ID_TRYAGAIN = 0x503
ID_ADDCOMMENT = 0x504
ID_ADDPYCHECKNO2 = 0x505


class TheView(docview.RichEditView):
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
        self.HookCommand(self.OnCmdThe, ID_PYCHECKER)
        self.HookCommand(self.OnCmdSave, ID_SAVERESULTS)
        self.HookCommand(self.OnTryAgain, ID_TRYAGAIN)
        self.HookCommand(self.OnAddComment, ID_ADDCOMMENT)
        self.HookCommand(self.OnAddComment, ID_ADDPYCHECKNO2)
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
        charstart, charend = self._obj_.GetSel()
        if regexGrepResult:
            self.fnm = regexGrepResult.group(1)
            self.lnnum = int(regexGrepResult.group(2))
            menu.AppendMenu(flags, ID_OPEN_FILE, "&Open " + self.fnm)
            menu.AppendMenu(
                flags, ID_ADDCOMMENT, "&Add to source: Comment Tag/#$pycheck_no .."
            )
            menu.AppendMenu(
                flags,
                ID_ADDPYCHECKNO2,
                "&Add to source: Specific #$pycheck_no=%(errtext)s ..",
            )
            menu.AppendMenu(win32con.MF_SEPARATOR)
        menu.AppendMenu(flags, ID_TRYAGAIN, "&Try Again")
        menu.AppendMenu(flags, win32ui.ID_EDIT_CUT, "Cu&t")
        menu.AppendMenu(flags, win32ui.ID_EDIT_COPY, "&Copy")
        menu.AppendMenu(flags, win32ui.ID_EDIT_PASTE, "&Paste")
        menu.AppendMenu(flags, win32con.MF_SEPARATOR)
        menu.AppendMenu(flags, win32ui.ID_EDIT_SELECT_ALL, "&Select all")
        menu.AppendMenu(flags, win32con.MF_SEPARATOR)
        menu.AppendMenu(flags, ID_SAVERESULTS, "Sa&ve results")
        menu.TrackPopupMenu(params[5])
        return 0

    def OnAddComment(self, cmd, code):
        addspecific = cmd == ID_ADDPYCHECKNO2
        _ = list(self.GetSel())
        _.sort()
        start, end = _
        line_start, line_end = self.LineFromChar(start), self.LineFromChar(end)
        first = 1
        for i in range(line_start, line_end + 1):
            line = self.GetLine(i)
            m = regexGrep.match(line)
            if m:
                if first:
                    first = 0
                    cmnt = dialog.GetSimpleInput(
                        "Add to %s lines" % (line_end - line_start + 1),
                        addspecific
                        and "  #$pycheck_no=%(errtext)s"
                        or "  #$pycheck_no",
                    )
                    if not cmnt:
                        return 0
                ##import pywin.debugger;pywin.debugger.set_trace()
                fname = m.group(1)
                line = int(m.group(2))
                view = scriptutils.JumpToDocument(fname, line)
                pos = view.LineIndex(line) - 1
                if view.GetTextRange(pos - 1, pos) in ("\r", "\n"):
                    pos -= 1
                view.SetSel(pos, pos)
                errtext = m.group(3)
                if start != end and line_start == line_end:
                    errtext = self.GetSelText()
                errtext = repr(re.escape(errtext).replace("\ ", " "))
                view.ReplaceSel(addspecific and cmnt % locals() or cmnt)
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

    def OnCmdThe(self, cmd, code):
        curparamsstr = self.GetDocument().GetParams()
        params = curparamsstr.split("\t")
        params[2] = self.sel
        greptemplate.setParams("\t".join(params))
        greptemplate.OpenDocumentFile()
        return 0

    def OnTryAgain(self, cmd, code):
        greptemplate.setParams(self.GetDocument().GetParams())
        greptemplate.OpenDocumentFile()
        return 0

    def OnCmdSave(self, cmd, code):
        flags = win32con.OFN_OVERWRITEPROMPT
        dlg = win32ui.CreateFileDialog(
            0, None, None, flags, "Text Files (*.txt)|*.txt||", self
        )
        dlg.SetOFNTitle("Save Results As")
        if dlg.DoModal() == win32con.IDOK:
            pn = dlg.GetPathName()
            self._obj_.SaveFile(pn)
        return 0

    def Append(self, strng):
        numlines = self.GetLineCount()
        endpos = self.LineIndex(numlines - 1) + len(self.GetLine(numlines - 1))
        self.SetSel(endpos, endpos)
        self.ReplaceSel(strng)


class TheDialog(dialog.Dialog):
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
            ["Pychecker Run", (0, 0, 210, 90), style, None, (8, "MS Sans Serif")],
        ]
        tmp.append([STATIC, "Files:", -1, (7, 7, 50, 9), CS])
        tmp.append(
            [
                EDIT,
                gp,
                103,
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
        tmp.append([STATIC, "Options:", -1, (7, 33, 50, 9), CS])
        tmp.append(
            [
                EDIT,
                fp,
                101,
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
        # tmp.append([BUTTON,'Case sensitive',       104, (7,  45,  72,  9), CS | win32con.BS_AUTOCHECKBOX | win32con.BS_LEFTTEXT| win32con.WS_TABSTOP])
        # tmp.append([BUTTON,'Subdirectories',       105, (7,  56,  72,  9), CS | win32con.BS_AUTOCHECKBOX | win32con.BS_LEFTTEXT| win32con.WS_TABSTOP])
        # tmp.append([BUTTON,'Verbose',              106, (7,  67,  72,  9), CS | win32con.BS_AUTOCHECKBOX | win32con.BS_LEFTTEXT| win32con.WS_TABSTOP])
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
        # self.AddDDX(104,'casesensitive')
        # self.AddDDX(105,'recursive')
        # self.AddDDX(106,'verbose')
        self._obj_.data["greppattern"] = gp
        self._obj_.data["dirpattern"] = dp
        self._obj_.data["filpattern"] = fp
        # self._obj_.data['casesensitive']  = cs
        # self._obj_.data['recursive'] = r
        # self._obj_.data['verbose']  = v
        self.HookCommand(self.OnMoreDirectories, 110)
        self.HookCommand(self.OnMoreFiles, 111)

    def OnMoreDirectories(self, cmd, code):
        self.getMore("Pychecker\\Directories", "dirpattern")

    def OnMoreFiles(self, cmd, code):
        self.getMore("Pychecker\\File Types", "filpattern")

    def getMore(self, section, key):
        self.UpdateData(1)
        # get the items out of the ini file
        ini = win32ui.GetProfileFileName()
        secitems = win32api.GetProfileSection(section, ini)
        items = []
        for secitem in secitems:
            items.append(secitem.split("=")[1])
        dlg = TheParamsDialog(items)
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


class TheParamsDialog(dialog.Dialog):
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
            [
                "Pychecker Parameters",
                (0, 0, 205, 100),
                style,
                None,
                (8, "MS Sans Serif"),
            ],
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

greptemplate = TheTemplate()
