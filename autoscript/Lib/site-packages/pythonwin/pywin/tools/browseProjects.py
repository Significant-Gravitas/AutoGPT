import glob
import os
import pyclbr

import afxres
import commctrl
import pywin.framework.scriptutils
import regutil
import win32api
import win32con
import win32ui
from pywin.mfc import dialog

from . import hierlist


class HLIErrorItem(hierlist.HierListItem):
    def __init__(self, text):
        self.text = text
        hierlist.HierListItem.__init__(self)

    def GetText(self):
        return self.text


class HLICLBRItem(hierlist.HierListItem):
    def __init__(self, name, file, lineno, suffix=""):
        # If the 'name' object itself has a .name, use it.  Not sure
        # how this happens, but seems pyclbr related.
        # See PyWin32 bug 817035
        self.name = getattr(name, "name", name)
        self.file = file
        self.lineno = lineno
        self.suffix = suffix

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name

    def GetText(self):
        return self.name + self.suffix

    def TakeDefaultAction(self):
        if self.file:
            pywin.framework.scriptutils.JumpToDocument(
                self.file, self.lineno, bScrollToTop=1
            )
        else:
            win32ui.SetStatusText("The source of this object is unknown")

    def PerformItemSelected(self):
        if self.file is None:
            msg = "%s - source can not be located." % (self.name,)
        else:
            msg = "%s defined at line %d of %s" % (self.name, self.lineno, self.file)
        win32ui.SetStatusText(msg)


class HLICLBRClass(HLICLBRItem):
    def __init__(self, clbrclass, suffix=""):
        try:
            name = clbrclass.name
            file = clbrclass.file
            lineno = clbrclass.lineno
            self.super = clbrclass.super
            self.methods = clbrclass.methods
        except AttributeError:
            name = clbrclass
            file = lineno = None
            self.super = []
            self.methods = {}
        HLICLBRItem.__init__(self, name, file, lineno, suffix)

    def GetSubList(self):
        ret = []
        for c in self.super:
            ret.append(HLICLBRClass(c, " (Parent class)"))
        for meth, lineno in self.methods.items():
            ret.append(HLICLBRMethod(meth, self.file, lineno, " (method)"))
        return ret

    def IsExpandable(self):
        return len(self.methods) + len(self.super)

    def GetBitmapColumn(self):
        return 21


class HLICLBRFunction(HLICLBRClass):
    def GetBitmapColumn(self):
        return 22


class HLICLBRMethod(HLICLBRItem):
    def GetBitmapColumn(self):
        return 22


class HLIModuleItem(hierlist.HierListItem):
    def __init__(self, path):
        hierlist.HierListItem.__init__(self)
        self.path = path

    def GetText(self):
        return os.path.split(self.path)[1] + " (module)"

    def IsExpandable(self):
        return 1

    def TakeDefaultAction(self):
        win32ui.GetApp().OpenDocumentFile(self.path)

    def GetBitmapColumn(self):
        col = 4  # Default
        try:
            if win32api.GetFileAttributes(self.path) & win32con.FILE_ATTRIBUTE_READONLY:
                col = 5
        except win32api.error:
            pass
        return col

    def GetSubList(self):
        mod, path = pywin.framework.scriptutils.GetPackageModuleName(self.path)
        win32ui.SetStatusText("Building class list - please wait...", 1)
        win32ui.DoWaitCursor(1)
        try:
            try:
                reader = pyclbr.readmodule_ex  # Post 1.5.2 interface.
                extra_msg = " or functions"
            except AttributeError:
                reader = pyclbr.readmodule
                extra_msg = ""
            data = reader(mod, [path])
            if data:
                ret = []
                for item in data.values():
                    if (
                        item.__class__ != pyclbr.Class
                    ):  # ie, it is a pyclbr Function instance (only introduced post 1.5.2)
                        ret.append(HLICLBRFunction(item, " (function)"))
                    else:
                        ret.append(HLICLBRClass(item, " (class)"))
                ret.sort()
                return ret
            else:
                return [HLIErrorItem("No Python classes%s in module." % (extra_msg,))]
        finally:
            win32ui.DoWaitCursor(0)
            win32ui.SetStatusText(win32ui.LoadString(afxres.AFX_IDS_IDLEMESSAGE))


def MakePathSubList(path):
    ret = []
    for filename in glob.glob(os.path.join(path, "*")):
        if os.path.isdir(filename) and os.path.isfile(
            os.path.join(filename, "__init__.py")
        ):
            ret.append(HLIDirectoryItem(filename, os.path.split(filename)[1]))
        else:
            if os.path.splitext(filename)[1].lower() in [".py", ".pyw"]:
                ret.append(HLIModuleItem(filename))
    return ret


class HLIDirectoryItem(hierlist.HierListItem):
    def __init__(self, path, displayName=None, bSubDirs=0):
        hierlist.HierListItem.__init__(self)
        self.path = path
        self.bSubDirs = bSubDirs
        if displayName:
            self.displayName = displayName
        else:
            self.displayName = path

    def IsExpandable(self):
        return 1

    def GetText(self):
        return self.displayName

    def GetSubList(self):
        ret = MakePathSubList(self.path)
        if (
            os.path.split(self.path)[1] == "win32com"
        ):  # Complete and utter hack for win32com.
            try:
                path = win32api.GetFullPathName(
                    os.path.join(self.path, "..\\win32comext")
                )
                ret = ret + MakePathSubList(path)
            except win32ui.error:
                pass
        return ret


class HLIProjectRoot(hierlist.HierListItem):
    def __init__(self, projectName, displayName=None):
        hierlist.HierListItem.__init__(self)
        self.projectName = projectName
        self.displayName = displayName or projectName

    def GetText(self):
        return self.displayName

    def IsExpandable(self):
        return 1

    def GetSubList(self):
        paths = regutil.GetRegisteredNamedPath(self.projectName)
        pathList = paths.split(";")
        if len(pathList) == 1:  # Single dir - dont bother putting the dir in
            ret = MakePathSubList(pathList[0])
        else:
            ret = list(map(HLIDirectoryItem, pathList))
        return ret


class HLIRoot(hierlist.HierListItem):
    def __init__(self):
        hierlist.HierListItem.__init__(self)

    def IsExpandable(self):
        return 1

    def GetSubList(self):
        keyStr = regutil.BuildDefaultPythonKey() + "\\PythonPath"
        hKey = win32api.RegOpenKey(regutil.GetRootKey(), keyStr)
        try:
            ret = []
            ret.append(HLIProjectRoot("", "Standard Python Library"))  # The core path.
            index = 0
            while 1:
                try:
                    ret.append(HLIProjectRoot(win32api.RegEnumKey(hKey, index)))
                    index = index + 1
                except win32api.error:
                    break
            return ret
        finally:
            win32api.RegCloseKey(hKey)


class dynamic_browser(dialog.Dialog):
    style = win32con.WS_OVERLAPPEDWINDOW | win32con.WS_VISIBLE
    cs = (
        win32con.WS_CHILD
        | win32con.WS_VISIBLE
        | commctrl.TVS_HASLINES
        | commctrl.TVS_LINESATROOT
        | commctrl.TVS_HASBUTTONS
    )

    dt = [
        ["Python Projects", (0, 0, 200, 200), style, None, (8, "MS Sans Serif")],
        ["SysTreeView32", None, win32ui.IDC_LIST1, (0, 0, 200, 200), cs],
    ]

    def __init__(self, hli_root):
        dialog.Dialog.__init__(self, self.dt)
        self.hier_list = hierlist.HierListWithItems(hli_root, win32ui.IDB_BROWSER_HIER)
        self.HookMessage(self.on_size, win32con.WM_SIZE)

    def OnInitDialog(self):
        self.hier_list.HierInit(self)
        return dialog.Dialog.OnInitDialog(self)

    def on_size(self, params):
        lparam = params[3]
        w = win32api.LOWORD(lparam)
        h = win32api.HIWORD(lparam)
        self.GetDlgItem(win32ui.IDC_LIST1).MoveWindow((0, 0, w, h))


def BrowseDialog():
    root = HLIRoot()
    if not root.IsExpandable():
        raise TypeError(
            "Browse() argument must have __dict__ attribute, or be a Browser supported type"
        )

    dlg = dynamic_browser(root)
    dlg.CreateWindow()


def DockableBrowserCreator(parent):
    root = HLIRoot()
    hl = hierlist.HierListWithItems(root, win32ui.IDB_BROWSER_HIER)

    style = (
        win32con.WS_CHILD
        | win32con.WS_VISIBLE
        | win32con.WS_BORDER
        | commctrl.TVS_HASLINES
        | commctrl.TVS_LINESATROOT
        | commctrl.TVS_HASBUTTONS
    )

    control = win32ui.CreateTreeCtrl()
    control.CreateWindow(style, (0, 0, 150, 300), parent, win32ui.IDC_LIST1)
    list = hl.HierInit(parent, control)
    return control


def DockablePathBrowser():
    import pywin.docking.DockingBar

    bar = pywin.docking.DockingBar.DockingBar()
    bar.CreateWindow(
        win32ui.GetMainFrame(), DockableBrowserCreator, "Path Browser", 0x8E0A
    )
    bar.SetBarStyle(
        bar.GetBarStyle()
        | afxres.CBRS_TOOLTIPS
        | afxres.CBRS_FLYBY
        | afxres.CBRS_SIZE_DYNAMIC
    )
    bar.EnableDocking(afxres.CBRS_ALIGN_ANY)
    win32ui.GetMainFrame().DockControlBar(bar)


# The "default" entry point
Browse = DockablePathBrowser
