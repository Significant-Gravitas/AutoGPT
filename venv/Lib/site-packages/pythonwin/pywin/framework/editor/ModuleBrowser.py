# ModuleBrowser.py - A view that provides a module browser for an editor document.
import pyclbr

import afxres
import commctrl
import pywin.framework.scriptutils
import pywin.mfc.docview
import win32api
import win32con
import win32ui
from pywin.tools import browser, hierlist


class HierListCLBRModule(hierlist.HierListItem):
    def __init__(self, modName, clbrdata):
        self.modName = modName
        self.clbrdata = clbrdata

    def GetText(self):
        return self.modName

    def GetSubList(self):
        ret = []
        for item in self.clbrdata.values():
            if (
                item.__class__ != pyclbr.Class
            ):  # ie, it is a pyclbr Function instance (only introduced post 1.5.2)
                ret.append(HierListCLBRFunction(item))
            else:
                ret.append(HierListCLBRClass(item))
        ret.sort()
        return ret

    def IsExpandable(self):
        return 1


class HierListCLBRItem(hierlist.HierListItem):
    def __init__(self, name, file, lineno, suffix=""):
        self.name = str(name)
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
            win32ui.SetStatusText("Can not locate the source code for this object.")

    def PerformItemSelected(self):
        if self.file is None:
            msg = "%s - source can not be located." % (self.name,)
        else:
            msg = "%s defined at line %d of %s" % (self.name, self.lineno, self.file)
        win32ui.SetStatusText(msg)


class HierListCLBRClass(HierListCLBRItem):
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
        HierListCLBRItem.__init__(self, name, file, lineno, suffix)

    def GetSubList(self):
        r1 = []
        for c in self.super:
            r1.append(HierListCLBRClass(c, " (Parent class)"))
        r1.sort()
        r2 = []
        for meth, lineno in self.methods.items():
            r2.append(HierListCLBRMethod(meth, self.file, lineno))
        r2.sort()
        return r1 + r2

    def IsExpandable(self):
        return len(self.methods) + len(self.super)

    def GetBitmapColumn(self):
        return 21


class HierListCLBRFunction(HierListCLBRItem):
    def __init__(self, clbrfunc, suffix=""):
        name = clbrfunc.name
        file = clbrfunc.file
        lineno = clbrfunc.lineno
        HierListCLBRItem.__init__(self, name, file, lineno, suffix)

    def GetBitmapColumn(self):
        return 22


class HierListCLBRMethod(HierListCLBRItem):
    def GetBitmapColumn(self):
        return 22


class HierListCLBRErrorItem(hierlist.HierListItem):
    def __init__(self, text):
        self.text = text

    def GetText(self):
        return self.text

    def GetSubList(self):
        return [HierListCLBRErrorItem(self.text)]

    def IsExpandable(self):
        return 0


class HierListCLBRErrorRoot(HierListCLBRErrorItem):
    def IsExpandable(self):
        return 1


class BrowserView(pywin.mfc.docview.TreeView):
    def OnInitialUpdate(self):
        self.list = None
        rc = self._obj_.OnInitialUpdate()
        self.HookMessage(self.OnSize, win32con.WM_SIZE)
        self.bDirty = 0
        self.destroying = 0
        return rc

    def DestroyBrowser(self):
        self.DestroyList()

    def OnActivateView(self, activate, av, dv):
        #        print "AV", self.bDirty, activate
        if activate:
            self.CheckRefreshList()
        return self._obj_.OnActivateView(activate, av, dv)

    def _MakeRoot(self):
        path = self.GetDocument().GetPathName()
        if not path:
            return HierListCLBRErrorRoot(
                "Error: Can not browse a file until it is saved"
            )
        else:
            mod, path = pywin.framework.scriptutils.GetPackageModuleName(path)
            if self.bDirty:
                what = "Refreshing"
                # Hack for pyclbr being too smart
                try:
                    del pyclbr._modules[mod]
                except (KeyError, AttributeError):
                    pass
            else:
                what = "Building"
            win32ui.SetStatusText("%s class list - please wait..." % (what,), 1)
            win32ui.DoWaitCursor(1)
            try:
                reader = pyclbr.readmodule_ex  # new version post 1.5.2
            except AttributeError:
                reader = pyclbr.readmodule
            try:
                data = reader(mod, [path])
                if data:
                    return HierListCLBRModule(mod, data)
                else:
                    return HierListCLBRErrorRoot("No Python classes in module.")

            finally:
                win32ui.DoWaitCursor(0)
                win32ui.SetStatusText(win32ui.LoadString(afxres.AFX_IDS_IDLEMESSAGE))

    def DestroyList(self):
        self.destroying = 1
        list = getattr(
            self, "list", None
        )  # If the document was not successfully opened, we may not have a list.
        self.list = None
        if list is not None:
            list.HierTerm()
        self.destroying = 0

    def CheckMadeList(self):
        if self.list is not None or self.destroying:
            return
        self.rootitem = root = self._MakeRoot()
        self.list = list = hierlist.HierListWithItems(root, win32ui.IDB_BROWSER_HIER)
        list.HierInit(self.GetParentFrame(), self)
        list.SetStyle(
            commctrl.TVS_HASLINES | commctrl.TVS_LINESATROOT | commctrl.TVS_HASBUTTONS
        )

    def CheckRefreshList(self):
        if self.bDirty:
            if self.list is None:
                self.CheckMadeList()
            else:
                new_root = self._MakeRoot()
                if self.rootitem.__class__ == new_root.__class__ == HierListCLBRModule:
                    self.rootitem.modName = new_root.modName
                    self.rootitem.clbrdata = new_root.clbrdata
                    self.list.Refresh()
                else:
                    self.list.AcceptRoot(self._MakeRoot())
            self.bDirty = 0

    def OnSize(self, params):
        lparam = params[3]
        w = win32api.LOWORD(lparam)
        h = win32api.HIWORD(lparam)
        if w != 0:
            self.CheckMadeList()
        elif w == 0:
            self.DestroyList()
        return 1

    def _UpdateUIForState(self):
        self.bDirty = 1
