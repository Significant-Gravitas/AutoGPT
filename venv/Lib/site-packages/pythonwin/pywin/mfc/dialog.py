""" \
Base class for Dialogs.  Also contains a few useful utility functions
"""
# dialog.py
# Python class for Dialog Boxes in PythonWin.

import win32con
import win32ui

# sob - 2to3 doesn't see this as a relative import :(
from pywin.mfc import window


def dllFromDll(dllid):
    "given a 'dll' (maybe a dll, filename, etc), return a DLL object"
    if dllid == None:
        return None
    elif type("") == type(dllid):
        return win32ui.LoadLibrary(dllid)
    else:
        try:
            dllid.GetFileName()
        except AttributeError:
            raise TypeError("DLL parameter must be None, a filename or a dll object")
        return dllid


class Dialog(window.Wnd):
    "Base class for a dialog"

    def __init__(self, id, dllid=None):
        """id is the resource ID, or a template
        dllid may be None, a dll object, or a string with a dll name"""
        # must take a reference to the DLL until InitDialog.
        self.dll = dllFromDll(dllid)
        if type(id) == type([]):  # a template
            dlg = win32ui.CreateDialogIndirect(id)
        else:
            dlg = win32ui.CreateDialog(id, self.dll)
        window.Wnd.__init__(self, dlg)
        self.HookCommands()
        self.bHaveInit = None

    def HookCommands(self):
        pass

    def OnAttachedObjectDeath(self):
        self.data = self._obj_.data
        window.Wnd.OnAttachedObjectDeath(self)

    # provide virtuals.
    def OnOK(self):
        self._obj_.OnOK()

    def OnCancel(self):
        self._obj_.OnCancel()

    def OnInitDialog(self):
        self.bHaveInit = 1
        if self._obj_.data:
            self._obj_.UpdateData(0)
        return 1  # I did NOT set focus to a child window.

    def OnDestroy(self, msg):
        self.dll = None  # theoretically not needed if object destructs normally.

    # DDX support
    def AddDDX(self, *args):
        self._obj_.datalist.append(args)

    # Make a dialog object look like a dictionary for the DDX support
    def __bool__(self):
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, item):
        self._obj_.data[key] = item  # self.UpdateData(0)

    def keys(self):
        return list(self.data.keys())

    def items(self):
        return list(self.data.items())

    def values(self):
        return list(self.data.values())

    # XXX - needs py3k work!
    def has_key(self, key):
        return key in self.data


class PrintDialog(Dialog):
    "Base class for a print dialog"

    def __init__(
        self,
        pInfo,
        dlgID,
        printSetupOnly=0,
        flags=(
            win32ui.PD_ALLPAGES
            | win32ui.PD_USEDEVMODECOPIES
            | win32ui.PD_NOPAGENUMS
            | win32ui.PD_HIDEPRINTTOFILE
            | win32ui.PD_NOSELECTION
        ),
        parent=None,
        dllid=None,
    ):
        self.dll = dllFromDll(dllid)
        if type(dlgID) == type([]):  # a template
            raise TypeError("dlgID parameter must be an integer resource ID")
        dlg = win32ui.CreatePrintDialog(dlgID, printSetupOnly, flags, parent, self.dll)
        window.Wnd.__init__(self, dlg)
        self.HookCommands()
        self.bHaveInit = None
        self.pInfo = pInfo
        # init values (if PrintSetup is called, values still available)
        flags = pInfo.GetFlags()
        self["toFile"] = flags & win32ui.PD_PRINTTOFILE != 0
        self["direct"] = pInfo.GetDirect()
        self["preview"] = pInfo.GetPreview()
        self["continuePrinting"] = pInfo.GetContinuePrinting()
        self["curPage"] = pInfo.GetCurPage()
        self["numPreviewPages"] = pInfo.GetNumPreviewPages()
        self["userData"] = pInfo.GetUserData()
        self["draw"] = pInfo.GetDraw()
        self["pageDesc"] = pInfo.GetPageDesc()
        self["minPage"] = pInfo.GetMinPage()
        self["maxPage"] = pInfo.GetMaxPage()
        self["offsetPage"] = pInfo.GetOffsetPage()
        self["fromPage"] = pInfo.GetFromPage()
        self["toPage"] = pInfo.GetToPage()
        # these values updated after OnOK
        self["copies"] = 0
        self["deviceName"] = ""
        self["driverName"] = ""
        self["printAll"] = 0
        self["printCollate"] = 0
        self["printRange"] = 0
        self["printSelection"] = 0

    def OnInitDialog(self):
        self.pInfo.CreatePrinterDC()  # This also sets the hDC of the pInfo structure.
        return self._obj_.OnInitDialog()

    def OnCancel(self):
        del self.pInfo

    def OnOK(self):
        """DoModal has finished. Can now access the users choices"""
        self._obj_.OnOK()
        pInfo = self.pInfo
        # user values
        flags = pInfo.GetFlags()
        self["toFile"] = flags & win32ui.PD_PRINTTOFILE != 0
        self["direct"] = pInfo.GetDirect()
        self["preview"] = pInfo.GetPreview()
        self["continuePrinting"] = pInfo.GetContinuePrinting()
        self["curPage"] = pInfo.GetCurPage()
        self["numPreviewPages"] = pInfo.GetNumPreviewPages()
        self["userData"] = pInfo.GetUserData()
        self["draw"] = pInfo.GetDraw()
        self["pageDesc"] = pInfo.GetPageDesc()
        self["minPage"] = pInfo.GetMinPage()
        self["maxPage"] = pInfo.GetMaxPage()
        self["offsetPage"] = pInfo.GetOffsetPage()
        self["fromPage"] = pInfo.GetFromPage()
        self["toPage"] = pInfo.GetToPage()
        self["copies"] = pInfo.GetCopies()
        self["deviceName"] = pInfo.GetDeviceName()
        self["driverName"] = pInfo.GetDriverName()
        self["printAll"] = pInfo.PrintAll()
        self["printCollate"] = pInfo.PrintCollate()
        self["printRange"] = pInfo.PrintRange()
        self["printSelection"] = pInfo.PrintSelection()
        del self.pInfo


class PropertyPage(Dialog):
    "Base class for a Property Page"

    def __init__(self, id, dllid=None, caption=0):
        """id is the resource ID
        dllid may be None, a dll object, or a string with a dll name"""

        self.dll = dllFromDll(dllid)
        if self.dll:
            oldRes = win32ui.SetResource(self.dll)
        if type(id) == type([]):
            dlg = win32ui.CreatePropertyPageIndirect(id)
        else:
            dlg = win32ui.CreatePropertyPage(id, caption)
        if self.dll:
            win32ui.SetResource(oldRes)
        # dont call dialog init!
        window.Wnd.__init__(self, dlg)
        self.HookCommands()


class PropertySheet(window.Wnd):
    def __init__(self, caption, dll=None, pageList=None):  # parent=None, style,etc):
        "Initialize a property sheet.  pageList is a list of ID's"
        # must take a reference to the DLL until InitDialog.
        self.dll = dllFromDll(dll)
        self.sheet = win32ui.CreatePropertySheet(caption)
        window.Wnd.__init__(self, self.sheet)
        if not pageList is None:
            self.AddPage(pageList)

    def OnInitDialog(self):
        return self._obj_.OnInitDialog()

    def DoModal(self):
        if self.dll:
            oldRes = win32ui.SetResource(self.dll)
        rc = self.sheet.DoModal()
        if self.dll:
            win32ui.SetResource(oldRes)
        return rc

    def AddPage(self, pages):
        if self.dll:
            oldRes = win32ui.SetResource(self.dll)
        try:  # try list style access
            pages[0]
            isSeq = 1
        except (TypeError, KeyError):
            isSeq = 0
        if isSeq:
            for page in pages:
                self.DoAddSinglePage(page)
        else:
            self.DoAddSinglePage(pages)
        if self.dll:
            win32ui.SetResource(oldRes)

    def DoAddSinglePage(self, page):
        "Page may be page, or int ID. Assumes DLL setup"
        if type(page) == type(0):
            self.sheet.AddPage(win32ui.CreatePropertyPage(page))
        else:
            self.sheet.AddPage(page)


# define some app utility functions.
def GetSimpleInput(prompt, defValue="", title=None):
    """displays a dialog, and returns a string, or None if cancelled.
    args prompt, defValue='', title=main frames title"""
    # uses a simple dialog to return a string object.
    if title is None:
        title = win32ui.GetMainFrame().GetWindowText()
    # 2to3 insists on converting 'Dialog.__init__' to 'tkinter.dialog...'
    DlgBaseClass = Dialog

    class DlgSimpleInput(DlgBaseClass):
        def __init__(self, prompt, defValue, title):
            self.title = title
            DlgBaseClass.__init__(self, win32ui.IDD_SIMPLE_INPUT)
            self.AddDDX(win32ui.IDC_EDIT1, "result")
            self.AddDDX(win32ui.IDC_PROMPT1, "prompt")
            self._obj_.data["result"] = defValue
            self._obj_.data["prompt"] = prompt

        def OnInitDialog(self):
            self.SetWindowText(self.title)
            return DlgBaseClass.OnInitDialog(self)

    dlg = DlgSimpleInput(prompt, defValue, title)
    if dlg.DoModal() != win32con.IDOK:
        return None
    return dlg["result"]
