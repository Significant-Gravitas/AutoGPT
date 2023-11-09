import commctrl
import pythoncom
import win32api
import win32con
import win32ui
from pywin.mfc import dialog


class TLBrowserException(Exception):
    "TypeLib browser internal error"


error = TLBrowserException

FRAMEDLG_STD = win32con.WS_CAPTION | win32con.WS_SYSMENU
SS_STD = win32con.WS_CHILD | win32con.WS_VISIBLE
BS_STD = SS_STD | win32con.WS_TABSTOP
ES_STD = BS_STD | win32con.WS_BORDER
LBS_STD = (
    ES_STD | win32con.LBS_NOTIFY | win32con.LBS_NOINTEGRALHEIGHT | win32con.WS_VSCROLL
)
CBS_STD = ES_STD | win32con.CBS_NOINTEGRALHEIGHT | win32con.WS_VSCROLL

typekindmap = {
    pythoncom.TKIND_ENUM: "Enumeration",
    pythoncom.TKIND_RECORD: "Record",
    pythoncom.TKIND_MODULE: "Module",
    pythoncom.TKIND_INTERFACE: "Interface",
    pythoncom.TKIND_DISPATCH: "Dispatch",
    pythoncom.TKIND_COCLASS: "CoClass",
    pythoncom.TKIND_ALIAS: "Alias",
    pythoncom.TKIND_UNION: "Union",
}

TypeBrowseDialog_Parent = dialog.Dialog


class TypeBrowseDialog(TypeBrowseDialog_Parent):
    "Browse a type library"

    IDC_TYPELIST = 1000
    IDC_MEMBERLIST = 1001
    IDC_PARAMLIST = 1002
    IDC_LISTVIEW = 1003

    def __init__(self, typefile=None):
        TypeBrowseDialog_Parent.__init__(self, self.GetTemplate())
        try:
            if typefile:
                self.tlb = pythoncom.LoadTypeLib(typefile)
            else:
                self.tlb = None
        except pythoncom.ole_error:
            self.MessageBox("The file does not contain type information")
            self.tlb = None
        self.HookCommand(self.CmdTypeListbox, self.IDC_TYPELIST)
        self.HookCommand(self.CmdMemberListbox, self.IDC_MEMBERLIST)

    def OnAttachedObjectDeath(self):
        self.tlb = None
        self.typeinfo = None
        self.attr = None
        return TypeBrowseDialog_Parent.OnAttachedObjectDeath(self)

    def _SetupMenu(self):
        menu = win32ui.CreateMenu()
        flags = win32con.MF_STRING | win32con.MF_ENABLED
        menu.AppendMenu(flags, win32ui.ID_FILE_OPEN, "&Open...")
        menu.AppendMenu(flags, win32con.IDCANCEL, "&Close")
        mainMenu = win32ui.CreateMenu()
        mainMenu.AppendMenu(flags | win32con.MF_POPUP, menu.GetHandle(), "&File")
        self.SetMenu(mainMenu)
        self.HookCommand(self.OnFileOpen, win32ui.ID_FILE_OPEN)

    def OnFileOpen(self, id, code):
        openFlags = win32con.OFN_OVERWRITEPROMPT | win32con.OFN_FILEMUSTEXIST
        fspec = "Type Libraries (*.tlb, *.olb)|*.tlb;*.olb|OCX Files (*.ocx)|*.ocx|DLL's (*.dll)|*.dll|All Files (*.*)|*.*||"
        dlg = win32ui.CreateFileDialog(1, None, None, openFlags, fspec)
        if dlg.DoModal() == win32con.IDOK:
            try:
                self.tlb = pythoncom.LoadTypeLib(dlg.GetPathName())
            except pythoncom.ole_error:
                self.MessageBox("The file does not contain type information")
                self.tlb = None
            self._SetupTLB()

    def OnInitDialog(self):
        self._SetupMenu()
        self.typelb = self.GetDlgItem(self.IDC_TYPELIST)
        self.memberlb = self.GetDlgItem(self.IDC_MEMBERLIST)
        self.paramlb = self.GetDlgItem(self.IDC_PARAMLIST)
        self.listview = self.GetDlgItem(self.IDC_LISTVIEW)

        # Setup the listview columns
        itemDetails = (commctrl.LVCFMT_LEFT, 100, "Item", 0)
        self.listview.InsertColumn(0, itemDetails)
        itemDetails = (commctrl.LVCFMT_LEFT, 1024, "Details", 0)
        self.listview.InsertColumn(1, itemDetails)

        if self.tlb is None:
            self.OnFileOpen(None, None)
        else:
            self._SetupTLB()
        return TypeBrowseDialog_Parent.OnInitDialog(self)

    def _SetupTLB(self):
        self.typelb.ResetContent()
        self.memberlb.ResetContent()
        self.paramlb.ResetContent()
        self.typeinfo = None
        self.attr = None
        if self.tlb is None:
            return
        n = self.tlb.GetTypeInfoCount()
        for i in range(n):
            self.typelb.AddString(self.tlb.GetDocumentation(i)[0])

    def _SetListviewTextItems(self, items):
        self.listview.DeleteAllItems()
        index = -1
        for item in items:
            index = self.listview.InsertItem(index + 1, item[0])
            data = item[1]
            if data is None:
                data = ""
            self.listview.SetItemText(index, 1, data)

    def SetupAllInfoTypes(self):
        infos = self._GetMainInfoTypes() + self._GetMethodInfoTypes()
        self._SetListviewTextItems(infos)

    def _GetMainInfoTypes(self):
        pos = self.typelb.GetCurSel()
        if pos < 0:
            return []
        docinfo = self.tlb.GetDocumentation(pos)
        infos = [("GUID", str(self.attr[0]))]
        infos.append(("Help File", docinfo[3]))
        infos.append(("Help Context", str(docinfo[2])))
        try:
            infos.append(("Type Kind", typekindmap[self.tlb.GetTypeInfoType(pos)]))
        except:
            pass

        info = self.tlb.GetTypeInfo(pos)
        attr = info.GetTypeAttr()
        infos.append(("Attributes", str(attr)))

        for j in range(attr[8]):
            flags = info.GetImplTypeFlags(j)
            refInfo = info.GetRefTypeInfo(info.GetRefTypeOfImplType(j))
            doc = refInfo.GetDocumentation(-1)
            attr = refInfo.GetTypeAttr()
            typeKind = attr[5]
            typeFlags = attr[11]

            desc = doc[0]
            desc = desc + ", Flags=0x%x, typeKind=0x%x, typeFlags=0x%x" % (
                flags,
                typeKind,
                typeFlags,
            )
            if flags & pythoncom.IMPLTYPEFLAG_FSOURCE:
                desc = desc + "(Source)"
            infos.append(("Implements", desc))

        return infos

    def _GetMethodInfoTypes(self):
        pos = self.memberlb.GetCurSel()
        if pos < 0:
            return []

        realPos, isMethod = self._GetRealMemberPos(pos)
        ret = []
        if isMethod:
            funcDesc = self.typeinfo.GetFuncDesc(realPos)
            id = funcDesc[0]
            ret.append(("Func Desc", str(funcDesc)))
        else:
            id = self.typeinfo.GetVarDesc(realPos)[0]

        docinfo = self.typeinfo.GetDocumentation(id)
        ret.append(("Help String", docinfo[1]))
        ret.append(("Help Context", str(docinfo[2])))
        return ret

    def CmdTypeListbox(self, id, code):
        if code == win32con.LBN_SELCHANGE:
            pos = self.typelb.GetCurSel()
            if pos >= 0:
                self.memberlb.ResetContent()
                self.typeinfo = self.tlb.GetTypeInfo(pos)
                self.attr = self.typeinfo.GetTypeAttr()
                for i in range(self.attr[7]):
                    id = self.typeinfo.GetVarDesc(i)[0]
                    self.memberlb.AddString(self.typeinfo.GetNames(id)[0])
                for i in range(self.attr[6]):
                    id = self.typeinfo.GetFuncDesc(i)[0]
                    self.memberlb.AddString(self.typeinfo.GetNames(id)[0])
                self.SetupAllInfoTypes()
            return 1

    def _GetRealMemberPos(self, pos):
        pos = self.memberlb.GetCurSel()
        if pos >= self.attr[7]:
            return pos - self.attr[7], 1
        elif pos >= 0:
            return pos, 0
        else:
            raise error("The position is not valid")

    def CmdMemberListbox(self, id, code):
        if code == win32con.LBN_SELCHANGE:
            self.paramlb.ResetContent()
            pos = self.memberlb.GetCurSel()
            realPos, isMethod = self._GetRealMemberPos(pos)
            if isMethod:
                id = self.typeinfo.GetFuncDesc(realPos)[0]
                names = self.typeinfo.GetNames(id)
                for i in range(len(names)):
                    if i > 0:
                        self.paramlb.AddString(names[i])
            self.SetupAllInfoTypes()
            return 1

    def GetTemplate(self):
        "Return the template used to create this dialog"

        w = 272  # Dialog width
        h = 192  # Dialog height
        style = (
            FRAMEDLG_STD
            | win32con.WS_VISIBLE
            | win32con.DS_SETFONT
            | win32con.WS_MINIMIZEBOX
        )
        template = [
            ["Type Library Browser", (0, 0, w, h), style, None, (8, "Helv")],
        ]
        template.append([130, "&Type", -1, (10, 10, 62, 9), SS_STD | win32con.SS_LEFT])
        template.append([131, None, self.IDC_TYPELIST, (10, 20, 80, 80), LBS_STD])
        template.append(
            [130, "&Members", -1, (100, 10, 62, 9), SS_STD | win32con.SS_LEFT]
        )
        template.append([131, None, self.IDC_MEMBERLIST, (100, 20, 80, 80), LBS_STD])
        template.append(
            [130, "&Parameters", -1, (190, 10, 62, 9), SS_STD | win32con.SS_LEFT]
        )
        template.append([131, None, self.IDC_PARAMLIST, (190, 20, 75, 80), LBS_STD])

        lvStyle = (
            SS_STD
            | commctrl.LVS_REPORT
            | commctrl.LVS_AUTOARRANGE
            | commctrl.LVS_ALIGNLEFT
            | win32con.WS_BORDER
            | win32con.WS_TABSTOP
        )
        template.append(
            ["SysListView32", "", self.IDC_LISTVIEW, (10, 110, 255, 65), lvStyle]
        )

        return template


if __name__ == "__main__":
    import sys

    fname = None
    try:
        fname = sys.argv[1]
    except:
        pass
    dlg = TypeBrowseDialog(fname)
    if win32api.GetConsoleTitle():  # empty string w/o console
        dlg.DoModal()
    else:
        dlg.CreateWindow(win32ui.GetMainFrame())
