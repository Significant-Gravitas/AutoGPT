# (sort-of) Registry editor
import commctrl
import dialog
import win32con
import win32ui


class RegistryControl:
    def __init__(self, key):
        self.key = key


class RegEditPropertyPage(dialog.PropertyPage):
    IDC_LISTVIEW = 1000

    def GetTemplate(self):
        "Return the template used to create this dialog"

        w = 152  # Dialog width
        h = 122  # Dialog height
        SS_STD = win32con.WS_CHILD | win32con.WS_VISIBLE
        FRAMEDLG_STD = win32con.WS_CAPTION | win32con.WS_SYSMENU
        style = (
            FRAMEDLG_STD
            | win32con.WS_VISIBLE
            | win32con.DS_SETFONT
            | win32con.WS_MINIMIZEBOX
        )
        template = [
            [self.caption, (0, 0, w, h), style, None, (8, "Helv")],
        ]
        lvStyle = (
            SS_STD
            | commctrl.LVS_EDITLABELS
            | commctrl.LVS_REPORT
            | commctrl.LVS_AUTOARRANGE
            | commctrl.LVS_ALIGNLEFT
            | win32con.WS_BORDER
            | win32con.WS_TABSTOP
        )
        template.append(
            ["SysListView32", "", self.IDC_LISTVIEW, (10, 10, 185, 100), lvStyle]
        )
        return template


class RegistryPage(RegEditPropertyPage):
    def __init__(self):
        self.caption = "Path"
        RegEditPropertyPage.__init__(self, self.GetTemplate())

    def OnInitDialog(self):
        self.listview = self.GetDlgItem(self.IDC_LISTVIEW)
        RegEditPropertyPage.OnInitDialog(self)
        # Setup the listview columns
        itemDetails = (commctrl.LVCFMT_LEFT, 100, "App", 0)
        self.listview.InsertColumn(0, itemDetails)
        itemDetails = (commctrl.LVCFMT_LEFT, 1024, "Paths", 0)
        self.listview.InsertColumn(1, itemDetails)

        index = self.listview.InsertItem(0, "App")
        self.listview.SetItemText(index, 1, "Path")


class RegistrySheet(dialog.PropertySheet):
    def __init__(self, title):
        dialog.PropertySheet.__init__(self, title)
        self.HookMessage(self.OnActivate, win32con.WM_ACTIVATE)

    def OnActivate(self, msg):
        print("OnAcivate")


def t():
    ps = RegistrySheet("Registry Settings")
    ps.AddPage(RegistryPage())
    ps.DoModal()


if __name__ == "__main__":
    t()
