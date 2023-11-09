# A Demo of Pythonwin's Dialog and Property Page support.

###################
#
# First demo - use the built-in to Pythonwin "Tab Stop" dialog, but
# customise it heavily.
#
# ID's for the tabstop dialog - out test.
#
import win32con
import win32ui
from pywin.mfc import dialog
from win32con import IDCANCEL
from win32ui import IDC_EDIT_TABS, IDC_PROMPT_TABS, IDD_SET_TABSTOPS


class TestDialog(dialog.Dialog):
    def __init__(self, modal=1):
        dialog.Dialog.__init__(self, IDD_SET_TABSTOPS)
        self.counter = 0
        if modal:
            self.DoModal()
        else:
            self.CreateWindow()

    def OnInitDialog(self):
        # Set the caption of the dialog itself.
        self.SetWindowText("Used to be Tab Stops!")
        # Get a child control, remember it, and change its text.
        self.edit = self.GetDlgItem(IDC_EDIT_TABS)  # the text box.
        self.edit.SetWindowText("Test")
        # Hook a Windows message for the dialog.
        self.edit.HookMessage(self.KillFocus, win32con.WM_KILLFOCUS)
        # Get the prompt control, and change its next.
        prompt = self.GetDlgItem(IDC_PROMPT_TABS)  # the prompt box.
        prompt.SetWindowText("Prompt")
        # And the same for the button..
        cancel = self.GetDlgItem(IDCANCEL)  # the cancel button
        cancel.SetWindowText("&Kill me")

        # And just for demonstration purposes, we hook the notify message for the dialog.
        # This allows us to be notified when the Edit Control text changes.
        self.HookCommand(self.OnNotify, IDC_EDIT_TABS)

    def OnNotify(self, controlid, code):
        if code == win32con.EN_CHANGE:
            print("Edit text changed!")
        return 1  # I handled this, so no need to call defaults!

    # kill focus for the edit box.
    # Simply increment the value in the text box.
    def KillFocus(self, msg):
        self.counter = self.counter + 1
        if self.edit != None:
            self.edit.SetWindowText(str(self.counter))

    # Called when the dialog box is terminating...
    def OnDestroy(self, msg):
        del self.edit
        del self.counter


# A very simply Property Sheet.
# We only make a new class for demonstration purposes.
class TestSheet(dialog.PropertySheet):
    def __init__(self, title):
        dialog.PropertySheet.__init__(self, title)
        self.HookMessage(self.OnActivate, win32con.WM_ACTIVATE)

    def OnActivate(self, msg):
        pass


# A very simply Property Page, which will be "owned" by the above
# Property Sheet.
# We create a new class, just so we can hook a control notification.
class TestPage(dialog.PropertyPage):
    def OnInitDialog(self):
        # We use the HookNotify function to allow Python to respond to
        # Windows WM_NOTIFY messages.
        # In this case, we are interested in BN_CLICKED messages.
        self.HookNotify(self.OnNotify, win32con.BN_CLICKED)

    def OnNotify(self, std, extra):
        print("OnNotify", std, extra)


# Some code that actually uses these objects.
def demo(modal=0):
    TestDialog(modal)

    # property sheet/page demo
    ps = win32ui.CreatePropertySheet("Property Sheet/Page Demo")
    # Create a completely standard PropertyPage.
    page1 = win32ui.CreatePropertyPage(win32ui.IDD_PROPDEMO1)
    # Create our custom property page.
    page2 = TestPage(win32ui.IDD_PROPDEMO2)
    ps.AddPage(page1)
    ps.AddPage(page2)
    if modal:
        ps.DoModal()
    else:
        style = (
            win32con.WS_SYSMENU
            | win32con.WS_POPUP
            | win32con.WS_CAPTION
            | win32con.DS_MODALFRAME
            | win32con.WS_VISIBLE
        )
        styleex = win32con.WS_EX_DLGMODALFRAME | win32con.WS_EX_PALETTEWINDOW
        ps.CreateWindow(win32ui.GetMainFrame(), style, styleex)


def test(modal=1):
    # 	dlg=dialog.Dialog(1010)
    # 	dlg.CreateWindow()
    # 	dlg.EndDialog(0)
    # 	del dlg
    # 	return
    # property sheet/page demo
    ps = TestSheet("Property Sheet/Page Demo")
    page1 = win32ui.CreatePropertyPage(win32ui.IDD_PROPDEMO1)
    page2 = win32ui.CreatePropertyPage(win32ui.IDD_PROPDEMO2)
    ps.AddPage(page1)
    ps.AddPage(page2)
    del page1
    del page2
    if modal:
        ps.DoModal()
    else:
        ps.CreateWindow(win32ui.GetMainFrame())
    return ps


def d():
    dlg = win32ui.CreateDialog(win32ui.IDD_DEBUGGER)
    dlg.datalist.append((win32ui.IDC_DBG_RADIOSTACK, "radio"))
    print("data list is ", dlg.datalist)
    dlg.data["radio"] = 1
    dlg.DoModal()
    print(dlg.data["radio"])


if __name__ == "__main__":
    demo(1)
