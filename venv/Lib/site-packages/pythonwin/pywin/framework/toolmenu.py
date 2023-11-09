# toolmenu.py

import sys

import win32api
import win32con
import win32ui

from . import app

tools = {}
idPos = 100

# The default items should no tools menu exist in the INI file.
defaultToolMenuItems = [
    ("Browser", "win32ui.GetApp().OnViewBrowse(0,0)"),
    (
        "Browse PythonPath",
        "from pywin.tools import browseProjects;browseProjects.Browse()",
    ),
    ("Edit Python Path", "from pywin.tools import regedit;regedit.EditRegistry()"),
    ("COM Makepy utility", "from win32com.client import makepy;makepy.main()"),
    (
        "COM Browser",
        "from win32com.client import combrowse;combrowse.main(modal=False)",
    ),
    (
        "Trace Collector Debugging tool",
        "from pywin.tools import TraceCollector;TraceCollector.MakeOutputWindow()",
    ),
]


def LoadToolMenuItems():
    # Load from the registry.
    items = []
    lookNo = 1
    while 1:
        menu = win32ui.GetProfileVal("Tools Menu\\%s" % lookNo, "", "")
        if menu == "":
            break
        cmd = win32ui.GetProfileVal("Tools Menu\\%s" % lookNo, "Command", "")
        items.append((menu, cmd))
        lookNo = lookNo + 1

    if len(items) == 0:
        items = defaultToolMenuItems
    return items


def WriteToolMenuItems(items):
    # Items is a list of (menu, command)
    # Delete the entire registry tree.
    try:
        mainKey = win32ui.GetAppRegistryKey()
        toolKey = win32api.RegOpenKey(mainKey, "Tools Menu")
    except win32ui.error:
        toolKey = None
    if toolKey is not None:
        while 1:
            try:
                subkey = win32api.RegEnumKey(toolKey, 0)
            except win32api.error:
                break
            win32api.RegDeleteKey(toolKey, subkey)
    # Keys are now removed - write the new ones.
    # But first check if we have the defaults - and if so, dont write anything!
    if items == defaultToolMenuItems:
        return
    itemNo = 1
    for menu, cmd in items:
        win32ui.WriteProfileVal("Tools Menu\\%s" % itemNo, "", menu)
        win32ui.WriteProfileVal("Tools Menu\\%s" % itemNo, "Command", cmd)
        itemNo = itemNo + 1


def SetToolsMenu(menu, menuPos=None):
    global tools
    global idPos

    # todo - check the menu does not already exist.
    # Create the new menu
    toolsMenu = win32ui.CreatePopupMenu()

    # Load from the ini file.
    items = LoadToolMenuItems()
    for menuString, cmd in items:
        tools[idPos] = (menuString, cmd, menuString)
        toolsMenu.AppendMenu(
            win32con.MF_ENABLED | win32con.MF_STRING, idPos, menuString
        )
        win32ui.GetMainFrame().HookCommand(HandleToolCommand, idPos)
        idPos = idPos + 1

    # Find the correct spot to insert the new tools menu.
    if menuPos is None:
        menuPos = menu.GetMenuItemCount() - 2
        if menuPos < 0:
            menuPos = 0

    menu.InsertMenu(
        menuPos,
        win32con.MF_BYPOSITION
        | win32con.MF_ENABLED
        | win32con.MF_STRING
        | win32con.MF_POPUP,
        toolsMenu.GetHandle(),
        "&Tools",
    )


def HandleToolCommand(cmd, code):
    import re
    import traceback

    global tools
    (menuString, pyCmd, desc) = tools[cmd]
    win32ui.SetStatusText("Executing tool %s" % desc, 1)
    pyCmd = re.sub("\\\\n", "\n", pyCmd)
    win32ui.DoWaitCursor(1)
    oldFlag = None
    try:
        oldFlag = sys.stdout.template.writeQueueing
        sys.stdout.template.writeQueueing = 0
    except (NameError, AttributeError):
        pass

    try:
        exec("%s\n" % pyCmd)
        worked = 1
    except SystemExit:
        # The program raised a SystemExit - ignore it.
        worked = 1
    except:
        print("Failed to execute command:\n%s" % pyCmd)
        traceback.print_exc()
        worked = 0
    if oldFlag is not None:
        sys.stdout.template.writeQueueing = oldFlag
    win32ui.DoWaitCursor(0)
    if worked:
        text = "Completed successfully."
    else:
        text = "Error executing %s." % desc
    win32ui.SetStatusText(text, 1)


# The property page for maintaing the items on the Tools menu.
import commctrl
from pywin.mfc import dialog

if win32ui.UNICODE:
    LVN_ENDLABELEDIT = commctrl.LVN_ENDLABELEDITW
else:
    LVN_ENDLABELEDIT = commctrl.LVN_ENDLABELEDITA


class ToolMenuPropPage(dialog.PropertyPage):
    def __init__(self):
        self.bImChangingEditControls = 0  # Am I programatically changing the controls?
        dialog.PropertyPage.__init__(self, win32ui.IDD_PP_TOOLMENU)

    def OnInitDialog(self):
        self.editMenuCommand = self.GetDlgItem(win32ui.IDC_EDIT2)
        self.butNew = self.GetDlgItem(win32ui.IDC_BUTTON3)

        # Now hook the change notification messages for the edit controls.
        self.HookCommand(self.OnCommandEditControls, win32ui.IDC_EDIT1)
        self.HookCommand(self.OnCommandEditControls, win32ui.IDC_EDIT2)

        self.HookNotify(self.OnNotifyListControl, commctrl.LVN_ITEMCHANGED)
        self.HookNotify(self.OnNotifyListControlEndLabelEdit, commctrl.LVN_ENDLABELEDIT)

        # Hook the button clicks.
        self.HookCommand(self.OnButtonNew, win32ui.IDC_BUTTON3)  # New Item
        self.HookCommand(self.OnButtonDelete, win32ui.IDC_BUTTON4)  # Delete item
        self.HookCommand(self.OnButtonMove, win32ui.IDC_BUTTON1)  # Move up
        self.HookCommand(self.OnButtonMove, win32ui.IDC_BUTTON2)  # Move down

        # Setup the columns in the list control
        lc = self.GetDlgItem(win32ui.IDC_LIST1)
        rect = lc.GetWindowRect()
        cx = rect[2] - rect[0]
        colSize = cx / 2 - win32api.GetSystemMetrics(win32con.SM_CXBORDER) - 1

        item = commctrl.LVCFMT_LEFT, colSize, "Menu Text"
        lc.InsertColumn(0, item)

        item = commctrl.LVCFMT_LEFT, colSize, "Python Command"
        lc.InsertColumn(1, item)

        # Insert the existing tools menu
        itemNo = 0
        for desc, cmd in LoadToolMenuItems():
            lc.InsertItem(itemNo, desc)
            lc.SetItemText(itemNo, 1, cmd)
            itemNo = itemNo + 1

        self.listControl = lc
        return dialog.PropertyPage.OnInitDialog(self)

    def OnOK(self):
        # Write the menu back to the registry.
        items = []
        itemLook = 0
        while 1:
            try:
                text = self.listControl.GetItemText(itemLook, 0)
                if not text:
                    break
                items.append((text, self.listControl.GetItemText(itemLook, 1)))
            except win32ui.error:
                # no more items!
                break
            itemLook = itemLook + 1
        WriteToolMenuItems(items)
        return self._obj_.OnOK()

    def OnCommandEditControls(self, id, cmd):
        # 		print "OnEditControls", id, cmd
        if cmd == win32con.EN_CHANGE and not self.bImChangingEditControls:
            itemNo = self.listControl.GetNextItem(-1, commctrl.LVNI_SELECTED)
            newText = self.editMenuCommand.GetWindowText()
            self.listControl.SetItemText(itemNo, 1, newText)

        return 0

    def OnNotifyListControlEndLabelEdit(self, id, cmd):
        newText = self.listControl.GetEditControl().GetWindowText()
        itemNo = self.listControl.GetNextItem(-1, commctrl.LVNI_SELECTED)
        self.listControl.SetItemText(itemNo, 0, newText)

    def OnNotifyListControl(self, id, cmd):
        # 		print id, cmd
        try:
            itemNo = self.listControl.GetNextItem(-1, commctrl.LVNI_SELECTED)
        except win32ui.error:  # No selection!
            return

        self.bImChangingEditControls = 1
        try:
            item = self.listControl.GetItem(itemNo, 1)
            self.editMenuCommand.SetWindowText(item[4])
        finally:
            self.bImChangingEditControls = 0

        return 0  # we have handled this!

    def OnButtonNew(self, id, cmd):
        if cmd == win32con.BN_CLICKED:
            newIndex = self.listControl.GetItemCount()
            self.listControl.InsertItem(newIndex, "Click to edit the text")
            self.listControl.EnsureVisible(newIndex, 0)

    def OnButtonMove(self, id, cmd):
        if cmd == win32con.BN_CLICKED:
            try:
                itemNo = self.listControl.GetNextItem(-1, commctrl.LVNI_SELECTED)
            except win32ui.error:
                return
            menu = self.listControl.GetItemText(itemNo, 0)
            cmd = self.listControl.GetItemText(itemNo, 1)
            if id == win32ui.IDC_BUTTON1:
                # Move up
                if itemNo > 0:
                    self.listControl.DeleteItem(itemNo)
                    # reinsert it.
                    self.listControl.InsertItem(itemNo - 1, menu)
                    self.listControl.SetItemText(itemNo - 1, 1, cmd)
            else:
                # Move down.
                if itemNo < self.listControl.GetItemCount() - 1:
                    self.listControl.DeleteItem(itemNo)
                    # reinsert it.
                    self.listControl.InsertItem(itemNo + 1, menu)
                    self.listControl.SetItemText(itemNo + 1, 1, cmd)

    def OnButtonDelete(self, id, cmd):
        if cmd == win32con.BN_CLICKED:
            try:
                itemNo = self.listControl.GetNextItem(-1, commctrl.LVNI_SELECTED)
            except win32ui.error:  # No selection!
                return
            self.listControl.DeleteItem(itemNo)
