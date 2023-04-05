# dyndlg.py
# contributed by Curt Hagenlocher <chi@earthlink.net>

# Dialog Template params:
# 	Parameter 0 - Window caption
# 	Parameter 1 - Bounds (rect tuple)
# 	Parameter 2 - Window style
# 	Parameter 3 - Extended style
# 	Parameter 4 - Font tuple
# 	Parameter 5 - Menu name
# 	Parameter 6 - Window class
# Dialog item params:
# 	Parameter 0 - Window class
# 	Parameter 1 - Text
# 	Parameter 2 - ID
# 	Parameter 3 - Bounds
# 	Parameter 4 - Style
# 	Parameter 5 - Extended style
# 	Parameter 6 - Extra data


import win32con
import win32ui
from pywin.mfc import dialog, window


def MakeDlgTemplate():
    style = (
        win32con.DS_MODALFRAME
        | win32con.WS_POPUP
        | win32con.WS_VISIBLE
        | win32con.WS_CAPTION
        | win32con.WS_SYSMENU
        | win32con.DS_SETFONT
    )
    cs = win32con.WS_CHILD | win32con.WS_VISIBLE
    dlg = [
        ["Select Warehouse", (0, 0, 177, 93), style, None, (8, "MS Sans Serif")],
    ]
    dlg.append([130, "Current Warehouse:", -1, (7, 7, 69, 9), cs | win32con.SS_LEFT])
    dlg.append([130, "ASTORIA", 128, (16, 17, 99, 7), cs | win32con.SS_LEFT])
    dlg.append([130, "New &Warehouse:", -1, (7, 29, 69, 9), cs | win32con.SS_LEFT])
    s = win32con.WS_TABSTOP | cs
    # 	dlg.append([131, None, 130, (5, 40, 110, 48),
    # 		s | win32con.LBS_NOTIFY | win32con.LBS_SORT | win32con.LBS_NOINTEGRALHEIGHT | win32con.WS_VSCROLL | win32con.WS_BORDER])
    dlg.append(
        [
            "{8E27C92B-1264-101C-8A2F-040224009C02}",
            None,
            131,
            (5, 40, 110, 48),
            win32con.WS_TABSTOP,
        ]
    )

    dlg.append(
        [128, "OK", win32con.IDOK, (124, 5, 50, 14), s | win32con.BS_DEFPUSHBUTTON]
    )
    s = win32con.BS_PUSHBUTTON | s
    dlg.append([128, "Cancel", win32con.IDCANCEL, (124, 22, 50, 14), s])
    dlg.append([128, "&Help", 100, (124, 74, 50, 14), s])

    return dlg


def test1():
    win32ui.CreateDialogIndirect(MakeDlgTemplate()).DoModal()


def test2():
    dialog.Dialog(MakeDlgTemplate()).DoModal()


def test3():
    dlg = win32ui.LoadDialogResource(win32ui.IDD_SET_TABSTOPS)
    dlg[0][0] = "New Dialog Title"
    dlg[0][1] = (80, 20, 161, 60)
    dlg[1][1] = "&Confusion:"
    cs = (
        win32con.WS_CHILD
        | win32con.WS_VISIBLE
        | win32con.WS_TABSTOP
        | win32con.BS_PUSHBUTTON
    )
    dlg.append([128, "&Help", 100, (111, 41, 40, 14), cs])
    dialog.Dialog(dlg).DoModal()


def test4():
    page1 = dialog.PropertyPage(win32ui.LoadDialogResource(win32ui.IDD_PROPDEMO1))
    page2 = dialog.PropertyPage(win32ui.LoadDialogResource(win32ui.IDD_PROPDEMO2))
    ps = dialog.PropertySheet("Property Sheet/Page Demo", None, [page1, page2])
    ps.DoModal()


def testall():
    test1()
    test2()
    test3()
    test4()


if __name__ == "__main__":
    testall()
