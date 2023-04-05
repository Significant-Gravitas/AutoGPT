"""login -- PythonWin user ID and password dialog box

(Adapted from originally distributed with Mark Hammond's PythonWin - 
this now replaces it!)

login.GetLogin() displays a modal "OK/Cancel" dialog box with input
fields for a user ID and password. The password field input is masked
with *'s. GetLogin takes two optional parameters, a window title, and a
default user ID. If these parameters are omitted, the title defaults to
"Login", and the user ID is left blank. GetLogin returns a (userid, password)
tuple. GetLogin can be called from scripts running on the console - i.e. you
don't need to write a full-blown GUI app to use it.

login.GetPassword() is similar, except there is no username field.

Example:
import pywin.dialogs.login
title = "FTP Login"
def_user = "fred"
userid, password = pywin.dialogs.login.GetLogin(title, def_user)

Jim Eggleston, 28 August 1996
Merged with dlgpass and moved to pywin.dialogs by Mark Hammond Jan 1998.
"""

import win32api
import win32con
import win32ui
from pywin.mfc import dialog


def MakeLoginDlgTemplate(title):
    style = (
        win32con.DS_MODALFRAME
        | win32con.WS_POPUP
        | win32con.WS_VISIBLE
        | win32con.WS_CAPTION
        | win32con.WS_SYSMENU
        | win32con.DS_SETFONT
    )
    cs = win32con.WS_CHILD | win32con.WS_VISIBLE

    # Window frame and title
    dlg = [
        [title, (0, 0, 184, 40), style, None, (8, "MS Sans Serif")],
    ]

    # ID label and text box
    dlg.append([130, "User ID:", -1, (7, 9, 69, 9), cs | win32con.SS_LEFT])
    s = cs | win32con.WS_TABSTOP | win32con.WS_BORDER
    dlg.append(["EDIT", None, win32ui.IDC_EDIT1, (50, 7, 60, 12), s])

    # Password label and text box
    dlg.append([130, "Password:", -1, (7, 22, 69, 9), cs | win32con.SS_LEFT])
    s = cs | win32con.WS_TABSTOP | win32con.WS_BORDER
    dlg.append(
        ["EDIT", None, win32ui.IDC_EDIT2, (50, 20, 60, 12), s | win32con.ES_PASSWORD]
    )

    # OK/Cancel Buttons
    s = cs | win32con.WS_TABSTOP
    dlg.append(
        [128, "OK", win32con.IDOK, (124, 5, 50, 14), s | win32con.BS_DEFPUSHBUTTON]
    )
    s = win32con.BS_PUSHBUTTON | s
    dlg.append([128, "Cancel", win32con.IDCANCEL, (124, 20, 50, 14), s])
    return dlg


def MakePasswordDlgTemplate(title):
    style = (
        win32con.DS_MODALFRAME
        | win32con.WS_POPUP
        | win32con.WS_VISIBLE
        | win32con.WS_CAPTION
        | win32con.WS_SYSMENU
        | win32con.DS_SETFONT
    )
    cs = win32con.WS_CHILD | win32con.WS_VISIBLE
    # Window frame and title
    dlg = [
        [title, (0, 0, 177, 45), style, None, (8, "MS Sans Serif")],
    ]

    # Password label and text box
    dlg.append([130, "Password:", -1, (7, 7, 69, 9), cs | win32con.SS_LEFT])
    s = cs | win32con.WS_TABSTOP | win32con.WS_BORDER
    dlg.append(
        ["EDIT", None, win32ui.IDC_EDIT1, (50, 7, 60, 12), s | win32con.ES_PASSWORD]
    )

    # OK/Cancel Buttons
    s = cs | win32con.WS_TABSTOP | win32con.BS_PUSHBUTTON
    dlg.append(
        [128, "OK", win32con.IDOK, (124, 5, 50, 14), s | win32con.BS_DEFPUSHBUTTON]
    )
    dlg.append([128, "Cancel", win32con.IDCANCEL, (124, 22, 50, 14), s])
    return dlg


class LoginDlg(dialog.Dialog):
    Cancel = 0

    def __init__(self, title):
        dialog.Dialog.__init__(self, MakeLoginDlgTemplate(title))
        self.AddDDX(win32ui.IDC_EDIT1, "userid")
        self.AddDDX(win32ui.IDC_EDIT2, "password")


def GetLogin(title="Login", userid="", password=""):
    d = LoginDlg(title)
    d["userid"] = userid
    d["password"] = password
    if d.DoModal() != win32con.IDOK:
        return (None, None)
    else:
        return (d["userid"], d["password"])


class PasswordDlg(dialog.Dialog):
    def __init__(self, title):
        dialog.Dialog.__init__(self, MakePasswordDlgTemplate(title))
        self.AddDDX(win32ui.IDC_EDIT1, "password")


def GetPassword(title="Password", password=""):
    d = PasswordDlg(title)
    d["password"] = password
    if d.DoModal() != win32con.IDOK:
        return None
    return d["password"]


if __name__ == "__main__":
    import sys

    title = "Login"
    def_user = ""
    if len(sys.argv) > 1:
        title = sys.argv[1]
    if len(sys.argv) > 2:
        def_userid = sys.argv[2]
    userid, password = GetLogin(title, def_user)
    if userid == password == None:
        print("User pressed Cancel")
    else:
        print("User ID: ", userid)
        print("Password:", password)
        newpassword = GetPassword("Reenter just for fun", password)
        if newpassword is None:
            print("User cancelled")
        else:
            what = ""
            if newpassword != password:
                what = "not "
            print("The passwords did %smatch" % (what))
