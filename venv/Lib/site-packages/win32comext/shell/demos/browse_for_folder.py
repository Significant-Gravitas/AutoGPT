# A couple of samples using SHBrowseForFolder

import os

import win32gui
from win32com.shell import shell, shellcon


# A callback procedure - called by SHBrowseForFolder
def BrowseCallbackProc(hwnd, msg, lp, data):
    if msg == shellcon.BFFM_INITIALIZED:
        win32gui.SendMessage(hwnd, shellcon.BFFM_SETSELECTION, 1, data)
    elif msg == shellcon.BFFM_SELCHANGED:
        # Set the status text of the
        # For this message, 'lp' is the address of the PIDL.
        pidl = shell.AddressAsPIDL(lp)
        try:
            path = shell.SHGetPathFromIDList(pidl)
            win32gui.SendMessage(hwnd, shellcon.BFFM_SETSTATUSTEXT, 0, path)
        except shell.error:
            # No path for this PIDL
            pass


if __name__ == "__main__":
    # Demonstrate a dialog with the cwd selected as the default - this
    # must be done via a callback function.
    flags = shellcon.BIF_STATUSTEXT
    shell.SHBrowseForFolder(
        0,  # parent HWND
        None,  # root PIDL.
        "Default of %s" % os.getcwd(),  # title
        flags,  # flags
        BrowseCallbackProc,  # callback function
        os.getcwd(),  # 'data' param for the callback
    )
    # Browse from this directory down only.
    # Get the PIDL for the cwd.
    desktop = shell.SHGetDesktopFolder()
    cb, pidl, extra = desktop.ParseDisplayName(0, None, os.getcwd())
    shell.SHBrowseForFolder(
        0,  # parent HWND
        pidl,  # root PIDL.
        "From %s down only" % os.getcwd(),  # title
    )
