# A sample context menu handler.
# Adds a 'Hello from Python' menu entry to .py files.  When clicked, a
# simple message box is displayed.
#
# To demostrate:
# * Execute this script to register the context menu.
# * Open Windows Explorer, and browse to a directory with a .py file.
# * Right-Click on a .py file - locate and click on 'Hello from Python' on
#   the context menu.

import pythoncom
import win32con
import win32gui
from win32com.shell import shell, shellcon


class ShellExtension:
    _reg_progid_ = "Python.ShellExtension.ContextMenu"
    _reg_desc_ = "Python Sample Shell Extension (context menu)"
    _reg_clsid_ = "{CED0336C-C9EE-4a7f-8D7F-C660393C381F}"
    _com_interfaces_ = [shell.IID_IShellExtInit, shell.IID_IContextMenu]
    _public_methods_ = shellcon.IContextMenu_Methods + shellcon.IShellExtInit_Methods

    def Initialize(self, folder, dataobj, hkey):
        print("Init", folder, dataobj, hkey)
        self.dataobj = dataobj

    def QueryContextMenu(self, hMenu, indexMenu, idCmdFirst, idCmdLast, uFlags):
        print("QCM", hMenu, indexMenu, idCmdFirst, idCmdLast, uFlags)
        # Query the items clicked on
        format_etc = win32con.CF_HDROP, None, 1, -1, pythoncom.TYMED_HGLOBAL
        sm = self.dataobj.GetData(format_etc)
        num_files = shell.DragQueryFile(sm.data_handle, -1)
        if num_files > 1:
            msg = "&Hello from Python (with %d files selected)" % num_files
        else:
            fname = shell.DragQueryFile(sm.data_handle, 0)
            msg = "&Hello from Python (with '%s' selected)" % fname
        idCmd = idCmdFirst
        items = ["First Python content menu item"]
        if (
            uFlags & 0x000F
        ) == shellcon.CMF_NORMAL:  # Check == here, since CMF_NORMAL=0
            print("CMF_NORMAL...")
            items.append(msg)
        elif uFlags & shellcon.CMF_VERBSONLY:
            print("CMF_VERBSONLY...")
            items.append(msg + " - shortcut")
        elif uFlags & shellcon.CMF_EXPLORE:
            print("CMF_EXPLORE...")
            items.append(msg + " - normal file, right-click in Explorer")
        elif uFlags & CMF_DEFAULTONLY:
            print("CMF_DEFAULTONLY...\r\n")
        else:
            print("** unknown flags", uFlags)
        win32gui.InsertMenu(
            hMenu, indexMenu, win32con.MF_SEPARATOR | win32con.MF_BYPOSITION, 0, None
        )
        indexMenu += 1
        for item in items:
            win32gui.InsertMenu(
                hMenu,
                indexMenu,
                win32con.MF_STRING | win32con.MF_BYPOSITION,
                idCmd,
                item,
            )
            indexMenu += 1
            idCmd += 1

        win32gui.InsertMenu(
            hMenu, indexMenu, win32con.MF_SEPARATOR | win32con.MF_BYPOSITION, 0, None
        )
        indexMenu += 1
        return idCmd - idCmdFirst  # Must return number of menu items we added.

    def InvokeCommand(self, ci):
        mask, hwnd, verb, params, dir, nShow, hotkey, hicon = ci
        win32gui.MessageBox(hwnd, "Hello", "Wow", win32con.MB_OK)

    def GetCommandString(self, cmd, typ):
        # If GetCommandString returns the same string for all items then
        # the shell seems to ignore all but one.  This is even true in
        # Win7 etc where there is no status bar (and hence this string seems
        # ignored)
        return "Hello from Python (cmd=%d)!!" % (cmd,)


def DllRegisterServer():
    import winreg

    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, "Python.File\\shellex")
    subkey = winreg.CreateKey(key, "ContextMenuHandlers")
    subkey2 = winreg.CreateKey(subkey, "PythonSample")
    winreg.SetValueEx(subkey2, None, 0, winreg.REG_SZ, ShellExtension._reg_clsid_)
    print(ShellExtension._reg_desc_, "registration complete.")


def DllUnregisterServer():
    import winreg

    try:
        key = winreg.DeleteKey(
            winreg.HKEY_CLASSES_ROOT,
            "Python.File\\shellex\\ContextMenuHandlers\\PythonSample",
        )
    except WindowsError as details:
        import errno

        if details.errno != errno.ENOENT:
            raise
    print(ShellExtension._reg_desc_, "unregistration complete.")


if __name__ == "__main__":
    from win32com.server import register

    register.UseCommandLine(
        ShellExtension,
        finalize_register=DllRegisterServer,
        finalize_unregister=DllUnregisterServer,
    )
