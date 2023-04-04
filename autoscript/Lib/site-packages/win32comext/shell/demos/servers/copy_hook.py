# A sample shell copy hook.

# To demostrate:
# * Execute this script to register the context menu.
# * Open Windows Explorer
# * Attempt to move or copy a directory.
# * Note our hook's dialog is displayed.

import pythoncom
import win32con
import win32gui
from win32com.shell import shell, shellcon


# Our shell extension.
class ShellExtension:
    _reg_progid_ = "Python.ShellExtension.CopyHook"
    _reg_desc_ = "Python Sample Shell Extension (copy hook)"
    _reg_clsid_ = "{1845b6ba-2bbd-4197-b930-46d8651497c1}"
    _com_interfaces_ = [shell.IID_ICopyHook]
    _public_methods_ = ["CopyCallBack"]

    def CopyCallBack(self, hwnd, func, flags, srcName, srcAttr, destName, destAttr):
        # This function should return:
        # IDYES Allows the operation.
        # IDNO Prevents the operation on this folder but continues with any other operations that have been approved (for example, a batch copy operation).
        # IDCANCEL Prevents the current operation and cancels any pending operations.
        print("CopyCallBack", hwnd, func, flags, srcName, srcAttr, destName, destAttr)
        return win32gui.MessageBox(
            hwnd, "Allow operation?", "CopyHook", win32con.MB_YESNO
        )


def DllRegisterServer():
    import winreg

    key = winreg.CreateKey(
        winreg.HKEY_CLASSES_ROOT,
        "directory\\shellex\\CopyHookHandlers\\" + ShellExtension._reg_desc_,
    )
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ShellExtension._reg_clsid_)
    key = winreg.CreateKey(
        winreg.HKEY_CLASSES_ROOT,
        "*\\shellex\\CopyHookHandlers\\" + ShellExtension._reg_desc_,
    )
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ShellExtension._reg_clsid_)
    print(ShellExtension._reg_desc_, "registration complete.")


def DllUnregisterServer():
    import winreg

    try:
        key = winreg.DeleteKey(
            winreg.HKEY_CLASSES_ROOT,
            "directory\\shellex\\CopyHookHandlers\\" + ShellExtension._reg_desc_,
        )
    except WindowsError as details:
        import errno

        if details.errno != errno.ENOENT:
            raise
    try:
        key = winreg.DeleteKey(
            winreg.HKEY_CLASSES_ROOT,
            "*\\shellex\\CopyHookHandlers\\" + ShellExtension._reg_desc_,
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
#!/usr/bin/env python
