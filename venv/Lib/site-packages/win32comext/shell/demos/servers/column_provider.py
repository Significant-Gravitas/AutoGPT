# A sample shell column provider
# Mainly ported from MSDN article:
#  Using Shell Column Handlers for Detailed File Information,
#  Raymond Chen, Microsoft Corporation, February 2000
#
# To demostrate:
# * Execute this script to register the namespace.
# * Open Windows Explorer
# * Right-click an explorer column header - select "More"
# * Locate column 'pyc size' or 'pyo size', and add it to the view.
# This handler is providing that column data.
import os
import stat

import commctrl
import pythoncom
from pywintypes import IID
from win32com.server.util import wrap
from win32com.shell import shell, shellcon

IPersist_Methods = ["GetClassID"]
IColumnProvider_Methods = IPersist_Methods + [
    "Initialize",
    "GetColumnInfo",
    "GetItemData",
]


class ColumnProvider:
    _reg_progid_ = "Python.ShellExtension.ColumnProvider"
    _reg_desc_ = "Python Sample Shell Extension (Column Provider)"
    _reg_clsid_ = IID("{0F14101A-E05E-4070-BD54-83DFA58C3D68}")
    _com_interfaces_ = [
        pythoncom.IID_IPersist,
        shell.IID_IColumnProvider,
    ]
    _public_methods_ = IColumnProvider_Methods

    # IPersist
    def GetClassID(self):
        return self._reg_clsid_

    # IColumnProvider
    def Initialize(self, colInit):
        flags, reserved, name = colInit
        print("ColumnProvider initializing for file", name)

    def GetColumnInfo(self, index):
        # We support exactly 2 columns - 'pyc size' and 'pyo size'
        if index in [0, 1]:
            # As per the MSDN sample, use our CLSID as the fmtid
            if index == 0:
                ext = ".pyc"
            else:
                ext = ".pyo"
            title = ext + " size"
            description = "Size of compiled %s file" % ext
            col_id = (self._reg_clsid_, index)  # fmtid  # pid
            col_info = (
                col_id,  # scid
                pythoncom.VT_I4,  # vt
                commctrl.LVCFMT_RIGHT,  # fmt
                20,  # cChars
                shellcon.SHCOLSTATE_TYPE_INT
                | shellcon.SHCOLSTATE_SECONDARYUI,  # csFlags
                title,
                description,
            )
            return col_info
        return None  # Indicate no more columns.

    def GetItemData(self, colid, colData):
        fmt_id, pid = colid
        fmt_id == self._reg_clsid_
        flags, attr, reserved, ext, name = colData
        if ext.lower() not in [".py", ".pyw"]:
            return None
        if pid == 0:
            ext = ".pyc"
        else:
            ext = ".pyo"
        check_file = os.path.splitext(name)[0] + ext
        try:
            st = os.stat(check_file)
            return st[stat.ST_SIZE]
        except OSError:
            # No file
            return None


def DllRegisterServer():
    import winreg

    # Special ColumnProvider key
    key = winreg.CreateKey(
        winreg.HKEY_CLASSES_ROOT,
        "Folder\\ShellEx\\ColumnHandlers\\" + str(ColumnProvider._reg_clsid_),
    )
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ColumnProvider._reg_desc_)
    print(ColumnProvider._reg_desc_, "registration complete.")


def DllUnregisterServer():
    import winreg

    try:
        key = winreg.DeleteKey(
            winreg.HKEY_CLASSES_ROOT,
            "Folder\\ShellEx\\ColumnHandlers\\" + str(ColumnProvider._reg_clsid_),
        )
    except WindowsError as details:
        import errno

        if details.errno != errno.ENOENT:
            raise
    print(ColumnProvider._reg_desc_, "unregistration complete.")


if __name__ == "__main__":
    from win32com.server import register

    register.UseCommandLine(
        ColumnProvider,
        finalize_register=DllRegisterServer,
        finalize_unregister=DllUnregisterServer,
    )
