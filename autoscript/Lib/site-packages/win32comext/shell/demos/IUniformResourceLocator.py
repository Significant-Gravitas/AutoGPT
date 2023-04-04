import os

import pythoncom
import win32api
from win32com.shell import shell, shellcon


class InternetShortcut:
    def __init__(self):
        self._base = pythoncom.CoCreateInstance(
            shell.CLSID_InternetShortcut,
            None,
            pythoncom.CLSCTX_INPROC_SERVER,
            shell.IID_IUniformResourceLocator,
        )

    def load(self, filename):
        # Get an IPersist interface
        # which allows save/restore of object to/from files
        self._base.QueryInterface(pythoncom.IID_IPersistFile).Load(filename)

    def save(self, filename):
        self._base.QueryInterface(pythoncom.IID_IPersistFile).Save(filename, 1)

    def __getattr__(self, name):
        if name != "_base":
            return getattr(self._base, name)


temp_dir = win32api.GetTempPath()
linkname = win32api.GetTempFileName(temp_dir, "ish")[0]
print("Link:", linkname)
os.remove(linkname)
linkname += ".url"

ish = InternetShortcut()
ish.SetURL("https://github.com/mhammond/pywin32")
ish.save(linkname)

## IUniformResourceLocator also give access to IPropertySetStorage
pss = ish.QueryInterface(pythoncom.IID_IPropertySetStorage)
ps = pss.Open(shell.FMTID_InternetSite)
property_ids = [
    (k, v) for k, v in shellcon.__dict__.items() if k.startswith("PID_INTSITE_")
]
for pname, pval in property_ids:
    print(pname, ps.ReadMultiple((pval,))[0])

ps = pss.Open(shell.FMTID_Intshcut)
property_ids = [(k, v) for k, v in shellcon.__dict__.items() if k.startswith("PID_IS_")]
for pname, pval in property_ids:
    print(pname, ps.ReadMultiple((pval,))[0])

new_sh = InternetShortcut()
new_sh.load(linkname)
new_sh.InvokeCommand("Open")
