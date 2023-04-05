import os
import sys

import pythoncom
import win32api
from win32com.shell import shell, shellcon

temp_dir = win32api.GetTempPath()
linkname = win32api.GetTempFileName(temp_dir, "cmd")[0]
os.remove(linkname)
linkname += ".lnk"
print("Link name:", linkname)
ish = pythoncom.CoCreateInstance(
    shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink
)
ish.SetPath(os.environ["cOMSPEC"])
ish.SetWorkingDirectory(os.path.split(sys.executable)[0])
ish.SetDescription("shortcut made by python")

console_props = {
    "Signature": shellcon.NT_CONSOLE_PROPS_SIG,
    "InsertMode": True,
    "FullScreen": False,  ## True looks like "DOS Mode" from win98!
    "FontFamily": 54,
    "CursorSize": 75,  ## pct of character size
    "ScreenBufferSize": (152, 256),
    "AutoPosition": False,
    "FontSize": (4, 5),
    "FaceName": "",
    "HistoryBufferSize": 32,
    "InputBufferSize": 0,
    "QuickEdit": True,
    "Font": 0,  ## 0 should always be present, use win32console.GetNumberOfConsoleFonts() to find how many available
    "FillAttribute": 7,
    "PopupFillAttribute": 245,
    "WindowSize": (128, 32),
    "WindowOrigin": (0, 0),
    "FontWeight": 400,
    "HistoryNoDup": False,
    "NumberOfHistoryBuffers": 32,
    ## ColorTable copied from a 'normal' console shortcut, with some obvious changes
    ## These do not appear to be documented.  From experimentation, [0] is background, [7] is foreground text
    "ColorTable": (
        255,
        8388608,
        32768,
        8421376,
        128,
        8388736,
        32896,
        12582912,
        8421504,
        16711680,
        65280,
        16776960,
        255,
        16711935,
        65535,
        16777215,
    ),
}

ishdl = ish.QueryInterface(shell.IID_IShellLinkDataList)
ishdl.AddDataBlock(console_props)
ipf = ish.QueryInterface(pythoncom.IID_IPersistFile)
ipf.Save(linkname, 1)
os.startfile(linkname)
