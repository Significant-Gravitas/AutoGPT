import os

import win32api

ver_strings = (
    "Comments",
    "InternalName",
    "ProductName",
    "CompanyName",
    "LegalCopyright",
    "ProductVersion",
    "FileDescription",
    "LegalTrademarks",
    "PrivateBuild",
    "FileVersion",
    "OriginalFilename",
    "SpecialBuild",
)
fname = os.environ["comspec"]
d = win32api.GetFileVersionInfo(fname, "\\")
## backslash as parm returns dictionary of numeric info corresponding to VS_FIXEDFILEINFO struc
for n, v in d.items():
    print(n, v)

pairs = win32api.GetFileVersionInfo(fname, "\\VarFileInfo\\Translation")
## \VarFileInfo\Translation returns list of available (language, codepage) pairs that can be used to retreive string info
## any other must be of the form \StringfileInfo\%04X%04X\parm_name, middle two are language/codepage pair returned from above
for lang, codepage in pairs:
    print("lang: ", lang, "codepage:", codepage)
    for ver_string in ver_strings:
        str_info = "\\StringFileInfo\\%04X%04X\\%s" % (lang, codepage, ver_string)
        ## print str_info
        print(ver_string, repr(win32api.GetFileVersionInfo(fname, str_info)))
