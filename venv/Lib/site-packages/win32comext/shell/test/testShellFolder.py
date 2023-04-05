from win32com.shell import shell
from win32com.shell.shellcon import *

sf = shell.SHGetDesktopFolder()
print("Shell Folder is", sf)

names = []
for i in sf:  # Magically calls EnumObjects
    name = sf.GetDisplayNameOf(i, SHGDN_NORMAL)
    names.append(name)

# And get the enumerator manually
enum = sf.EnumObjects(0, SHCONTF_FOLDERS | SHCONTF_NONFOLDERS | SHCONTF_INCLUDEHIDDEN)
num = 0
for i in enum:
    num += 1
if num != len(names):
    print("Should have got the same number of names!?")
print("Found", len(names), "items on the desktop")
for name in names:
    print(name)
