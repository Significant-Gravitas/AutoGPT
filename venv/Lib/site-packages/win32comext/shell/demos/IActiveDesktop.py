import time

import pythoncom
from win32com.shell import shell, shellcon

website = "https://github.com/mhammond/pywin32/"
iad = pythoncom.CoCreateInstance(
    shell.CLSID_ActiveDesktop,
    None,
    pythoncom.CLSCTX_INPROC_SERVER,
    shell.IID_IActiveDesktop,
)
opts = iad.GetDesktopItemOptions()
if not (opts["ActiveDesktop"] and opts["EnableComponents"]):
    print("Warning: Enabling Active Desktop")
    opts["ActiveDesktop"] = True
    opts["EnableComponents"] = True
    iad.SetDesktopItemOptions(opts)
    iad.ApplyChanges(0xFFFF)
    iad = None
    ## apparently takes a short while for it to become active
    time.sleep(2)
    iad = pythoncom.CoCreateInstance(
        shell.CLSID_ActiveDesktop,
        None,
        pythoncom.CLSCTX_INPROC_SERVER,
        shell.IID_IActiveDesktop,
    )

cnt = iad.GetDesktopItemCount()
print("Count:", cnt)
for i in range(cnt):
    print(iad.GetDesktopItem(i))

component = {
    "ID": cnt + 1,
    "ComponentType": shellcon.COMP_TYPE_WEBSITE,
    "CurItemState": shellcon.IS_NORMAL,
    "SubscribedURL": website,
    "Source": website,
    "FriendlyName": "Pywin32 on SF",
    "Checked": True,  ## this controls whether item is currently displayed
    "NoScroll": False,
    "Dirty": False,
    "Pos": {
        "Top": 69,
        "Left": 69,
        "Height": 400,
        "Width": 400,
        "zIndex": 1002,
        "CanResize": True,
        "CanResizeX": True,
        "CanResizeY": True,
        "PreferredLeftPercent": 0,
        "PreferredTopPercent": 0,
    },
    "Original": {
        "Top": 33,
        "Left": 304,
        "Height": 362,
        "Width": 372,
        "ItemState": shellcon.IS_NORMAL,
    },
    "Restored": {
        "Top": 33,
        "Left": 304,
        "Height": 362,
        "Width": 372,
        "ItemState": shellcon.IS_NORMAL,
    },
}


try:
    existing_item = iad.GetDesktopItemBySource(website)
except pythoncom.com_error:
    pass
else:
    iad.RemoveDesktopItem(existing_item)
    iad.ApplyChanges(0xFFFF)

iad.AddDesktopItem(component)
iad.ApplyChanges(0xFFFF)  ## need to check which AD_APPLY constants are actually needed
