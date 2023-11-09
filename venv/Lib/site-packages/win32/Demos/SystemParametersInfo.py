import glob
import os
import time

import win32api
import win32con
import win32gui

## some of these tests will fail for systems prior to XP

for pname in (
    ## Set actions all take an unsigned int in pvParam
    "SPI_GETMOUSESPEED",
    "SPI_GETACTIVEWNDTRKTIMEOUT",
    "SPI_GETCARETWIDTH",
    "SPI_GETFOREGROUNDFLASHCOUNT",
    "SPI_GETFOREGROUNDLOCKTIMEOUT",
    ## Set actions all take an unsigned int in uiParam
    "SPI_GETWHEELSCROLLLINES",
    "SPI_GETKEYBOARDDELAY",
    "SPI_GETKEYBOARDSPEED",
    "SPI_GETMOUSEHOVERHEIGHT",
    "SPI_GETMOUSEHOVERWIDTH",
    "SPI_GETMOUSEHOVERTIME",
    "SPI_GETSCREENSAVETIMEOUT",
    "SPI_GETMENUSHOWDELAY",
    "SPI_GETLOWPOWERTIMEOUT",
    "SPI_GETPOWEROFFTIMEOUT",
    "SPI_GETBORDER",
    ## below are winxp only:
    "SPI_GETFONTSMOOTHINGCONTRAST",
    "SPI_GETFONTSMOOTHINGTYPE",
    "SPI_GETFOCUSBORDERHEIGHT",
    "SPI_GETFOCUSBORDERWIDTH",
    "SPI_GETMOUSECLICKLOCKTIME",
):
    print(pname)
    cget = getattr(win32con, pname)
    cset = getattr(win32con, pname.replace("_GET", "_SET"))
    orig_value = win32gui.SystemParametersInfo(cget)
    print("\toriginal setting:", orig_value)
    win32gui.SystemParametersInfo(cset, orig_value + 1)
    new_value = win32gui.SystemParametersInfo(cget)
    print("\tnew value:", new_value)
    # On Vista, some of these values seem to be ignored.  So only "fail" if
    # the new value isn't what we set or the original
    if new_value != orig_value + 1:
        assert new_value == orig_value
        print("Strange - setting %s seems to have been ignored" % (pname,))
    win32gui.SystemParametersInfo(cset, orig_value)
    assert win32gui.SystemParametersInfo(cget) == orig_value


# these take a boolean value in pvParam
# change to opposite, check that it was changed and change back
for pname in (
    "SPI_GETFLATMENU",
    "SPI_GETDROPSHADOW",
    "SPI_GETKEYBOARDCUES",
    "SPI_GETMENUFADE",
    "SPI_GETCOMBOBOXANIMATION",
    "SPI_GETCURSORSHADOW",
    "SPI_GETGRADIENTCAPTIONS",
    "SPI_GETHOTTRACKING",
    "SPI_GETLISTBOXSMOOTHSCROLLING",
    "SPI_GETMENUANIMATION",
    "SPI_GETSELECTIONFADE",
    "SPI_GETTOOLTIPANIMATION",
    "SPI_GETTOOLTIPFADE",
    "SPI_GETUIEFFECTS",
    "SPI_GETACTIVEWINDOWTRACKING",
    "SPI_GETACTIVEWNDTRKZORDER",
):
    print(pname)
    cget = getattr(win32con, pname)
    cset = getattr(win32con, pname.replace("_GET", "_SET"))
    orig_value = win32gui.SystemParametersInfo(cget)
    print(orig_value)
    win32gui.SystemParametersInfo(cset, not orig_value)
    new_value = win32gui.SystemParametersInfo(cget)
    print(new_value)
    assert orig_value != new_value
    win32gui.SystemParametersInfo(cset, orig_value)
    assert win32gui.SystemParametersInfo(cget) == orig_value


# these take a boolean in uiParam
#  could combine with above section now that SystemParametersInfo only takes a single parameter
for pname in (
    "SPI_GETFONTSMOOTHING",
    "SPI_GETICONTITLEWRAP",
    "SPI_GETBEEP",
    "SPI_GETBLOCKSENDINPUTRESETS",
    "SPI_GETKEYBOARDPREF",
    "SPI_GETSCREENSAVEACTIVE",
    "SPI_GETMENUDROPALIGNMENT",
    "SPI_GETDRAGFULLWINDOWS",
    "SPI_GETSHOWIMEUI",
):
    cget = getattr(win32con, pname)
    cset = getattr(win32con, pname.replace("_GET", "_SET"))
    orig_value = win32gui.SystemParametersInfo(cget)
    win32gui.SystemParametersInfo(cset, not orig_value)
    new_value = win32gui.SystemParametersInfo(cget)
    # Some of these also can't be changed (eg, SPI_GETSCREENSAVEACTIVE) so
    # don't actually get upset.
    if orig_value != new_value:
        print("successfully toggled", pname, "from", orig_value, "to", new_value)
    else:
        print("couldn't toggle", pname, "from", orig_value)
    win32gui.SystemParametersInfo(cset, orig_value)
    assert win32gui.SystemParametersInfo(cget) == orig_value


print("SPI_GETICONTITLELOGFONT")
lf = win32gui.SystemParametersInfo(win32con.SPI_GETICONTITLELOGFONT)
orig_height = lf.lfHeight
orig_italic = lf.lfItalic
print("Height:", orig_height, "Italic:", orig_italic)
lf.lfHeight += 2
lf.lfItalic = not lf.lfItalic
win32gui.SystemParametersInfo(win32con.SPI_SETICONTITLELOGFONT, lf)
new_lf = win32gui.SystemParametersInfo(win32con.SPI_GETICONTITLELOGFONT)
print("New Height:", new_lf.lfHeight, "New Italic:", new_lf.lfItalic)
assert new_lf.lfHeight == orig_height + 2
assert new_lf.lfItalic != orig_italic

lf.lfHeight = orig_height
lf.lfItalic = orig_italic
win32gui.SystemParametersInfo(win32con.SPI_SETICONTITLELOGFONT, lf)
new_lf = win32gui.SystemParametersInfo(win32con.SPI_GETICONTITLELOGFONT)
assert new_lf.lfHeight == orig_height
assert new_lf.lfItalic == orig_italic


print("SPI_GETMOUSEHOVERWIDTH, SPI_GETMOUSEHOVERHEIGHT, SPI_GETMOUSEHOVERTIME")
w = win32gui.SystemParametersInfo(win32con.SPI_GETMOUSEHOVERWIDTH)
h = win32gui.SystemParametersInfo(win32con.SPI_GETMOUSEHOVERHEIGHT)
t = win32gui.SystemParametersInfo(win32con.SPI_GETMOUSEHOVERTIME)
print("w,h,t:", w, h, t)

win32gui.SystemParametersInfo(win32con.SPI_SETMOUSEHOVERWIDTH, w + 1)
win32gui.SystemParametersInfo(win32con.SPI_SETMOUSEHOVERHEIGHT, h + 2)
win32gui.SystemParametersInfo(win32con.SPI_SETMOUSEHOVERTIME, t + 3)
new_w = win32gui.SystemParametersInfo(win32con.SPI_GETMOUSEHOVERWIDTH)
new_h = win32gui.SystemParametersInfo(win32con.SPI_GETMOUSEHOVERHEIGHT)
new_t = win32gui.SystemParametersInfo(win32con.SPI_GETMOUSEHOVERTIME)
print("new w,h,t:", new_w, new_h, new_t)
assert new_w == w + 1
assert new_h == h + 2
assert new_t == t + 3

win32gui.SystemParametersInfo(win32con.SPI_SETMOUSEHOVERWIDTH, w)
win32gui.SystemParametersInfo(win32con.SPI_SETMOUSEHOVERHEIGHT, h)
win32gui.SystemParametersInfo(win32con.SPI_SETMOUSEHOVERTIME, t)
new_w = win32gui.SystemParametersInfo(win32con.SPI_GETMOUSEHOVERWIDTH)
new_h = win32gui.SystemParametersInfo(win32con.SPI_GETMOUSEHOVERHEIGHT)
new_t = win32gui.SystemParametersInfo(win32con.SPI_GETMOUSEHOVERTIME)
assert new_w == w
assert new_h == h
assert new_t == t


print("SPI_SETDOUBLECLKWIDTH, SPI_SETDOUBLECLKHEIGHT")
x = win32api.GetSystemMetrics(win32con.SM_CXDOUBLECLK)
y = win32api.GetSystemMetrics(win32con.SM_CYDOUBLECLK)
print("x,y:", x, y)
win32gui.SystemParametersInfo(win32con.SPI_SETDOUBLECLKWIDTH, x + 1)
win32gui.SystemParametersInfo(win32con.SPI_SETDOUBLECLKHEIGHT, y + 2)
new_x = win32api.GetSystemMetrics(win32con.SM_CXDOUBLECLK)
new_y = win32api.GetSystemMetrics(win32con.SM_CYDOUBLECLK)
print("new x,y:", new_x, new_y)
assert new_x == x + 1
assert new_y == y + 2
win32gui.SystemParametersInfo(win32con.SPI_SETDOUBLECLKWIDTH, x)
win32gui.SystemParametersInfo(win32con.SPI_SETDOUBLECLKHEIGHT, y)
new_x = win32api.GetSystemMetrics(win32con.SM_CXDOUBLECLK)
new_y = win32api.GetSystemMetrics(win32con.SM_CYDOUBLECLK)
assert new_x == x
assert new_y == y


print("SPI_SETDRAGWIDTH, SPI_SETDRAGHEIGHT")
dw = win32api.GetSystemMetrics(win32con.SM_CXDRAG)
dh = win32api.GetSystemMetrics(win32con.SM_CYDRAG)
print("dw,dh:", dw, dh)
win32gui.SystemParametersInfo(win32con.SPI_SETDRAGWIDTH, dw + 1)
win32gui.SystemParametersInfo(win32con.SPI_SETDRAGHEIGHT, dh + 2)
new_dw = win32api.GetSystemMetrics(win32con.SM_CXDRAG)
new_dh = win32api.GetSystemMetrics(win32con.SM_CYDRAG)
print("new dw,dh:", new_dw, new_dh)
assert new_dw == dw + 1
assert new_dh == dh + 2
win32gui.SystemParametersInfo(win32con.SPI_SETDRAGWIDTH, dw)
win32gui.SystemParametersInfo(win32con.SPI_SETDRAGHEIGHT, dh)
new_dw = win32api.GetSystemMetrics(win32con.SM_CXDRAG)
new_dh = win32api.GetSystemMetrics(win32con.SM_CYDRAG)
assert new_dw == dw
assert new_dh == dh


orig_wallpaper = win32gui.SystemParametersInfo(Action=win32con.SPI_GETDESKWALLPAPER)
print("Original: ", orig_wallpaper)
for bmp in glob.glob(os.path.join(os.environ["windir"], "*.bmp")):
    print(bmp)
    win32gui.SystemParametersInfo(win32con.SPI_SETDESKWALLPAPER, Param=bmp)
    print(win32gui.SystemParametersInfo(Action=win32con.SPI_GETDESKWALLPAPER))
    time.sleep(1)

win32gui.SystemParametersInfo(win32con.SPI_SETDESKWALLPAPER, Param=orig_wallpaper)
