# Demonstrates some advanced menu concepts using win32gui.
# This creates a taskbar icon which has some fancy menus (but note that
# selecting the menu items does nothing useful - see win32gui_taskbar.py
# for examples of this.

# NOTE: This is a work in progress.  Todo:
# * The "Checked" menu items don't work correctly - I'm not sure why.
# * No support for GetMenuItemInfo.

# Based on Andy McKay's demo code.
from win32api import *

# Try and use XP features, so we get alpha-blending etc.
try:
    from winxpgui import *
except ImportError:
    from win32gui import *

import array
import os
import struct
import sys

import win32con
from win32gui_struct import *

this_dir = os.path.split(sys.argv[0])[0]


class MainWindow:
    def __init__(self):
        message_map = {
            win32con.WM_DESTROY: self.OnDestroy,
            win32con.WM_COMMAND: self.OnCommand,
            win32con.WM_USER + 20: self.OnTaskbarNotify,
            # owner-draw related handlers.
            win32con.WM_MEASUREITEM: self.OnMeasureItem,
            win32con.WM_DRAWITEM: self.OnDrawItem,
        }
        # Register the Window class.
        wc = WNDCLASS()
        hinst = wc.hInstance = GetModuleHandle(None)
        wc.lpszClassName = "PythonTaskbarDemo"
        wc.lpfnWndProc = message_map  # could also specify a wndproc.
        classAtom = RegisterClass(wc)
        # Create the Window.
        style = win32con.WS_OVERLAPPED | win32con.WS_SYSMENU
        self.hwnd = CreateWindow(
            classAtom,
            "Taskbar Demo",
            style,
            0,
            0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            hinst,
            None,
        )
        UpdateWindow(self.hwnd)
        iconPathName = os.path.abspath(os.path.join(sys.prefix, "pyc.ico"))
        # py2.5 includes the .ico files in the DLLs dir for some reason.
        if not os.path.isfile(iconPathName):
            iconPathName = os.path.abspath(
                os.path.join(os.path.split(sys.executable)[0], "DLLs", "pyc.ico")
            )
        if not os.path.isfile(iconPathName):
            # Look in the source tree.
            iconPathName = os.path.abspath(
                os.path.join(os.path.split(sys.executable)[0], "..\\PC\\pyc.ico")
            )
        if os.path.isfile(iconPathName):
            icon_flags = win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
            hicon = LoadImage(
                hinst, iconPathName, win32con.IMAGE_ICON, 0, 0, icon_flags
            )
        else:
            iconPathName = None
            print("Can't find a Python icon file - using default")
            hicon = LoadIcon(0, win32con.IDI_APPLICATION)
        self.iconPathName = iconPathName

        # Load up some information about menus needed by our owner-draw code.
        # The font to use on the menu.
        ncm = SystemParametersInfo(win32con.SPI_GETNONCLIENTMETRICS)
        self.font_menu = CreateFontIndirect(ncm["lfMenuFont"])
        # spacing for our ownerdraw menus - not sure exactly what constants
        # should be used (and if you owner-draw all items on the menu, it
        # doesn't matter!)
        self.menu_icon_height = GetSystemMetrics(win32con.SM_CYMENU) - 4
        self.menu_icon_width = self.menu_icon_height
        self.icon_x_pad = 8  # space from end of icon to start of text.
        # A map we use to stash away data we need for ownerdraw.  Keyed
        # by integer ID - that ID will be set in dwTypeData of the menu item.
        self.menu_item_map = {}

        # Finally, create the menu
        self.createMenu()

        flags = NIF_ICON | NIF_MESSAGE | NIF_TIP
        nid = (self.hwnd, 0, flags, win32con.WM_USER + 20, hicon, "Python Demo")
        Shell_NotifyIcon(NIM_ADD, nid)
        print("Please right-click on the Python icon in the taskbar")

    def createMenu(self):
        self.hmenu = menu = CreatePopupMenu()
        # Create our 'Exit' item with the standard, ugly 'close' icon.
        item, extras = PackMENUITEMINFO(
            text="Exit", hbmpItem=win32con.HBMMENU_MBAR_CLOSE, wID=1000
        )
        InsertMenuItem(menu, 0, 1, item)
        # Create a 'text only' menu via InsertMenuItem rather then
        # AppendMenu, just to prove we can!
        item, extras = PackMENUITEMINFO(text="Text only item", wID=1001)
        InsertMenuItem(menu, 0, 1, item)

        load_bmp_flags = win32con.LR_LOADFROMFILE | win32con.LR_LOADTRANSPARENT
        # These images are "over sized", so we load them scaled.
        hbmp = LoadImage(
            0,
            os.path.join(this_dir, "images/smiley.bmp"),
            win32con.IMAGE_BITMAP,
            20,
            20,
            load_bmp_flags,
        )

        # Create a top-level menu with a bitmap
        item, extras = PackMENUITEMINFO(
            text="Menu with bitmap", hbmpItem=hbmp, wID=1002
        )
        InsertMenuItem(menu, 0, 1, item)

        # Owner-draw menus mainly from:
        # http://windowssdk.msdn.microsoft.com/en-us/library/ms647558.aspx
        # and:
        # http://www.codeguru.com/cpp/controls/menu/bitmappedmenus/article.php/c165

        # Create one with an icon - this is *lots* more work - we do it
        # owner-draw!  The primary reason is to handle transparency better -
        # converting to a bitmap causes the background to be incorrect when
        # the menu item is selected.  I can't see a simpler way.
        # First, load the icon we want to use.
        ico_x = GetSystemMetrics(win32con.SM_CXSMICON)
        ico_y = GetSystemMetrics(win32con.SM_CYSMICON)
        if self.iconPathName:
            hicon = LoadImage(
                0,
                self.iconPathName,
                win32con.IMAGE_ICON,
                ico_x,
                ico_y,
                win32con.LR_LOADFROMFILE,
            )
        else:
            shell_dll = os.path.join(GetSystemDirectory(), "shell32.dll")
            large, small = win32gui.ExtractIconEx(shell_dll, 4, 1)
            hicon = small[0]
            DestroyIcon(large[0])

        # Stash away the text and hicon in our map, and add the owner-draw
        # item to the menu.
        index = 0
        self.menu_item_map[index] = (hicon, "Menu with owner-draw icon")
        item, extras = PackMENUITEMINFO(
            fType=win32con.MFT_OWNERDRAW, dwItemData=index, wID=1009
        )
        InsertMenuItem(menu, 0, 1, item)

        # Add another icon-based icon - but this time using HBMMENU_CALLBACK
        # in the hbmpItem elt, so we only need to draw the icon (ie, not the
        # text or checkmark)
        index = 1
        self.menu_item_map[index] = (hicon, None)
        item, extras = PackMENUITEMINFO(
            text="Menu with o-d icon 2",
            dwItemData=index,
            hbmpItem=win32con.HBMMENU_CALLBACK,
            wID=1010,
        )
        InsertMenuItem(menu, 0, 1, item)

        # Add another icon-based icon - this time by converting
        # via bitmap.  Note the icon background when selected is ugly :(
        hdcBitmap = CreateCompatibleDC(0)
        hdcScreen = GetDC(0)
        hbm = CreateCompatibleBitmap(hdcScreen, ico_x, ico_y)
        hbmOld = SelectObject(hdcBitmap, hbm)
        SetBkMode(hdcBitmap, win32con.TRANSPARENT)
        # Fill the background.
        brush = GetSysColorBrush(win32con.COLOR_MENU)
        FillRect(hdcBitmap, (0, 0, 16, 16), brush)
        # unclear if brush needs to be freed.  Best clue I can find is:
        # "GetSysColorBrush returns a cached brush instead of allocating a new
        # one." - implies no DeleteObject.
        # draw the icon
        DrawIconEx(hdcBitmap, 0, 0, hicon, ico_x, ico_y, 0, 0, win32con.DI_NORMAL)
        SelectObject(hdcBitmap, hbmOld)
        DeleteDC(hdcBitmap)
        item, extras = PackMENUITEMINFO(
            text="Menu with icon", hbmpItem=hbm.Detach(), wID=1011
        )
        InsertMenuItem(menu, 0, 1, item)

        # Create a sub-menu, and put a few funky ones there.
        self.sub_menu = sub_menu = CreatePopupMenu()
        # A 'checkbox' menu.
        item, extras = PackMENUITEMINFO(
            fState=win32con.MFS_CHECKED, text="Checkbox menu", hbmpItem=hbmp, wID=1003
        )
        InsertMenuItem(sub_menu, 0, 1, item)
        # A 'radio' menu.
        InsertMenu(sub_menu, 0, win32con.MF_BYPOSITION, win32con.MF_SEPARATOR, None)
        item, extras = PackMENUITEMINFO(
            fType=win32con.MFT_RADIOCHECK,
            fState=win32con.MFS_CHECKED,
            text="Checkbox menu - bullet 1",
            hbmpItem=hbmp,
            wID=1004,
        )
        InsertMenuItem(sub_menu, 0, 1, item)
        item, extras = PackMENUITEMINFO(
            fType=win32con.MFT_RADIOCHECK,
            fState=win32con.MFS_UNCHECKED,
            text="Checkbox menu - bullet 2",
            hbmpItem=hbmp,
            wID=1005,
        )
        InsertMenuItem(sub_menu, 0, 1, item)
        # And add the sub-menu to the top-level menu.
        item, extras = PackMENUITEMINFO(text="Sub-Menu", hSubMenu=sub_menu)
        InsertMenuItem(menu, 0, 1, item)

        # Set 'Exit' as the default option.
        SetMenuDefaultItem(menu, 1000, 0)

    def OnDestroy(self, hwnd, msg, wparam, lparam):
        nid = (self.hwnd, 0)
        Shell_NotifyIcon(NIM_DELETE, nid)
        PostQuitMessage(0)  # Terminate the app.

    def OnTaskbarNotify(self, hwnd, msg, wparam, lparam):
        if lparam == win32con.WM_RBUTTONUP:
            print("You right clicked me.")
            # display the menu at the cursor pos.
            pos = GetCursorPos()
            SetForegroundWindow(self.hwnd)
            TrackPopupMenu(
                self.hmenu, win32con.TPM_LEFTALIGN, pos[0], pos[1], 0, self.hwnd, None
            )
            PostMessage(self.hwnd, win32con.WM_NULL, 0, 0)
        elif lparam == win32con.WM_LBUTTONDBLCLK:
            print("You double-clicked me")
            # find the default menu item and fire it.
            cmd = GetMenuDefaultItem(self.hmenu, False, 0)
            if cmd == -1:
                print("Can't find a default!")
            # and just pretend it came from the menu
            self.OnCommand(hwnd, win32con.WM_COMMAND, cmd, 0)
        return 1

    def OnCommand(self, hwnd, msg, wparam, lparam):
        id = LOWORD(wparam)
        if id == 1000:
            print("Goodbye")
            DestroyWindow(self.hwnd)
        elif id in (1003, 1004, 1005):
            # Our 'checkbox' and 'radio' items
            state = GetMenuState(self.sub_menu, id, win32con.MF_BYCOMMAND)
            if state == -1:
                raise RuntimeError("No item found")
            if state & win32con.MF_CHECKED:
                check_flags = win32con.MF_UNCHECKED
                print("Menu was checked - unchecking")
            else:
                check_flags = win32con.MF_CHECKED
                print("Menu was unchecked - checking")

            if id == 1003:
                # simple checkbox
                rc = CheckMenuItem(
                    self.sub_menu, id, win32con.MF_BYCOMMAND | check_flags
                )
            else:
                # radio button - must pass the first and last IDs in the
                # "group", and the ID in the group that is to be selected.
                rc = CheckMenuRadioItem(
                    self.sub_menu, 1004, 1005, id, win32con.MF_BYCOMMAND
                )
            # Get and check the new state - first the simple way...
            new_state = GetMenuState(self.sub_menu, id, win32con.MF_BYCOMMAND)
            if new_state & win32con.MF_CHECKED != check_flags:
                raise RuntimeError("The new item didn't get the new checked state!")
            # Now the long-winded way via GetMenuItemInfo...
            buf, extras = EmptyMENUITEMINFO()
            win32gui.GetMenuItemInfo(self.sub_menu, id, False, buf)
            (
                fType,
                fState,
                wID,
                hSubMenu,
                hbmpChecked,
                hbmpUnchecked,
                dwItemData,
                text,
                hbmpItem,
            ) = UnpackMENUITEMINFO(buf)

            if fState & win32con.MF_CHECKED != check_flags:
                raise RuntimeError("The new item didn't get the new checked state!")
        else:
            print("OnCommand for ID", id)

    # Owner-draw related functions.  We only have 1 owner-draw item, but
    # we pretend we have more than that :)
    def OnMeasureItem(self, hwnd, msg, wparam, lparam):
        ## Last item of MEASUREITEMSTRUCT is a ULONG_PTR
        fmt = "5iP"
        buf = PyMakeBuffer(struct.calcsize(fmt), lparam)
        data = struct.unpack(fmt, buf)
        ctlType, ctlID, itemID, itemWidth, itemHeight, itemData = data

        hicon, text = self.menu_item_map[itemData]
        if text is None:
            # Only drawing icon due to HBMMENU_CALLBACK
            cx = self.menu_icon_width
            cy = self.menu_icon_height
        else:
            # drawing the lot!
            dc = GetDC(hwnd)
            oldFont = SelectObject(dc, self.font_menu)
            cx, cy = GetTextExtentPoint32(dc, text)
            SelectObject(dc, oldFont)
            ReleaseDC(hwnd, dc)

            cx += GetSystemMetrics(win32con.SM_CXMENUCHECK)
            cx += self.menu_icon_width + self.icon_x_pad

            cy = GetSystemMetrics(win32con.SM_CYMENU)

        new_data = struct.pack(fmt, ctlType, ctlID, itemID, cx, cy, itemData)
        PySetMemory(lparam, new_data)
        return True

    def OnDrawItem(self, hwnd, msg, wparam, lparam):
        ## lparam is a DRAWITEMSTRUCT
        fmt = "5i2P4iP"
        data = struct.unpack(fmt, PyGetMemory(lparam, struct.calcsize(fmt)))
        (
            ctlType,
            ctlID,
            itemID,
            itemAction,
            itemState,
            hwndItem,
            hDC,
            left,
            top,
            right,
            bot,
            itemData,
        ) = data

        rect = left, top, right, bot
        hicon, text = self.menu_item_map[itemData]

        if text is None:
            # This means the menu-item had HBMMENU_CALLBACK - so all we
            # draw is the icon.  rect is the entire area we should use.
            DrawIconEx(
                hDC, left, top, hicon, right - left, bot - top, 0, 0, win32con.DI_NORMAL
            )
        else:
            # If the user has selected the item, use the selected
            # text and background colors to display the item.
            selected = itemState & win32con.ODS_SELECTED
            if selected:
                crText = SetTextColor(hDC, GetSysColor(win32con.COLOR_HIGHLIGHTTEXT))
                crBkgnd = SetBkColor(hDC, GetSysColor(win32con.COLOR_HIGHLIGHT))

            each_pad = self.icon_x_pad // 2
            x_icon = left + GetSystemMetrics(win32con.SM_CXMENUCHECK) + each_pad
            x_text = x_icon + self.menu_icon_width + each_pad

            # Draw text first, specifying a complete rect to fill - this sets
            # up the background (but overwrites anything else already there!)
            # Select the font, draw it, and restore the previous font.
            hfontOld = SelectObject(hDC, self.font_menu)
            ExtTextOut(hDC, x_text, top + 2, win32con.ETO_OPAQUE, rect, text)
            SelectObject(hDC, hfontOld)

            # Icon image next.  Icons are transparent - no need to handle
            # selection specially.
            DrawIconEx(
                hDC,
                x_icon,
                top + 2,
                hicon,
                self.menu_icon_width,
                self.menu_icon_height,
                0,
                0,
                win32con.DI_NORMAL,
            )

            # Return the text and background colors to their
            # normal state (not selected).
            if selected:
                SetTextColor(hDC, crText)
                SetBkColor(hDC, crBkgnd)


def main():
    w = MainWindow()
    PumpMessages()


if __name__ == "__main__":
    main()
