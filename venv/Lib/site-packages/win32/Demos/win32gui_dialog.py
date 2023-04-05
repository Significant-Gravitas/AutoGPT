# A demo of a fairly complex dialog.
#
# Features:
# * Uses a "dynamic dialog resource" to build the dialog.
# * Uses a ListView control.
# * Dynamically resizes content.
# * Uses a second worker thread to fill the list.
# * Demostrates support for windows XP themes.

# If you are on Windows XP, and specify a '--noxp' argument, you will see:
# * alpha-blend issues with icons
# * The buttons are "old" style, rather than based on the XP theme.
# Hence, using:
#   import winxpgui as win32gui
# is recommended.
# Please report any problems.
import sys

if "--noxp" in sys.argv:
    import win32gui
else:
    import winxpgui as win32gui

import array
import os
import queue
import struct

import commctrl
import win32api
import win32con
import win32gui_struct
import winerror

IDC_SEARCHTEXT = 1024
IDC_BUTTON_SEARCH = 1025
IDC_BUTTON_DISPLAY = 1026
IDC_LISTBOX = 1027

WM_SEARCH_RESULT = win32con.WM_USER + 512
WM_SEARCH_FINISHED = win32con.WM_USER + 513


class _WIN32MASKEDSTRUCT:
    def __init__(self, **kw):
        full_fmt = ""
        for name, fmt, default, mask in self._struct_items_:
            self.__dict__[name] = None
            if fmt == "z":
                full_fmt += "pi"
            else:
                full_fmt += fmt
        for name, val in kw.items():
            if name not in self.__dict__:
                raise ValueError("LVITEM structures do not have an item '%s'" % (name,))
            self.__dict__[name] = val

    def __setattr__(self, attr, val):
        if not attr.startswith("_") and attr not in self.__dict__:
            raise AttributeError(attr)
        self.__dict__[attr] = val

    def toparam(self):
        self._buffs = []
        full_fmt = ""
        vals = []
        mask = 0
        # calc the mask
        for name, fmt, default, this_mask in self._struct_items_:
            if this_mask is not None and self.__dict__.get(name) is not None:
                mask |= this_mask
        self.mask = mask
        for name, fmt, default, this_mask in self._struct_items_:
            val = self.__dict__[name]
            if fmt == "z":
                fmt = "Pi"
                if val is None:
                    vals.append(0)
                    vals.append(0)
                else:
                    # Note this demo still works with byte strings.  An
                    # alternate strategy would be to use unicode natively
                    # and use the 'W' version of the messages - eg,
                    # LVM_SETITEMW etc.
                    val = val + "\0"
                    if isinstance(val, str):
                        val = val.encode("mbcs")
                    str_buf = array.array("b", val)
                    vals.append(str_buf.buffer_info()[0])
                    vals.append(len(val))
                    self._buffs.append(str_buf)  # keep alive during the call.
            else:
                if val is None:
                    val = default
                vals.append(val)
            full_fmt += fmt
        return struct.pack(*(full_fmt,) + tuple(vals))


# NOTE: See the win32gui_struct module for an alternative way of dealing
# with these structures
class LVITEM(_WIN32MASKEDSTRUCT):
    _struct_items_ = [
        ("mask", "I", 0, None),
        ("iItem", "i", 0, None),
        ("iSubItem", "i", 0, None),
        ("state", "I", 0, commctrl.LVIF_STATE),
        ("stateMask", "I", 0, None),
        ("text", "z", None, commctrl.LVIF_TEXT),
        ("iImage", "i", 0, commctrl.LVIF_IMAGE),
        ("lParam", "i", 0, commctrl.LVIF_PARAM),
        ("iIdent", "i", 0, None),
    ]


class LVCOLUMN(_WIN32MASKEDSTRUCT):
    _struct_items_ = [
        ("mask", "I", 0, None),
        ("fmt", "i", 0, commctrl.LVCF_FMT),
        ("cx", "i", 0, commctrl.LVCF_WIDTH),
        ("text", "z", None, commctrl.LVCF_TEXT),
        ("iSubItem", "i", 0, commctrl.LVCF_SUBITEM),
        ("iImage", "i", 0, commctrl.LVCF_IMAGE),
        ("iOrder", "i", 0, commctrl.LVCF_ORDER),
    ]


class DemoWindowBase:
    def __init__(self):
        win32gui.InitCommonControls()
        self.hinst = win32gui.dllhandle
        self.list_data = {}

    def _RegisterWndClass(self):
        className = "PythonDocSearch"
        message_map = {}
        wc = win32gui.WNDCLASS()
        wc.SetDialogProc()  # Make it a dialog class.
        wc.hInstance = self.hinst
        wc.lpszClassName = className
        wc.style = win32con.CS_VREDRAW | win32con.CS_HREDRAW
        wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        wc.hbrBackground = win32con.COLOR_WINDOW + 1
        wc.lpfnWndProc = message_map  # could also specify a wndproc.
        # C code: wc.cbWndExtra = DLGWINDOWEXTRA + sizeof(HBRUSH) + (sizeof(COLORREF));
        wc.cbWndExtra = win32con.DLGWINDOWEXTRA + struct.calcsize("Pi")
        icon_flags = win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE

        ## py.ico went away in python 2.5, load from executable instead
        this_app = win32api.GetModuleHandle(None)
        try:
            wc.hIcon = win32gui.LoadIcon(this_app, 1)  ## python.exe and pythonw.exe
        except win32gui.error:
            wc.hIcon = win32gui.LoadIcon(this_app, 135)  ## pythonwin's icon
        try:
            classAtom = win32gui.RegisterClass(wc)
        except win32gui.error as err_info:
            if err_info.winerror != winerror.ERROR_CLASS_ALREADY_EXISTS:
                raise
        return className

    def _GetDialogTemplate(self, dlgClassName):
        style = (
            win32con.WS_THICKFRAME
            | win32con.WS_POPUP
            | win32con.WS_VISIBLE
            | win32con.WS_CAPTION
            | win32con.WS_SYSMENU
            | win32con.DS_SETFONT
            | win32con.WS_MINIMIZEBOX
        )
        cs = win32con.WS_CHILD | win32con.WS_VISIBLE
        title = "Dynamic Dialog Demo"

        # Window frame and title
        dlg = [
            [
                title,
                (0, 0, 210, 250),
                style,
                None,
                (8, "MS Sans Serif"),
                None,
                dlgClassName,
            ],
        ]

        # ID label and text box
        dlg.append([130, "Enter something", -1, (5, 5, 200, 9), cs | win32con.SS_LEFT])
        s = cs | win32con.WS_TABSTOP | win32con.WS_BORDER
        dlg.append(["EDIT", None, IDC_SEARCHTEXT, (5, 15, 200, 12), s])

        # Search/Display Buttons
        # (x positions don't matter here)
        s = cs | win32con.WS_TABSTOP
        dlg.append(
            [
                128,
                "Fill List",
                IDC_BUTTON_SEARCH,
                (5, 35, 50, 14),
                s | win32con.BS_DEFPUSHBUTTON,
            ]
        )
        s = win32con.BS_PUSHBUTTON | s
        dlg.append([128, "Display", IDC_BUTTON_DISPLAY, (100, 35, 50, 14), s])

        # List control.
        # Can't make this work :(
        ##        s = cs | win32con.WS_TABSTOP
        ##        dlg.append(['SysListView32', "Title", IDC_LISTBOX, (5, 505, 200, 200), s])

        return dlg

    def _DoCreate(self, fn):
        message_map = {
            win32con.WM_SIZE: self.OnSize,
            win32con.WM_COMMAND: self.OnCommand,
            win32con.WM_NOTIFY: self.OnNotify,
            win32con.WM_INITDIALOG: self.OnInitDialog,
            win32con.WM_CLOSE: self.OnClose,
            win32con.WM_DESTROY: self.OnDestroy,
            WM_SEARCH_RESULT: self.OnSearchResult,
            WM_SEARCH_FINISHED: self.OnSearchFinished,
        }
        dlgClassName = self._RegisterWndClass()
        template = self._GetDialogTemplate(dlgClassName)
        return fn(self.hinst, template, 0, message_map)

    def _SetupList(self):
        child_style = (
            win32con.WS_CHILD
            | win32con.WS_VISIBLE
            | win32con.WS_BORDER
            | win32con.WS_HSCROLL
            | win32con.WS_VSCROLL
        )
        child_style |= (
            commctrl.LVS_SINGLESEL | commctrl.LVS_SHOWSELALWAYS | commctrl.LVS_REPORT
        )
        self.hwndList = win32gui.CreateWindow(
            "SysListView32",
            None,
            child_style,
            0,
            0,
            100,
            100,
            self.hwnd,
            IDC_LISTBOX,
            self.hinst,
            None,
        )

        child_ex_style = win32gui.SendMessage(
            self.hwndList, commctrl.LVM_GETEXTENDEDLISTVIEWSTYLE, 0, 0
        )
        child_ex_style |= commctrl.LVS_EX_FULLROWSELECT
        win32gui.SendMessage(
            self.hwndList, commctrl.LVM_SETEXTENDEDLISTVIEWSTYLE, 0, child_ex_style
        )

        # Add an image list - use the builtin shell folder icon - this
        # demonstrates the problem with alpha-blending of icons on XP if
        # winxpgui is not used in place of win32gui.
        il = win32gui.ImageList_Create(
            win32api.GetSystemMetrics(win32con.SM_CXSMICON),
            win32api.GetSystemMetrics(win32con.SM_CYSMICON),
            commctrl.ILC_COLOR32 | commctrl.ILC_MASK,
            1,  # initial size
            0,
        )  # cGrow

        shell_dll = os.path.join(win32api.GetSystemDirectory(), "shell32.dll")
        large, small = win32gui.ExtractIconEx(shell_dll, 4, 1)
        win32gui.ImageList_ReplaceIcon(il, -1, small[0])
        win32gui.DestroyIcon(small[0])
        win32gui.DestroyIcon(large[0])
        win32gui.SendMessage(
            self.hwndList, commctrl.LVM_SETIMAGELIST, commctrl.LVSIL_SMALL, il
        )

        # Setup the list control columns.
        lvc = LVCOLUMN(
            mask=commctrl.LVCF_FMT
            | commctrl.LVCF_WIDTH
            | commctrl.LVCF_TEXT
            | commctrl.LVCF_SUBITEM
        )
        lvc.fmt = commctrl.LVCFMT_LEFT
        lvc.iSubItem = 1
        lvc.text = "Title"
        lvc.cx = 200
        win32gui.SendMessage(self.hwndList, commctrl.LVM_INSERTCOLUMN, 0, lvc.toparam())
        lvc.iSubItem = 0
        lvc.text = "Order"
        lvc.cx = 50
        win32gui.SendMessage(self.hwndList, commctrl.LVM_INSERTCOLUMN, 0, lvc.toparam())

        win32gui.UpdateWindow(self.hwnd)

    def ClearListItems(self):
        win32gui.SendMessage(self.hwndList, commctrl.LVM_DELETEALLITEMS)
        self.list_data = {}

    def AddListItem(self, data, *columns):
        num_items = win32gui.SendMessage(self.hwndList, commctrl.LVM_GETITEMCOUNT)
        item = LVITEM(text=columns[0], iItem=num_items)
        new_index = win32gui.SendMessage(
            self.hwndList, commctrl.LVM_INSERTITEM, 0, item.toparam()
        )
        col_no = 1
        for col in columns[1:]:
            item = LVITEM(text=col, iItem=new_index, iSubItem=col_no)
            win32gui.SendMessage(self.hwndList, commctrl.LVM_SETITEM, 0, item.toparam())
            col_no += 1
        self.list_data[new_index] = data

    def OnInitDialog(self, hwnd, msg, wparam, lparam):
        self.hwnd = hwnd
        # centre the dialog
        desktop = win32gui.GetDesktopWindow()
        l, t, r, b = win32gui.GetWindowRect(self.hwnd)
        dt_l, dt_t, dt_r, dt_b = win32gui.GetWindowRect(desktop)
        centre_x, centre_y = win32gui.ClientToScreen(
            desktop, ((dt_r - dt_l) // 2, (dt_b - dt_t) // 2)
        )
        win32gui.MoveWindow(
            hwnd, centre_x - (r // 2), centre_y - (b // 2), r - l, b - t, 0
        )
        self._SetupList()
        l, t, r, b = win32gui.GetClientRect(self.hwnd)
        self._DoSize(r - l, b - t, 1)

    def _DoSize(self, cx, cy, repaint=1):
        # right-justify the textbox.
        ctrl = win32gui.GetDlgItem(self.hwnd, IDC_SEARCHTEXT)
        l, t, r, b = win32gui.GetWindowRect(ctrl)
        l, t = win32gui.ScreenToClient(self.hwnd, (l, t))
        r, b = win32gui.ScreenToClient(self.hwnd, (r, b))
        win32gui.MoveWindow(ctrl, l, t, cx - l - 5, b - t, repaint)
        # The button.
        ctrl = win32gui.GetDlgItem(self.hwnd, IDC_BUTTON_DISPLAY)
        l, t, r, b = win32gui.GetWindowRect(ctrl)
        l, t = win32gui.ScreenToClient(self.hwnd, (l, t))
        r, b = win32gui.ScreenToClient(self.hwnd, (r, b))
        list_y = b + 10
        w = r - l
        win32gui.MoveWindow(ctrl, cx - 5 - w, t, w, b - t, repaint)

        # The list control
        win32gui.MoveWindow(self.hwndList, 0, list_y, cx, cy - list_y, repaint)
        # The last column of the list control.
        new_width = cx - win32gui.SendMessage(
            self.hwndList, commctrl.LVM_GETCOLUMNWIDTH, 0
        )
        win32gui.SendMessage(self.hwndList, commctrl.LVM_SETCOLUMNWIDTH, 1, new_width)

    def OnSize(self, hwnd, msg, wparam, lparam):
        x = win32api.LOWORD(lparam)
        y = win32api.HIWORD(lparam)
        self._DoSize(x, y)
        return 1

    def OnSearchResult(self, hwnd, msg, wparam, lparam):
        try:
            while 1:
                params = self.result_queue.get(0)
                self.AddListItem(*params)
        except queue.Empty:
            pass

    def OnSearchFinished(self, hwnd, msg, wparam, lparam):
        print("OnSearchFinished")

    def OnNotify(self, hwnd, msg, wparam, lparam):
        info = win32gui_struct.UnpackNMITEMACTIVATE(lparam)
        if info.code == commctrl.NM_DBLCLK:
            print("Double click on item", info.iItem + 1)
        return 1

    def OnCommand(self, hwnd, msg, wparam, lparam):
        id = win32api.LOWORD(wparam)
        if id == IDC_BUTTON_SEARCH:
            self.ClearListItems()

            def fill_slowly(q, hwnd):
                import time

                for i in range(20):
                    q.put(("whatever", str(i + 1), "Search result " + str(i)))
                    win32gui.PostMessage(hwnd, WM_SEARCH_RESULT, 0, 0)
                    time.sleep(0.25)
                win32gui.PostMessage(hwnd, WM_SEARCH_FINISHED, 0, 0)

            import threading

            self.result_queue = queue.Queue()
            thread = threading.Thread(
                target=fill_slowly, args=(self.result_queue, self.hwnd)
            )
            thread.start()
        elif id == IDC_BUTTON_DISPLAY:
            print("Display button selected")
            sel = win32gui.SendMessage(
                self.hwndList, commctrl.LVM_GETNEXTITEM, -1, commctrl.LVNI_SELECTED
            )
            print("The selected item is", sel + 1)

    # These function differ based on how the window is used, so may be overridden
    def OnClose(self, hwnd, msg, wparam, lparam):
        raise NotImplementedError

    def OnDestroy(self, hwnd, msg, wparam, lparam):
        pass


# An implementation suitable for use with the Win32 Window functions (ie, not
# a true dialog)
class DemoWindow(DemoWindowBase):
    def CreateWindow(self):
        # Create the window via CreateDialogBoxIndirect - it can then
        # work as a "normal" window, once a message loop is established.
        self._DoCreate(win32gui.CreateDialogIndirect)

    def OnClose(self, hwnd, msg, wparam, lparam):
        win32gui.DestroyWindow(hwnd)

    # We need to arrange to a WM_QUIT message to be sent to our
    # PumpMessages() loop.
    def OnDestroy(self, hwnd, msg, wparam, lparam):
        win32gui.PostQuitMessage(0)  # Terminate the app.


# An implementation suitable for use with the Win32 Dialog functions.
class DemoDialog(DemoWindowBase):
    def DoModal(self):
        return self._DoCreate(win32gui.DialogBoxIndirect)

    def OnClose(self, hwnd, msg, wparam, lparam):
        win32gui.EndDialog(hwnd, 0)


def DemoModal():
    w = DemoDialog()
    w.DoModal()


def DemoCreateWindow():
    w = DemoWindow()
    w.CreateWindow()
    # PumpMessages runs until PostQuitMessage() is called by someone.
    win32gui.PumpMessages()


if __name__ == "__main__":
    DemoModal()
    DemoCreateWindow()
