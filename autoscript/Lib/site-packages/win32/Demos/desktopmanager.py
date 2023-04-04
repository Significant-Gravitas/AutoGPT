# Demonstrates using a taskbar icon to create and navigate between desktops

import _thread
import io
import time
import traceback

import pywintypes
import win32api
import win32con
import win32gui
import win32process
import win32service

## "Shell_TrayWnd" is class of system tray window, broadcasts "TaskbarCreated" when initialized


def desktop_name_dlgproc(hwnd, msg, wparam, lparam):
    """Handles messages from the desktop name dialog box"""
    if msg in (win32con.WM_CLOSE, win32con.WM_DESTROY):
        win32gui.DestroyWindow(hwnd)
    elif msg == win32con.WM_COMMAND:
        if wparam == win32con.IDOK:
            desktop_name = win32gui.GetDlgItemText(hwnd, 72)
            print("new desktop name: ", desktop_name)
            win32gui.DestroyWindow(hwnd)
            create_desktop(desktop_name)

        elif wparam == win32con.IDCANCEL:
            win32gui.DestroyWindow(hwnd)


def get_new_desktop_name(parent_hwnd):
    """Create a dialog box to ask the user for name of desktop to be created"""
    msgs = {
        win32con.WM_COMMAND: desktop_name_dlgproc,
        win32con.WM_CLOSE: desktop_name_dlgproc,
        win32con.WM_DESTROY: desktop_name_dlgproc,
    }
    # dlg item [type, caption, id, (x,y,cx,cy), style, ex style
    style = (
        win32con.WS_BORDER
        | win32con.WS_VISIBLE
        | win32con.WS_CAPTION
        | win32con.WS_SYSMENU
    )  ## |win32con.DS_SYSMODAL
    h = win32gui.CreateDialogIndirect(
        win32api.GetModuleHandle(None),
        [
            ["One ugly dialog box !", (100, 100, 200, 100), style, 0],
            [
                "Button",
                "Create",
                win32con.IDOK,
                (10, 10, 30, 20),
                win32con.WS_VISIBLE
                | win32con.WS_TABSTOP
                | win32con.BS_HOLLOW
                | win32con.BS_DEFPUSHBUTTON,
            ],
            [
                "Button",
                "Never mind",
                win32con.IDCANCEL,
                (45, 10, 50, 20),
                win32con.WS_VISIBLE | win32con.WS_TABSTOP | win32con.BS_HOLLOW,
            ],
            ["Static", "Desktop name:", 71, (10, 40, 70, 10), win32con.WS_VISIBLE],
            ["Edit", "", 72, (75, 40, 90, 10), win32con.WS_VISIBLE],
        ],
        parent_hwnd,
        msgs,
    )  ## parent_hwnd, msgs)

    win32gui.EnableWindow(h, True)
    hcontrol = win32gui.GetDlgItem(h, 72)
    win32gui.EnableWindow(hcontrol, True)
    win32gui.SetFocus(hcontrol)


def new_icon(hdesk, desktop_name):
    """Runs as a thread on each desktop to create a new tray icon and handle its messages"""
    global id
    id = id + 1
    hdesk.SetThreadDesktop()
    ## apparently the threads can't use same hinst, so each needs its own window class
    windowclassname = "PythonDesktopManager" + desktop_name
    wc = win32gui.WNDCLASS()
    wc.hInstance = win32api.GetModuleHandle(None)
    wc.lpszClassName = windowclassname
    wc.style = win32con.CS_VREDRAW | win32con.CS_HREDRAW | win32con.CS_GLOBALCLASS
    wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
    wc.hbrBackground = win32con.COLOR_WINDOW
    wc.lpfnWndProc = icon_wndproc
    windowclass = win32gui.RegisterClass(wc)
    style = win32con.WS_OVERLAPPED | win32con.WS_SYSMENU
    hwnd = win32gui.CreateWindow(
        windowclass,
        "dm_" + desktop_name,
        win32con.WS_SYSMENU,
        0,
        0,
        win32con.CW_USEDEFAULT,
        win32con.CW_USEDEFAULT,
        0,
        0,
        wc.hInstance,
        None,
    )
    win32gui.UpdateWindow(hwnd)
    flags = win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP
    notify_info = (
        hwnd,
        id,
        flags,
        win32con.WM_USER + 20,
        hicon,
        "Desktop Manager (%s)" % desktop_name,
    )
    window_info[hwnd] = notify_info
    ## wait for explorer to initialize system tray for new desktop
    tray_found = 0
    while not tray_found:
        try:
            tray_found = win32gui.FindWindow("Shell_TrayWnd", None)
        except win32gui.error:
            traceback.print_exc
            time.sleep(0.5)
    win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, notify_info)
    win32gui.PumpMessages()


def create_desktop(desktop_name, start_explorer=1):
    """Creates a new desktop and spawns a thread running on it
    Will also start a new icon thread on an existing desktop
    """
    sa = pywintypes.SECURITY_ATTRIBUTES()
    sa.bInheritHandle = 1

    try:
        hdesk = win32service.CreateDesktop(
            desktop_name, 0, win32con.MAXIMUM_ALLOWED, sa
        )
    except win32service.error:
        traceback.print_exc()
        errbuf = io.StringIO()
        traceback.print_exc(None, errbuf)
        win32api.MessageBox(0, errbuf.getvalue(), "Desktop creation failed")
        return
    if start_explorer:
        s = win32process.STARTUPINFO()
        s.lpDesktop = desktop_name
        prc_info = win32process.CreateProcess(
            None,
            "Explorer.exe",
            None,
            None,
            True,
            win32con.CREATE_NEW_CONSOLE,
            None,
            "c:\\",
            s,
        )

    th = _thread.start_new_thread(new_icon, (hdesk, desktop_name))
    hdesk.SwitchDesktop()


def icon_wndproc(hwnd, msg, wp, lp):
    """Window proc for the tray icons"""
    if lp == win32con.WM_LBUTTONDOWN:
        ## popup menu won't disappear if you don't do this
        win32gui.SetForegroundWindow(hwnd)

        curr_desktop = win32service.OpenInputDesktop(0, True, win32con.MAXIMUM_ALLOWED)
        curr_desktop_name = win32service.GetUserObjectInformation(
            curr_desktop, win32con.UOI_NAME
        )
        winsta = win32service.GetProcessWindowStation()
        desktops = winsta.EnumDesktops()
        m = win32gui.CreatePopupMenu()
        desktop_cnt = len(desktops)
        ## *don't* create an item 0
        for d in range(1, desktop_cnt + 1):
            mf_flags = win32con.MF_STRING
            ## if you switch to winlogon yourself, there's nothing there and you're stuck
            if desktops[d - 1].lower() in ("winlogon", "disconnect"):
                mf_flags = mf_flags | win32con.MF_GRAYED | win32con.MF_DISABLED
            if desktops[d - 1] == curr_desktop_name:
                mf_flags = mf_flags | win32con.MF_CHECKED
            win32gui.AppendMenu(m, mf_flags, d, desktops[d - 1])
        win32gui.AppendMenu(m, win32con.MF_STRING, desktop_cnt + 1, "Create new ...")
        win32gui.AppendMenu(m, win32con.MF_STRING, desktop_cnt + 2, "Exit")

        x, y = win32gui.GetCursorPos()
        d = win32gui.TrackPopupMenu(
            m,
            win32con.TPM_LEFTBUTTON | win32con.TPM_RETURNCMD | win32con.TPM_NONOTIFY,
            x,
            y,
            0,
            hwnd,
            None,
        )
        win32gui.PumpWaitingMessages()
        win32gui.DestroyMenu(m)
        if d == desktop_cnt + 1:  ## Create new
            get_new_desktop_name(hwnd)
        elif d == desktop_cnt + 2:  ## Exit
            win32gui.PostQuitMessage(0)
            win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, window_info[hwnd])
            del window_info[hwnd]
            origin_desktop.SwitchDesktop()
        elif d > 0:
            hdesk = win32service.OpenDesktop(
                desktops[d - 1], 0, 0, win32con.MAXIMUM_ALLOWED
            )
            hdesk.SwitchDesktop()
        return 0
    else:
        return win32gui.DefWindowProc(hwnd, msg, wp, lp)


window_info = {}
origin_desktop = win32service.OpenInputDesktop(0, True, win32con.MAXIMUM_ALLOWED)
origin_desktop_name = win32service.GetUserObjectInformation(
    origin_desktop, win32service.UOI_NAME
)

hinst = win32api.GetModuleHandle(None)
try:
    hicon = win32gui.LoadIcon(hinst, 1)  ## python.exe and pythonw.exe
except win32gui.error:
    hicon = win32gui.LoadIcon(hinst, 135)  ## pythonwin's icon
id = 0

create_desktop(str(origin_desktop_name), 0)

## wait for first thread to initialize its icon
while not window_info:
    time.sleep(1)

## exit when last tray icon goes away
while window_info:
    win32gui.PumpWaitingMessages()
    time.sleep(3)
