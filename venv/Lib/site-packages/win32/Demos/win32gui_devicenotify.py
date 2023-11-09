# Demo RegisterDeviceNotification etc.  Creates a hidden window to receive
# notifications.  See serviceEvents.py for an example of a service doing
# that.
import sys
import time

import win32api
import win32con
import win32file
import win32gui
import win32gui_struct
import winnt

# These device GUIDs are from Ioevent.h in the Windows SDK.  Ideally they
# could be collected somewhere for pywin32...
GUID_DEVINTERFACE_USB_DEVICE = "{A5DCBF10-6530-11D2-901F-00C04FB951ED}"


# WM_DEVICECHANGE message handler.
def OnDeviceChange(hwnd, msg, wp, lp):
    # Unpack the 'lp' into the appropriate DEV_BROADCAST_* structure,
    # using the self-identifying data inside the DEV_BROADCAST_HDR.
    info = win32gui_struct.UnpackDEV_BROADCAST(lp)
    print("Device change notification:", wp, str(info))
    if (
        wp == win32con.DBT_DEVICEQUERYREMOVE
        and info.devicetype == win32con.DBT_DEVTYP_HANDLE
    ):
        # Our handle is stored away in the structure - just close it
        print("Device being removed - closing handle")
        win32file.CloseHandle(info.handle)
        # and cancel our notifications - if it gets plugged back in we get
        # the same notification and try and close the same handle...
        win32gui.UnregisterDeviceNotification(info.hdevnotify)
    return True


def TestDeviceNotifications(dir_names):
    wc = win32gui.WNDCLASS()
    wc.lpszClassName = "test_devicenotify"
    wc.style = win32con.CS_GLOBALCLASS | win32con.CS_VREDRAW | win32con.CS_HREDRAW
    wc.hbrBackground = win32con.COLOR_WINDOW + 1
    wc.lpfnWndProc = {win32con.WM_DEVICECHANGE: OnDeviceChange}
    class_atom = win32gui.RegisterClass(wc)
    hwnd = win32gui.CreateWindow(
        wc.lpszClassName,
        "Testing some devices",
        # no need for it to be visible.
        win32con.WS_CAPTION,
        100,
        100,
        900,
        900,
        0,
        0,
        0,
        None,
    )

    hdevs = []
    # Watch for all USB device notifications
    filter = win32gui_struct.PackDEV_BROADCAST_DEVICEINTERFACE(
        GUID_DEVINTERFACE_USB_DEVICE
    )
    hdev = win32gui.RegisterDeviceNotification(
        hwnd, filter, win32con.DEVICE_NOTIFY_WINDOW_HANDLE
    )
    hdevs.append(hdev)
    # and create handles for all specified directories
    for d in dir_names:
        hdir = win32file.CreateFile(
            d,
            winnt.FILE_LIST_DIRECTORY,
            winnt.FILE_SHARE_READ | winnt.FILE_SHARE_WRITE | winnt.FILE_SHARE_DELETE,
            None,  # security attributes
            win32con.OPEN_EXISTING,
            win32con.FILE_FLAG_BACKUP_SEMANTICS
            | win32con.FILE_FLAG_OVERLAPPED,  # required privileges: SE_BACKUP_NAME and SE_RESTORE_NAME.
            None,
        )

        filter = win32gui_struct.PackDEV_BROADCAST_HANDLE(hdir)
        hdev = win32gui.RegisterDeviceNotification(
            hwnd, filter, win32con.DEVICE_NOTIFY_WINDOW_HANDLE
        )
        hdevs.append(hdev)

    # now start a message pump and wait for messages to be delivered.
    print("Watching", len(hdevs), "handles - press Ctrl+C to terminate, or")
    print("add and remove some USB devices...")
    if not dir_names:
        print("(Note you can also pass paths to watch on the command-line - eg,")
        print("pass the root of an inserted USB stick to see events specific to")
        print("that volume)")
    while 1:
        win32gui.PumpWaitingMessages()
        time.sleep(0.01)
    win32gui.DestroyWindow(hwnd)
    win32gui.UnregisterClass(wc.lpszClassName, None)


if __name__ == "__main__":
    # optionally pass device/directory names to watch for notifications.
    # Eg, plug in a USB device - assume it connects as E: - then execute:
    # % win32gui_devicenotify.py E:
    # Then remove and insert the device.
    TestDeviceNotifications(sys.argv[1:])
