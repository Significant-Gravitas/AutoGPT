# -*- coding: latin-1 -*-

# PyWin32 Internet Explorer Toolbar
#
# written by Leonard Ritter (paniq@gmx.net)
# and Robert Förtsch (info@robert-foertsch.com)


"""
This sample implements a simple IE Toolbar COM server
supporting Windows XP styles and access to
the IWebBrowser2 interface.

It also demonstrates how to hijack the parent window
to catch WM_COMMAND messages.
"""

# imports section
import sys
import winreg

import pythoncom
import win32com
from win32com import universal
from win32com.axcontrol import axcontrol
from win32com.client import Dispatch, DispatchWithEvents, constants, gencache, getevents
from win32com.shell import shell
from win32com.shell.shellcon import *

try:
    # try to get styles (winxp)
    import winxpgui as win32gui
except:
    # import default module (win2k and lower)
    import win32gui

import array
import struct

import commctrl
import win32con
import win32ui

# ensure we know the ms internet controls typelib so we have access to IWebBrowser2 later on
win32com.client.gencache.EnsureModule("{EAB22AC0-30C1-11CF-A7EB-0000C05BAE0B}", 0, 1, 1)

#
IDeskBand_methods = ["GetBandInfo"]
IDockingWindow_methods = ["ShowDW", "CloseDW", "ResizeBorderDW"]
IOleWindow_methods = ["GetWindow", "ContextSensitiveHelp"]
IInputObject_methods = ["UIActivateIO", "HasFocusIO", "TranslateAcceleratorIO"]
IObjectWithSite_methods = ["SetSite", "GetSite"]
IPersistStream_methods = ["GetClassID", "IsDirty", "Load", "Save", "GetSizeMax"]

_ietoolbar_methods_ = (
    IDeskBand_methods
    + IDockingWindow_methods
    + IOleWindow_methods
    + IInputObject_methods
    + IObjectWithSite_methods
    + IPersistStream_methods
)
_ietoolbar_com_interfaces_ = [
    shell.IID_IDeskBand,  # IDeskBand
    axcontrol.IID_IObjectWithSite,  # IObjectWithSite
    pythoncom.IID_IPersistStream,
    axcontrol.IID_IOleCommandTarget,
]


class WIN32STRUCT:
    def __init__(self, **kw):
        full_fmt = ""
        for name, fmt, default in self._struct_items_:
            self.__dict__[name] = None
            if fmt == "z":
                full_fmt += "pi"
            else:
                full_fmt += fmt
        for name, val in kw.items():
            self.__dict__[name] = val

    def __setattr__(self, attr, val):
        if not attr.startswith("_") and attr not in self.__dict__:
            raise AttributeError(attr)
        self.__dict__[attr] = val

    def toparam(self):
        self._buffs = []
        full_fmt = ""
        vals = []
        for name, fmt, default in self._struct_items_:
            val = self.__dict__[name]
            if fmt == "z":
                fmt = "Pi"
                if val is None:
                    vals.append(0)
                    vals.append(0)
                else:
                    str_buf = array.array("c", val + "\0")
                    vals.append(str_buf.buffer_info()[0])
                    vals.append(len(val))
                    self._buffs.append(str_buf)  # keep alive during the call.
            else:
                if val is None:
                    val = default
                vals.append(val)
            full_fmt += fmt
        return struct.pack(*(full_fmt,) + tuple(vals))


class TBBUTTON(WIN32STRUCT):
    _struct_items_ = [
        ("iBitmap", "i", 0),
        ("idCommand", "i", 0),
        ("fsState", "B", 0),
        ("fsStyle", "B", 0),
        ("bReserved", "H", 0),
        ("dwData", "I", 0),
        ("iString", "z", None),
    ]


class Stub:
    """
    this class serves as a method stub,
    outputting debug info whenever the object
    is being called.
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, *args):
        print("STUB: ", self.name, args)


class IEToolbarCtrl:
    """
    a tiny wrapper for our winapi-based
    toolbar control implementation.
    """

    def __init__(self, hwndparent):
        styles = (
            win32con.WS_CHILD
            | win32con.WS_VISIBLE
            | win32con.WS_CLIPSIBLINGS
            | win32con.WS_CLIPCHILDREN
            | commctrl.TBSTYLE_LIST
            | commctrl.TBSTYLE_FLAT
            | commctrl.TBSTYLE_TRANSPARENT
            | commctrl.CCS_TOP
            | commctrl.CCS_NODIVIDER
            | commctrl.CCS_NORESIZE
            | commctrl.CCS_NOPARENTALIGN
        )
        self.hwnd = win32gui.CreateWindow(
            "ToolbarWindow32",
            None,
            styles,
            0,
            0,
            100,
            100,
            hwndparent,
            0,
            win32gui.dllhandle,
            None,
        )
        win32gui.SendMessage(self.hwnd, commctrl.TB_BUTTONSTRUCTSIZE, 20, 0)

    def ShowWindow(self, mode):
        win32gui.ShowWindow(self.hwnd, mode)

    def AddButtons(self, *buttons):
        tbbuttons = ""
        for button in buttons:
            tbbuttons += button.toparam()
        return win32gui.SendMessage(
            self.hwnd, commctrl.TB_ADDBUTTONS, len(buttons), tbbuttons
        )

    def GetSafeHwnd(self):
        return self.hwnd


class IEToolbar:
    """
    The actual COM server class
    """

    _com_interfaces_ = _ietoolbar_com_interfaces_
    _public_methods_ = _ietoolbar_methods_
    _reg_clsctx_ = pythoncom.CLSCTX_INPROC_SERVER
    # if you copy and modify this example, be sure to change the clsid below
    _reg_clsid_ = "{F21202A2-959A-4149-B1C3-68B9013F3335}"
    _reg_progid_ = "PyWin32.IEToolbar"
    _reg_desc_ = "PyWin32 IE Toolbar"

    def __init__(self):
        # put stubs for non-implemented methods
        for method in self._public_methods_:
            if not hasattr(self, method):
                print("providing default stub for %s" % method)
                setattr(self, method, Stub(method))

    def GetWindow(self):
        return self.toolbar.GetSafeHwnd()

    def Load(self, stream):
        # called when the toolbar is loaded
        pass

    def Save(self, pStream, fClearDirty):
        # called when the toolbar shall save its information
        pass

    def CloseDW(self, dwReserved):
        del self.toolbar

    def ShowDW(self, bShow):
        if bShow:
            self.toolbar.ShowWindow(win32con.SW_SHOW)
        else:
            self.toolbar.ShowWindow(win32con.SW_HIDE)

    def on_first_button(self):
        print("first!")
        self.webbrowser.Navigate2("http://starship.python.net/crew/mhammond/")

    def on_second_button(self):
        print("second!")

    def on_third_button(self):
        print("third!")

    def toolbar_command_handler(self, args):
        hwnd, message, wparam, lparam, time, point = args
        if lparam == self.toolbar.GetSafeHwnd():
            self._command_map[wparam]()

    def SetSite(self, unknown):
        if unknown:
            # retrieve the parent window interface for this site
            olewindow = unknown.QueryInterface(pythoncom.IID_IOleWindow)
            # ask the window for its handle
            hwndparent = olewindow.GetWindow()

            # first get a command target
            cmdtarget = unknown.QueryInterface(axcontrol.IID_IOleCommandTarget)
            # then travel over to a service provider
            serviceprovider = cmdtarget.QueryInterface(pythoncom.IID_IServiceProvider)
            # finally ask for the internet explorer application, returned as a dispatch object
            self.webbrowser = win32com.client.Dispatch(
                serviceprovider.QueryService(
                    "{0002DF05-0000-0000-C000-000000000046}", pythoncom.IID_IDispatch
                )
            )

            # now create and set up the toolbar
            self.toolbar = IEToolbarCtrl(hwndparent)

            buttons = [
                ("Visit PyWin32 Homepage", self.on_first_button),
                ("Another Button", self.on_second_button),
                ("Yet Another Button", self.on_third_button),
            ]

            self._command_map = {}
            # wrap our parent window so we can hook message handlers
            window = win32ui.CreateWindowFromHandle(hwndparent)

            # add the buttons
            for i in range(len(buttons)):
                button = TBBUTTON()
                name, func = buttons[i]
                id = 0x4444 + i
                button.iBitmap = -2
                button.idCommand = id
                button.fsState = commctrl.TBSTATE_ENABLED
                button.fsStyle = commctrl.TBSTYLE_BUTTON
                button.iString = name
                self._command_map[0x4444 + i] = func
                self.toolbar.AddButtons(button)
                window.HookMessage(self.toolbar_command_handler, win32con.WM_COMMAND)
        else:
            # lose all references
            self.webbrowser = None

    def GetClassID(self):
        return self._reg_clsid_

    def GetBandInfo(self, dwBandId, dwViewMode, dwMask):
        ptMinSize = (0, 24)
        ptMaxSize = (2000, 24)
        ptIntegral = (0, 0)
        ptActual = (2000, 24)
        wszTitle = "PyWin32 IE Toolbar"
        dwModeFlags = DBIMF_VARIABLEHEIGHT
        crBkgnd = 0
        return (
            ptMinSize,
            ptMaxSize,
            ptIntegral,
            ptActual,
            wszTitle,
            dwModeFlags,
            crBkgnd,
        )


# used for HKLM install
def DllInstall(bInstall, cmdLine):
    comclass = IEToolbar


# register plugin
def DllRegisterServer():
    comclass = IEToolbar

    # register toolbar with IE
    try:
        print("Trying to register Toolbar.\n")
        hkey = winreg.CreateKey(
            winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Internet Explorer\\Toolbar"
        )
        subKey = winreg.SetValueEx(
            hkey, comclass._reg_clsid_, 0, winreg.REG_BINARY, "\0"
        )
    except WindowsError:
        print(
            "Couldn't set registry value.\nhkey: %d\tCLSID: %s\n"
            % (hkey, comclass._reg_clsid_)
        )
    else:
        print(
            "Set registry value.\nhkey: %d\tCLSID: %s\n" % (hkey, comclass._reg_clsid_)
        )
    # TODO: implement reg settings for standard toolbar button


# unregister plugin
def DllUnregisterServer():
    comclass = IEToolbar

    # unregister toolbar from internet explorer
    try:
        print("Trying to unregister Toolbar.\n")
        hkey = winreg.CreateKey(
            winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Internet Explorer\\Toolbar"
        )
        winreg.DeleteValue(hkey, comclass._reg_clsid_)
    except WindowsError:
        print(
            "Couldn't delete registry value.\nhkey: %d\tCLSID: %s\n"
            % (hkey, comclass._reg_clsid_)
        )
    else:
        print("Deleting reg key succeeded.\n")


# entry point
if __name__ == "__main__":
    import win32com.server.register

    win32com.server.register.UseCommandLine(IEToolbar)

    # parse actual command line option
    if "--unregister" in sys.argv:
        DllUnregisterServer()
    else:
        DllRegisterServer()
else:
    # import trace utility for remote debugging
    import win32traceutil
