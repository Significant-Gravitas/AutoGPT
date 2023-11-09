# -*- coding: latin-1 -*-

# PyWin32 Internet Explorer Button
#
# written by Leonard Ritter (paniq@gmx.net)
# and Robert Förtsch (info@robert-foertsch.com)


"""
This sample implements a simple IE Button COM server
with access to the IWebBrowser2 interface.

To demonstrate:
* Execute this script to register the server.
* Open Pythonwin's Tools -> Trace Collector Debugging Tool, so you can
  see the output of 'print' statements in this demo.
* Open a new IE instance.  The toolbar should have a new "scissors" icon,
  with tooltip text "IE Button" - this is our new button - click it.
* Switch back to the Pythonwin window - you should see:
   IOleCommandTarget::Exec called.
  This is the button being clicked.  Extending this to do something more
  useful is left as an exercise.

Contribtions to this sample to make it a little "friendlier" welcome!
"""

# imports section

import pythoncom
import win32api
import win32com
import win32com.server.register
from win32com import universal
from win32com.client import Dispatch, DispatchWithEvents, constants, gencache, getevents

# This demo uses 'print' - use win32traceutil to see it if we have no
# console.
try:
    win32api.GetConsoleTitle()
except win32api.error:
    import win32traceutil

import array

from win32com.axcontrol import axcontrol

# ensure we know the ms internet controls typelib so we have access to IWebBrowser2 later on
win32com.client.gencache.EnsureModule("{EAB22AC0-30C1-11CF-A7EB-0000C05BAE0B}", 0, 1, 1)


#
IObjectWithSite_methods = ["SetSite", "GetSite"]
IOleCommandTarget_methods = ["Exec", "QueryStatus"]

_iebutton_methods_ = IOleCommandTarget_methods + IObjectWithSite_methods
_iebutton_com_interfaces_ = [
    axcontrol.IID_IOleCommandTarget,
    axcontrol.IID_IObjectWithSite,  # IObjectWithSite
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


class IEButton:
    """
    The actual COM server class
    """

    _com_interfaces_ = _iebutton_com_interfaces_
    _public_methods_ = _iebutton_methods_
    _reg_clsctx_ = pythoncom.CLSCTX_INPROC_SERVER
    _button_text_ = "IE Button"
    _tool_tip_ = "An example implementation for an IE Button."
    _icon_ = ""
    _hot_icon_ = ""

    def __init__(self):
        # put stubs for non-implemented methods
        for method in self._public_methods_:
            if not hasattr(self, method):
                print("providing default stub for %s" % method)
                setattr(self, method, Stub(method))

    def QueryStatus(self, pguidCmdGroup, prgCmds, cmdtextf):
        # 'cmdtextf' is the 'cmdtextf' element from the OLECMDTEXT structure,
        # or None if a NULL pointer was passed.
        result = []
        for id, flags in prgCmds:
            flags |= axcontrol.OLECMDF_SUPPORTED | axcontrol.OLECMDF_ENABLED
            result.append((id, flags))
        if cmdtextf is None:
            cmdtext = None  # must return None if nothing requested.
        # IE never seems to want any text - this code is here for
        # demo purposes only
        elif cmdtextf == axcontrol.OLECMDTEXTF_NAME:
            cmdtext = "IEButton Name"
        else:
            cmdtext = "IEButton State"
        return result, cmdtext

    def Exec(self, pguidCmdGroup, nCmdID, nCmdExecOpt, pvaIn):
        print(pguidCmdGroup, nCmdID, nCmdExecOpt, pvaIn)
        print("IOleCommandTarget::Exec called.")
        # self.webbrowser.ShowBrowserBar(GUID_IETOOLBAR, not is_ietoolbar_visible())

    def SetSite(self, unknown):
        if unknown:
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
        else:
            # lose all references
            self.webbrowser = None

    def GetClassID(self):
        return self._reg_clsid_


def register(classobj):
    import winreg

    subKeyCLSID = (
        "SOFTWARE\\Microsoft\\Internet Explorer\\Extensions\\%38s"
        % classobj._reg_clsid_
    )
    try:
        hKey = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, subKeyCLSID)
        subKey = winreg.SetValueEx(
            hKey, "ButtonText", 0, winreg.REG_SZ, classobj._button_text_
        )
        winreg.SetValueEx(
            hKey, "ClsidExtension", 0, winreg.REG_SZ, classobj._reg_clsid_
        )  # reg value for calling COM object
        winreg.SetValueEx(
            hKey, "CLSID", 0, winreg.REG_SZ, "{1FBA04EE-3024-11D2-8F1F-0000F87ABD16}"
        )  # CLSID for button that sends command to COM object
        winreg.SetValueEx(hKey, "Default Visible", 0, winreg.REG_SZ, "Yes")
        winreg.SetValueEx(hKey, "ToolTip", 0, winreg.REG_SZ, classobj._tool_tip_)
        winreg.SetValueEx(hKey, "Icon", 0, winreg.REG_SZ, classobj._icon_)
        winreg.SetValueEx(hKey, "HotIcon", 0, winreg.REG_SZ, classobj._hot_icon_)
    except WindowsError:
        print("Couldn't set standard toolbar reg keys.")
    else:
        print("Set standard toolbar reg keys.")


def unregister(classobj):
    import winreg

    subKeyCLSID = (
        "SOFTWARE\\Microsoft\\Internet Explorer\\Extensions\\%38s"
        % classobj._reg_clsid_
    )
    try:
        hKey = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, subKeyCLSID)
        subKey = winreg.DeleteValue(hKey, "ButtonText")
        winreg.DeleteValue(hKey, "ClsidExtension")  # for calling COM object
        winreg.DeleteValue(hKey, "CLSID")
        winreg.DeleteValue(hKey, "Default Visible")
        winreg.DeleteValue(hKey, "ToolTip")
        winreg.DeleteValue(hKey, "Icon")
        winreg.DeleteValue(hKey, "HotIcon")
        winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, subKeyCLSID)
    except WindowsError:
        print("Couldn't delete Standard toolbar regkey.")
    else:
        print("Deleted Standard toolbar regkey.")


#
# test implementation
#


class PyWin32InternetExplorerButton(IEButton):
    _reg_clsid_ = "{104B66A9-9E68-49D1-A3F5-94754BE9E0E6}"
    _reg_progid_ = "PyWin32.IEButton"
    _reg_desc_ = "Test Button"
    _button_text_ = "IE Button"
    _tool_tip_ = "An example implementation for an IE Button."
    _icon_ = ""
    _hot_icon_ = _icon_


def DllRegisterServer():
    register(PyWin32InternetExplorerButton)


def DllUnregisterServer():
    unregister(PyWin32InternetExplorerButton)


if __name__ == "__main__":
    win32com.server.register.UseCommandLine(
        PyWin32InternetExplorerButton,
        finalize_register=DllRegisterServer,
        finalize_unregister=DllUnregisterServer,
    )
