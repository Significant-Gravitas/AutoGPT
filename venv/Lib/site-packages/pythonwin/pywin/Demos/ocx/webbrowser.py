# This demo uses the IE4 Web Browser control.

# It catches an "OnNavigate" event, and updates the frame title.
# (event stuff by Neil Hodgson)

import sys

import regutil
import win32api
import win32con
import win32ui
from pywin.mfc import activex, window
from win32com.client import gencache

WebBrowserModule = gencache.EnsureModule(
    "{EAB22AC0-30C1-11CF-A7EB-0000C05BAE0B}", 0, 1, 1
)
if WebBrowserModule is None:
    raise ImportError("IE4 does not appear to be installed.")


class MyWebBrowser(activex.Control, WebBrowserModule.WebBrowser):
    def OnBeforeNavigate2(
        self, pDisp, URL, Flags, TargetFrameName, PostData, Headers, Cancel
    ):
        self.GetParent().OnNavigate(URL)
        # print "BeforeNavigate2", pDisp, URL, Flags, TargetFrameName, PostData, Headers, Cancel


class BrowserFrame(window.MDIChildWnd):
    def __init__(self, url=None):
        if url is None:
            self.url = regutil.GetRegisteredHelpFile("Main Python Documentation")
            if self.url is None:
                self.url = "http://www.python.org"
        else:
            self.url = url
        pass  # Dont call base class doc/view version...

    def Create(self, title, rect=None, parent=None):
        style = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_OVERLAPPEDWINDOW
        self._obj_ = win32ui.CreateMDIChild()
        self._obj_.AttachObject(self)
        self._obj_.CreateWindow(None, title, style, rect, parent)
        rect = self.GetClientRect()
        rect = (0, 0, rect[2] - rect[0], rect[3] - rect[1])
        self.ocx = MyWebBrowser()
        self.ocx.CreateControl(
            "Web Browser", win32con.WS_VISIBLE | win32con.WS_CHILD, rect, self, 1000
        )
        self.ocx.Navigate(self.url)
        self.HookMessage(self.OnSize, win32con.WM_SIZE)

    def OnSize(self, params):
        rect = self.GetClientRect()
        rect = (0, 0, rect[2] - rect[0], rect[3] - rect[1])
        self.ocx.SetWindowPos(0, rect, 0)

    def OnNavigate(self, url):
        title = "Web Browser - %s" % (url,)
        self.SetWindowText(title)


def Demo(url=None):
    if url is None and len(sys.argv) > 1:
        url = win32api.GetFullPathName(sys.argv[1])
    f = BrowserFrame(url)
    f.Create("Web Browser")


if __name__ == "__main__":
    Demo()
