# By Bradley Schatz
# simple flash/python application demonstrating bidirectional
# communicaion between flash and python. Click the sphere to see
# behavior. Uses Bounce.swf from FlashBounce.zip, available from
# http://pages.cpsc.ucalgary.ca/~saul/vb_examples/tutorial12/

# Update to the path of the .swf file (note it could be a true URL)
flash_url = "c:\\bounce.swf"

import sys

import regutil
import win32api
import win32con
import win32ui
from pywin.mfc import activex, window
from win32com.client import gencache

FlashModule = gencache.EnsureModule("{D27CDB6B-AE6D-11CF-96B8-444553540000}", 0, 1, 0)

if FlashModule is None:
    raise ImportError("Flash does not appear to be installed.")


class MyFlashComponent(activex.Control, FlashModule.ShockwaveFlash):
    def __init__(self):
        activex.Control.__init__(self)
        FlashModule.ShockwaveFlash.__init__(self)
        self.x = 50
        self.y = 50
        self.angle = 30
        self.started = 0

    def OnFSCommand(self, command, args):
        print("FSCommend", command, args)
        self.x = self.x + 20
        self.y = self.y + 20
        self.angle = self.angle + 20
        if self.x > 200 or self.y > 200:
            self.x = 0
            self.y = 0
        if self.angle > 360:
            self.angle = 0
        self.SetVariable("xVal", self.x)
        self.SetVariable("yVal", self.y)
        self.SetVariable("angle", self.angle)
        self.TPlay("_root.mikeBall")

    def OnProgress(self, percentDone):
        print("PercentDone", percentDone)

    def OnReadyStateChange(self, newState):
        # 0=Loading, 1=Uninitialized, 2=Loaded, 3=Interactive, 4=Complete
        print("State", newState)


class BrowserFrame(window.MDIChildWnd):
    def __init__(self, url=None):
        if url is None:
            self.url = regutil.GetRegisteredHelpFile("Main Python Documentation")
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
        self.ocx = MyFlashComponent()
        self.ocx.CreateControl(
            "Flash Player", win32con.WS_VISIBLE | win32con.WS_CHILD, rect, self, 1000
        )
        self.ocx.LoadMovie(0, flash_url)
        self.ocx.Play()
        self.HookMessage(self.OnSize, win32con.WM_SIZE)

    def OnSize(self, params):
        rect = self.GetClientRect()
        rect = (0, 0, rect[2] - rect[0], rect[3] - rect[1])
        self.ocx.SetWindowPos(0, rect, 0)


def Demo():
    url = None
    if len(sys.argv) > 1:
        url = win32api.GetFullPathName(sys.argv[1])
    f = BrowserFrame(url)
    f.Create("Flash Player")


if __name__ == "__main__":
    Demo()
