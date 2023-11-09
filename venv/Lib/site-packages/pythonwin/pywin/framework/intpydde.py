# DDE support for Pythonwin
#
# Seems to work fine (in the context that IE4 seems to have broken
# DDE on _all_ NT4 machines I have tried, but only when a "Command Prompt" window
# is open.  Strange, but true.  If you have problems with this, close all Command Prompts!


import sys
import traceback

import win32api
import win32ui
from dde import *
from pywin.mfc import object


class DDESystemTopic(object.Object):
    def __init__(self, app):
        self.app = app
        object.Object.__init__(self, CreateServerSystemTopic())

    def Exec(self, data):
        try:
            # 			print "Executing", cmd
            self.app.OnDDECommand(data)
        except:
            t, v, tb = sys.exc_info()
            # The DDE Execution failed.
            print("Error executing DDE command.")
            traceback.print_exception(t, v, tb)
            return 0


class DDEServer(object.Object):
    def __init__(self, app):
        self.app = app
        object.Object.__init__(self, CreateServer())
        self.topic = self.item = None

    def CreateSystemTopic(self):
        return DDESystemTopic(self.app)

    def Shutdown(self):
        self._obj_.Shutdown()
        self._obj_.Destroy()
        if self.topic is not None:
            self.topic.Destroy()
            self.topic = None
        if self.item is not None:
            self.item.Destroy()
            self.item = None

    def OnCreate(self):
        return 1

    def Status(self, msg):
        try:
            win32ui.SetStatusText(msg)
        except win32ui.error:
            pass
