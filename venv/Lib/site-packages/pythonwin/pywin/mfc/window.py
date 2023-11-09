# The MFCish window classes.
import win32con
import win32ui

from . import object


class Wnd(object.CmdTarget):
    def __init__(self, initobj=None):
        object.CmdTarget.__init__(self, initobj)
        if self._obj_:
            self._obj_.HookMessage(self.OnDestroy, win32con.WM_DESTROY)

    def OnDestroy(self, msg):
        pass


# NOTE NOTE - This facility is currently disabled in Pythonwin!!!!!
# Note - to process all messages for your window, add the following method
# to a derived class.  This code provides default message handling (ie, is
# identical, except presumably in speed, as if the method did not exist at
# all, so presumably will be modified to test for specific messages to be
# useful!
# 	def WindowProc(self, msg, wParam, lParam):
# 		rc, lResult = self._obj_.OnWndMsg(msg, wParam, lParam)
# 		if not rc: lResult = self._obj_.DefWindowProc(msg, wParam, lParam)
# 		return lResult


class FrameWnd(Wnd):
    def __init__(self, wnd):
        Wnd.__init__(self, wnd)


class MDIChildWnd(FrameWnd):
    def __init__(self, wnd=None):
        if wnd is None:
            wnd = win32ui.CreateMDIChild()
        FrameWnd.__init__(self, wnd)

    def OnCreateClient(self, cp, context):
        if context is not None and context.template is not None:
            context.template.CreateView(self, context)


class MDIFrameWnd(FrameWnd):
    def __init__(self, wnd=None):
        if wnd is None:
            wnd = win32ui.CreateMDIFrame()
        FrameWnd.__init__(self, wnd)
