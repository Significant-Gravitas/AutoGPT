# win32traceutil like utility for Pythonwin
import _thread

import win32api
import win32event
import win32trace
from pywin.framework import winout

outputWindow = None


def CollectorThread(stopEvent, file):
    win32trace.InitRead()
    handle = win32trace.GetHandle()
    # Run this thread at a lower priority to the main message-loop (and printing output)
    # thread can keep up
    import win32process

    win32process.SetThreadPriority(
        win32api.GetCurrentThread(), win32process.THREAD_PRIORITY_BELOW_NORMAL
    )

    try:
        while 1:
            rc = win32event.WaitForMultipleObjects(
                (handle, stopEvent), 0, win32event.INFINITE
            )
            if rc == win32event.WAIT_OBJECT_0:
                # About the only char we can't live with is \0!
                file.write(win32trace.read().replace("\0", "<null>"))
            else:
                # Stop event
                break
    finally:
        win32trace.TermRead()
        print("Thread dieing")


class WindowOutput(winout.WindowOutput):
    def __init__(self, *args):
        winout.WindowOutput.__init__(*(self,) + args)
        self.hStopThread = win32event.CreateEvent(None, 0, 0, None)
        _thread.start_new(CollectorThread, (self.hStopThread, self))

    def _StopThread(self):
        win32event.SetEvent(self.hStopThread)
        self.hStopThread = None

    def Close(self):
        self._StopThread()
        winout.WindowOutput.Close(self)
        # 	def OnViewDestroy(self, frame):
        # 		return winout.WindowOutput.OnViewDestroy(self, frame)
        # 	def Create(self, title=None, style = None):
        # 		rc = winout.WindowOutput.Create(self, title, style)
        return rc


def MakeOutputWindow():
    # Note that it will not show until the first string written or
    # you pass bShow = 1
    global outputWindow
    if outputWindow is None:
        title = "Python Trace Collector"
        # queueingFlag doesnt matter, as all output will come from new thread
        outputWindow = WindowOutput(title, title)
        # Let people know what this does!
        msg = """\
# This window will display output from any programs that import win32traceutil
# win32com servers registered with '--debug' are in this category.
"""
        outputWindow.write(msg)
    # force existing window open
    outputWindow.write("")
    return outputWindow


if __name__ == "__main__":
    MakeOutputWindow()
