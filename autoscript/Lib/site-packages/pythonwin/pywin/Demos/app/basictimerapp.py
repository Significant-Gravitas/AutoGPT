# basictimerapp - a really simple timer application.
# This should be run using the command line:
# pythonwin /app demos\basictimerapp.py
import sys
import time

import timer
import win32api
import win32con
import win32ui
from pywin.framework import app, cmdline, dlgappcore


class TimerAppDialog(dlgappcore.AppDialog):
    softspace = 1

    def __init__(self, appName=""):
        dlgappcore.AppDialog.__init__(self, win32ui.IDD_GENERAL_STATUS)
        self.timerAppName = appName
        self.argOff = 0
        if len(self.timerAppName) == 0:
            if len(sys.argv) > 1 and sys.argv[1][0] != "/":
                self.timerAppName = sys.argv[1]
                self.argOff = 1

    def PreDoModal(self):
        # 		sys.stderr = sys.stdout
        pass

    def ProcessArgs(self, args):
        for arg in args:
            if arg == "/now":
                self.OnOK()

    def OnInitDialog(self):
        win32ui.SetProfileFileName("pytimer.ini")
        self.title = win32ui.GetProfileVal(
            self.timerAppName, "Title", "Remote System Timer"
        )
        self.buildTimer = win32ui.GetProfileVal(
            self.timerAppName, "Timer", "EachMinuteIntervaler()"
        )
        self.doWork = win32ui.GetProfileVal(self.timerAppName, "Work", "DoDemoWork()")
        # replace "\n" with real \n.
        self.doWork = self.doWork.replace("\\n", "\n")
        dlgappcore.AppDialog.OnInitDialog(self)

        self.SetWindowText(self.title)
        self.prompt1 = self.GetDlgItem(win32ui.IDC_PROMPT1)
        self.prompt2 = self.GetDlgItem(win32ui.IDC_PROMPT2)
        self.prompt3 = self.GetDlgItem(win32ui.IDC_PROMPT3)
        self.butOK = self.GetDlgItem(win32con.IDOK)
        self.butCancel = self.GetDlgItem(win32con.IDCANCEL)
        self.prompt1.SetWindowText("Python Timer App")
        self.prompt2.SetWindowText("")
        self.prompt3.SetWindowText("")
        self.butOK.SetWindowText("Do it now")
        self.butCancel.SetWindowText("Close")

        self.timerManager = TimerManager(self)
        self.ProcessArgs(sys.argv[self.argOff :])
        self.timerManager.go()
        return 1

    def OnDestroy(self, msg):
        dlgappcore.AppDialog.OnDestroy(self, msg)
        self.timerManager.stop()

    def OnOK(self):
        # stop the timer, then restart after setting special boolean
        self.timerManager.stop()
        self.timerManager.bConnectNow = 1
        self.timerManager.go()
        return


# 	def OnCancel(self): default behaviour - cancel == close.
# 		return


class TimerManager:
    def __init__(self, dlg):
        self.dlg = dlg
        self.timerId = None
        self.intervaler = eval(self.dlg.buildTimer)
        self.bConnectNow = 0
        self.bHaveSetPrompt1 = 0

    def CaptureOutput(self):
        self.oldOut = sys.stdout
        self.oldErr = sys.stderr
        sys.stdout = sys.stderr = self
        self.bHaveSetPrompt1 = 0

    def ReleaseOutput(self):
        sys.stdout = self.oldOut
        sys.stderr = self.oldErr

    def write(self, str):
        s = str.strip()
        if len(s):
            if self.bHaveSetPrompt1:
                dest = self.dlg.prompt3
            else:
                dest = self.dlg.prompt1
                self.bHaveSetPrompt1 = 1
            dest.SetWindowText(s)

    def go(self):
        self.OnTimer(None, None)

    def stop(self):
        if self.timerId:
            timer.kill_timer(self.timerId)
        self.timerId = None

    def OnTimer(self, id, timeVal):
        if id:
            timer.kill_timer(id)
        if self.intervaler.IsTime() or self.bConnectNow:
            # do the work.
            try:
                self.dlg.SetWindowText(self.dlg.title + " - Working...")
                self.dlg.butOK.EnableWindow(0)
                self.dlg.butCancel.EnableWindow(0)
                self.CaptureOutput()
                try:
                    exec(self.dlg.doWork)
                    print("The last operation completed successfully.")
                except:
                    t, v, tb = sys.exc_info()
                    str = "Failed: %s: %s" % (t, repr(v))
                    print(str)
                    self.oldErr.write(str)
                    tb = None  # Prevent cycle
            finally:
                self.ReleaseOutput()
                self.dlg.butOK.EnableWindow()
                self.dlg.butCancel.EnableWindow()
                self.dlg.SetWindowText(self.dlg.title)
        else:
            now = time.time()
            nextTime = self.intervaler.GetNextTime()
            if nextTime:
                timeDiffSeconds = nextTime - now
                timeDiffMinutes = int(timeDiffSeconds / 60)
                timeDiffSeconds = timeDiffSeconds % 60
                timeDiffHours = int(timeDiffMinutes / 60)
                timeDiffMinutes = timeDiffMinutes % 60
                self.dlg.prompt1.SetWindowText(
                    "Next connection due in %02d:%02d:%02d"
                    % (timeDiffHours, timeDiffMinutes, timeDiffSeconds)
                )
        self.timerId = timer.set_timer(
            self.intervaler.GetWakeupInterval(), self.OnTimer
        )
        self.bConnectNow = 0


class TimerIntervaler:
    def __init__(self):
        self.nextTime = None
        self.wakeUpInterval = 2000

    def GetWakeupInterval(self):
        return self.wakeUpInterval

    def GetNextTime(self):
        return self.nextTime

    def IsTime(self):
        now = time.time()
        if self.nextTime is None:
            self.nextTime = self.SetFirstTime(now)
        ret = 0
        if now >= self.nextTime:
            ret = 1
            self.nextTime = self.SetNextTime(self.nextTime, now)
            # do the work.
        return ret


class EachAnyIntervaler(TimerIntervaler):
    def __init__(self, timeAt, timePos, timeAdd, wakeUpInterval=None):
        TimerIntervaler.__init__(self)
        self.timeAt = timeAt
        self.timePos = timePos
        self.timeAdd = timeAdd
        if wakeUpInterval:
            self.wakeUpInterval = wakeUpInterval

    def SetFirstTime(self, now):
        timeTup = time.localtime(now)
        lst = []
        for item in timeTup:
            lst.append(item)
        bAdd = timeTup[self.timePos] > self.timeAt
        lst[self.timePos] = self.timeAt
        for pos in range(self.timePos + 1, 6):
            lst[pos] = 0
        ret = time.mktime(tuple(lst))
        if bAdd:
            ret = ret + self.timeAdd
        return ret

    def SetNextTime(self, lastTime, now):
        return lastTime + self.timeAdd


class EachMinuteIntervaler(EachAnyIntervaler):
    def __init__(self, at=0):
        EachAnyIntervaler.__init__(self, at, 5, 60, 2000)


class EachHourIntervaler(EachAnyIntervaler):
    def __init__(self, at=0):
        EachAnyIntervaler.__init__(self, at, 4, 3600, 10000)


class EachDayIntervaler(EachAnyIntervaler):
    def __init__(self, at=0):
        EachAnyIntervaler.__init__(self, at, 3, 86400, 10000)


class TimerDialogApp(dlgappcore.DialogApp):
    def CreateDialog(self):
        return TimerAppDialog()


def DoDemoWork():
    print("Doing the work...")
    print("About to connect")
    win32api.MessageBeep(win32con.MB_ICONASTERISK)
    win32api.Sleep(2000)
    print("Doing something else...")
    win32api.MessageBeep(win32con.MB_ICONEXCLAMATION)
    win32api.Sleep(2000)
    print("More work.")
    win32api.MessageBeep(win32con.MB_ICONHAND)
    win32api.Sleep(2000)
    print("The last bit.")
    win32api.MessageBeep(win32con.MB_OK)
    win32api.Sleep(2000)


app = TimerDialogApp()


def t():
    t = TimerAppDialog("Test Dialog")
    t.DoModal()
    return t


if __name__ == "__main__":
    import demoutils

    demoutils.NeedApp()
