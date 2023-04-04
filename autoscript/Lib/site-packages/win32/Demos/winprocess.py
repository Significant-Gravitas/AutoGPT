"""
Windows Process Control

winprocess.run launches a child process and returns the exit code.
Optionally, it can:
  redirect stdin, stdout & stderr to files
  run the command as another user
  limit the process's running time
  control the process window (location, size, window state, desktop)
Works on Windows NT, 2000 & XP. Requires Mark Hammond's win32
extensions.

This code is free for any purpose, with no warranty of any kind.
-- John B. Dell'Aquila <jbd@alum.mit.edu>
"""

import msvcrt
import os

import win32api
import win32con
import win32event
import win32gui
import win32process
import win32security


def logonUser(loginString):
    """
    Login as specified user and return handle.
    loginString:  'Domain\nUser\nPassword'; for local
        login use . or empty string as domain
        e.g. '.\nadministrator\nsecret_password'
    """
    domain, user, passwd = loginString.split("\n")
    return win32security.LogonUser(
        user,
        domain,
        passwd,
        win32con.LOGON32_LOGON_INTERACTIVE,
        win32con.LOGON32_PROVIDER_DEFAULT,
    )


class Process:
    """
    A Windows process.
    """

    def __init__(
        self,
        cmd,
        login=None,
        hStdin=None,
        hStdout=None,
        hStderr=None,
        show=1,
        xy=None,
        xySize=None,
        desktop=None,
    ):
        """
        Create a Windows process.
        cmd:     command to run
        login:   run as user 'Domain\nUser\nPassword'
        hStdin, hStdout, hStderr:
                 handles for process I/O; default is caller's stdin,
                 stdout & stderr
        show:    wShowWindow (0=SW_HIDE, 1=SW_NORMAL, ...)
        xy:      window offset (x, y) of upper left corner in pixels
        xySize:  window size (width, height) in pixels
        desktop: lpDesktop - name of desktop e.g. 'winsta0\\default'
                 None = inherit current desktop
                 '' = create new desktop if necessary

        User calling login requires additional privileges:
          Act as part of the operating system [not needed on Windows XP]
          Increase quotas
          Replace a process level token
        Login string must EITHER be an administrator's account
        (ordinary user can't access current desktop - see Microsoft
        Q165194) OR use desktop='' to run another desktop invisibly
        (may be very slow to startup & finalize).
        """
        si = win32process.STARTUPINFO()
        si.dwFlags = win32con.STARTF_USESTDHANDLES ^ win32con.STARTF_USESHOWWINDOW
        if hStdin is None:
            si.hStdInput = win32api.GetStdHandle(win32api.STD_INPUT_HANDLE)
        else:
            si.hStdInput = hStdin
        if hStdout is None:
            si.hStdOutput = win32api.GetStdHandle(win32api.STD_OUTPUT_HANDLE)
        else:
            si.hStdOutput = hStdout
        if hStderr is None:
            si.hStdError = win32api.GetStdHandle(win32api.STD_ERROR_HANDLE)
        else:
            si.hStdError = hStderr
        si.wShowWindow = show
        if xy is not None:
            si.dwX, si.dwY = xy
            si.dwFlags ^= win32con.STARTF_USEPOSITION
        if xySize is not None:
            si.dwXSize, si.dwYSize = xySize
            si.dwFlags ^= win32con.STARTF_USESIZE
        if desktop is not None:
            si.lpDesktop = desktop
        procArgs = (
            None,  # appName
            cmd,  # commandLine
            None,  # processAttributes
            None,  # threadAttributes
            1,  # bInheritHandles
            win32process.CREATE_NEW_CONSOLE,  # dwCreationFlags
            None,  # newEnvironment
            None,  # currentDirectory
            si,
        )  # startupinfo
        if login is not None:
            hUser = logonUser(login)
            win32security.ImpersonateLoggedOnUser(hUser)
            procHandles = win32process.CreateProcessAsUser(hUser, *procArgs)
            win32security.RevertToSelf()
        else:
            procHandles = win32process.CreateProcess(*procArgs)
        self.hProcess, self.hThread, self.PId, self.TId = procHandles

    def wait(self, mSec=None):
        """
        Wait for process to finish or for specified number of
        milliseconds to elapse.
        """
        if mSec is None:
            mSec = win32event.INFINITE
        return win32event.WaitForSingleObject(self.hProcess, mSec)

    def kill(self, gracePeriod=5000):
        """
        Kill process. Try for an orderly shutdown via WM_CLOSE.  If
        still running after gracePeriod (5 sec. default), terminate.
        """
        win32gui.EnumWindows(self.__close__, 0)
        if self.wait(gracePeriod) != win32event.WAIT_OBJECT_0:
            win32process.TerminateProcess(self.hProcess, 0)
            win32api.Sleep(100)  # wait for resources to be released

    def __close__(self, hwnd, dummy):
        """
        EnumWindows callback - sends WM_CLOSE to any window
        owned by this process.
        """
        TId, PId = win32process.GetWindowThreadProcessId(hwnd)
        if PId == self.PId:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

    def exitCode(self):
        """
        Return process exit code.
        """
        return win32process.GetExitCodeProcess(self.hProcess)


def run(cmd, mSec=None, stdin=None, stdout=None, stderr=None, **kw):
    """
    Run cmd as a child process and return exit code.
    mSec:  terminate cmd after specified number of milliseconds
    stdin, stdout, stderr:
           file objects for child I/O (use hStdin etc. to attach
           handles instead of files); default is caller's stdin,
           stdout & stderr;
    kw:    see Process.__init__ for more keyword options
    """
    if stdin is not None:
        kw["hStdin"] = msvcrt.get_osfhandle(stdin.fileno())
    if stdout is not None:
        kw["hStdout"] = msvcrt.get_osfhandle(stdout.fileno())
    if stderr is not None:
        kw["hStderr"] = msvcrt.get_osfhandle(stderr.fileno())
    child = Process(cmd, **kw)
    if child.wait(mSec) != win32event.WAIT_OBJECT_0:
        child.kill()
        raise WindowsError("process timeout exceeded")
    return child.exitCode()


if __name__ == "__main__":
    # Pipe commands to a shell and display the output in notepad
    print("Testing winprocess.py...")

    import tempfile

    timeoutSeconds = 15
    cmdString = (
        """\
REM      Test of winprocess.py piping commands to a shell.\r
REM      This 'notepad' process will terminate in %d seconds.\r
vol\r
net user\r
_this_is_a_test_of_stderr_\r
"""
        % timeoutSeconds
    )

    cmd_name = tempfile.mktemp()
    out_name = cmd_name + ".txt"
    try:
        cmd = open(cmd_name, "w+b")
        out = open(out_name, "w+b")
        cmd.write(cmdString.encode("mbcs"))
        cmd.seek(0)
        print(
            "CMD.EXE exit code:",
            run("cmd.exe", show=0, stdin=cmd, stdout=out, stderr=out),
        )
        cmd.close()
        print(
            "NOTEPAD exit code:",
            run(
                "notepad.exe %s" % out.name,
                show=win32con.SW_MAXIMIZE,
                mSec=timeoutSeconds * 1000,
            ),
        )
        out.close()
    finally:
        for n in (cmd_name, out_name):
            try:
                os.unlink(cmd_name)
            except os.error:
                pass
