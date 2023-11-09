"""runproc.py

start a process with three inherited pipes.
Try to write to and read from those.
"""

import msvcrt
import os

import win32api
import win32con
import win32file
import win32pipe
import win32process
import win32security


class Process:
    def run(self, cmdline):
        # security attributes for pipes
        sAttrs = win32security.SECURITY_ATTRIBUTES()
        sAttrs.bInheritHandle = 1

        # create pipes
        hStdin_r, self.hStdin_w = win32pipe.CreatePipe(sAttrs, 0)
        self.hStdout_r, hStdout_w = win32pipe.CreatePipe(sAttrs, 0)
        self.hStderr_r, hStderr_w = win32pipe.CreatePipe(sAttrs, 0)

        # set the info structure for the new process.
        StartupInfo = win32process.STARTUPINFO()
        StartupInfo.hStdInput = hStdin_r
        StartupInfo.hStdOutput = hStdout_w
        StartupInfo.hStdError = hStderr_w
        StartupInfo.dwFlags = win32process.STARTF_USESTDHANDLES
        # Mark doesn't support wShowWindow yet.
        # StartupInfo.dwFlags = StartupInfo.dwFlags | win32process.STARTF_USESHOWWINDOW
        # StartupInfo.wShowWindow = win32con.SW_HIDE

        # Create new output read handles and the input write handle. Set
        # the inheritance properties to FALSE. Otherwise, the child inherits
        # the these handles; resulting in non-closeable handles to the pipes
        # being created.
        pid = win32api.GetCurrentProcess()

        tmp = win32api.DuplicateHandle(
            pid,
            self.hStdin_w,
            pid,
            0,
            0,  # non-inheritable!!
            win32con.DUPLICATE_SAME_ACCESS,
        )
        # Close the inhertible version of the handle
        win32file.CloseHandle(self.hStdin_w)
        self.hStdin_w = tmp
        tmp = win32api.DuplicateHandle(
            pid,
            self.hStdout_r,
            pid,
            0,
            0,  # non-inheritable!
            win32con.DUPLICATE_SAME_ACCESS,
        )
        # Close the inhertible version of the handle
        win32file.CloseHandle(self.hStdout_r)
        self.hStdout_r = tmp

        # start the process.
        hProcess, hThread, dwPid, dwTid = win32process.CreateProcess(
            None,  # program
            cmdline,  # command line
            None,  # process security attributes
            None,  # thread attributes
            1,  # inherit handles, or USESTDHANDLES won't work.
            # creation flags. Don't access the console.
            0,  # Don't need anything here.
            # If you're in a GUI app, you should use
            # CREATE_NEW_CONSOLE here, or any subprocesses
            # might fall victim to the problem described in:
            # KB article: Q156755, cmd.exe requires
            # an NT console in order to perform redirection..
            None,  # no new environment
            None,  # current directory (stay where we are)
            StartupInfo,
        )
        # normally, we would save the pid etc. here...

        # Child is launched. Close the parents copy of those pipe handles
        # that only the child should have open.
        # You need to make sure that no handles to the write end of the
        # output pipe are maintained in this process or else the pipe will
        # not close when the child process exits and the ReadFile will hang.
        win32file.CloseHandle(hStderr_w)
        win32file.CloseHandle(hStdout_w)
        win32file.CloseHandle(hStdin_r)

        self.stdin = os.fdopen(msvcrt.open_osfhandle(self.hStdin_w, 0), "wb")
        self.stdin.write("hmmmmm\r\n")
        self.stdin.flush()
        self.stdin.close()

        self.stdout = os.fdopen(msvcrt.open_osfhandle(self.hStdout_r, 0), "rb")
        print("Read on stdout: ", repr(self.stdout.read()))

        self.stderr = os.fdopen(msvcrt.open_osfhandle(self.hStderr_r, 0), "rb")
        print("Read on stderr: ", repr(self.stderr.read()))


if __name__ == "__main__":
    p = Process()
    exe = win32api.GetModuleFileName(0)
    p.run(exe + " cat.py")

# end of runproc.py
