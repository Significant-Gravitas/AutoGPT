# cmdserver.py

# Demo code that is not Pythonwin related, but too good to throw away...

import _thread
import sys
import traceback

import win32api
from pywin.framework import winout


class ThreadWriter:
    "Assign an instance to sys.stdout for per-thread printing objects - Courtesy Guido!"

    def __init__(self):
        "Constructor -- initialize the table of writers"
        self.writers = {}
        self.origStdOut = None

    def register(self, writer):
        "Register the writer for the current thread"
        self.writers[_thread.get_ident()] = writer
        if self.origStdOut is None:
            self.origStdOut = sys.stdout
            sys.stdout = self

    def unregister(self):
        "Remove the writer for the current thread, if any"
        try:
            del self.writers[_thread.get_ident()]
        except KeyError:
            pass
        if len(self.writers) == 0:
            sys.stdout = self.origStdOut
            self.origStdOut = None

    def getwriter(self):
        "Return the current thread's writer, default sys.stdout"
        try:
            return self.writers[_thread.get_ident()]
        except KeyError:
            return self.origStdOut

    def write(self, str):
        "Write to the current thread's writer, default sys.stdout"
        self.getwriter().write(str)


def Test():
    num = 1
    while num < 1000:
        print("Hello there no " + str(num))
        win32api.Sleep(50)
        num = num + 1


class flags:
    SERVER_BEST = 0
    SERVER_IMMEDIATE = 1
    SERVER_THREAD = 2
    SERVER_PROCESS = 3


def StartServer(cmd, title=None, bCloseOnEnd=0, serverFlags=flags.SERVER_BEST):
    out = winout.WindowOutput(title, None, winout.flags.WQ_IDLE)
    if not title:
        title = cmd
    out.Create(title)
    # 	ServerThread((out, cmd, title, bCloseOnEnd))
    # 	out = sys.stdout
    _thread.start_new_thread(ServerThread, (out, cmd, title, bCloseOnEnd))


def ServerThread(myout, cmd, title, bCloseOnEnd):
    try:
        writer.register(myout)
        print('Executing "%s"\n' % cmd)
        bOK = 1
        try:
            import __main__

            exec(cmd + "\n", __main__.__dict__)
        except:
            bOK = 0
        if bOK:
            print("Command terminated without errors.")
        else:
            t, v, tb = sys.exc_info()
            print(t, ": ", v)
            traceback.print_tb(tb)
            tb = None  # prevent a cycle
            print("Command terminated with an unhandled exception")
        writer.unregister()
        if bOK and bCloseOnEnd:
            myout.frame.DestroyWindow()

    # Unhandled exception of any kind in a thread kills the gui!
    except:
        t, v, tb = sys.exc_info()
        print(t, ": ", v)
        traceback.print_tb(tb)
        tb = None
        print("Thread failed")


# assist for reloading (when debugging) - use only 1 tracer object,
# else a large chain of tracer objects will exist.
# try:
# 	writer
# except NameError:
# 	writer=ThreadWriter()
if __name__ == "__main__":
    import demoutils

    demoutils.NotAScript()
