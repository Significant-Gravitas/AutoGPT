# This is a helper for the win32trace module

# If imported from a normal Python program, it sets up sys.stdout and sys.stderr
# so output goes to the collector.

# If run from the command line, it creates a collector loop.

# Eg:
# C:>start win32traceutil.py (or python.exe win32traceutil.py)
# will start a process with a (pretty much) blank screen.
#
# then, switch to a DOS prompt, and type:
# C:>python.exe
# Python 1.4 etc...
# >>> import win32traceutil
# Redirecting output to win32trace remote collector
# >>> print "Hello"
# >>>
# And the output will appear in the first collector process.

# Note - the client or the collector can be started first.
# There is a 0x20000 byte buffer.  If this gets full, it is reset, and new
# output appended from the start.

import win32trace


def RunAsCollector():
    import sys

    try:
        import win32api

        win32api.SetConsoleTitle("Python Trace Collector")
    except:
        pass  # Oh well!
    win32trace.InitRead()
    print("Collecting Python Trace Output...")
    try:
        while 1:
            # a short timeout means ctrl+c works next time we wake...
            sys.stdout.write(win32trace.blockingread(500))
    except KeyboardInterrupt:
        print("Ctrl+C")


def SetupForPrint():
    win32trace.InitWrite()
    try:  # Under certain servers, sys.stdout may be invalid.
        print("Redirecting output to win32trace remote collector")
    except:
        pass
    win32trace.setprint()  # this works in an rexec environment.


if __name__ == "__main__":
    RunAsCollector()
else:
    SetupForPrint()
