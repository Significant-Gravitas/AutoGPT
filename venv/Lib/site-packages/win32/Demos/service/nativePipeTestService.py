# This is an example of a service hosted by python.exe rather than
# pythonservice.exe.

# Note that it is very rare that using python.exe is a better option
# than the default pythonservice.exe - the latter has better error handling
# so that if Python itself can't be initialized or there are very early
# import errors, you will get error details written to the event log.  When
# using python.exe instead, you are forced to wait for the interpreter startup
# and imports to succeed before you are able to effectively setup your own
# error handling.

# So in short, please make sure you *really* want to do this, otherwise just
# stick with the default.

import os
import sys

import servicemanager
import win32serviceutil
from pipeTestService import TestPipeService


class NativeTestPipeService(TestPipeService):
    _svc_name_ = "PyNativePipeTestService"
    _svc_display_name_ = "Python Native Pipe Test Service"
    _svc_description_ = "Tests Python.exe hosted services"
    # tell win32serviceutil we have a custom executable and custom args
    # so registration does the right thing.
    _exe_name_ = sys.executable
    _exe_args_ = '"' + os.path.abspath(sys.argv[0]) + '"'


def main():
    if len(sys.argv) == 1:
        # service must be starting...
        print("service is starting...")
        print("(execute this script with '--help' if that isn't what you want)")

        # for the sake of debugging etc, we use win32traceutil to see
        # any unhandled exceptions and print statements.
        import win32traceutil

        print("service is still starting...")

        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(NativeTestPipeService)
        # Now ask the service manager to fire things up for us...
        servicemanager.StartServiceCtrlDispatcher()
        print("service done!")
    else:
        win32serviceutil.HandleCommandLine(NativeTestPipeService)


if __name__ == "__main__":
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except:
        print("Something went bad!")
        import traceback

        traceback.print_exc()
