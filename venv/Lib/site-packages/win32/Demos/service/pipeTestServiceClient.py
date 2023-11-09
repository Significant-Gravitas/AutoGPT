# A Test Program for pipeTestService.py
#
# Install and start the Pipe Test service, then run this test
# either from the same machine, or from another using the "-s" param.
#
# Eg: pipeTestServiceClient.py -s server_name Hi There
# Should work.

import os
import sys
import traceback

import pywintypes
import win32api
import winerror
from win32event import *
from win32file import *
from win32pipe import *

verbose = 0

# def ReadFromPipe(pipeName):
# Could (Should?) use CallNamedPipe, but this technique allows variable size
# messages (whereas you must supply a buffer size for CallNamedPipe!
#       hPipe = CreateFile(pipeName, GENERIC_WRITE, 0, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0)
#       more = 1
#       while more:
#               hr = ReadFile(hPipe, 256)
#               if hr==0:
#                       more = 0
#               except win32api.error (hr, fn, desc):
#                       if hr==winerror.ERROR_MORE_DATA:
#                               data = dat
#


def CallPipe(fn, args):
    ret = None
    retryCount = 0
    while retryCount < 8:  # Keep looping until user cancels.
        retryCount = retryCount + 1
        try:
            return fn(*args)
        except win32api.error as exc:
            if exc.winerror == winerror.ERROR_PIPE_BUSY:
                win32api.Sleep(5000)
                continue
            else:
                raise

    raise RuntimeError("Could not make a connection to the server")


def testClient(server, msg):
    if verbose:
        print("Sending", msg)
    data = CallPipe(
        CallNamedPipe,
        ("\\\\%s\\pipe\\PyPipeTest" % server, msg, 256, NMPWAIT_WAIT_FOREVER),
    )
    if verbose:
        print("Server sent back '%s'" % data)
    print("Sent and received a message!")


def testLargeMessage(server, size=4096):
    if verbose:
        print("Sending message of size %d" % (size))
    msg = "*" * size
    data = CallPipe(
        CallNamedPipe,
        ("\\\\%s\\pipe\\PyPipeTest" % server, msg, 512, NMPWAIT_WAIT_FOREVER),
    )
    if len(data) - size:
        print("Sizes are all wrong - send %d, got back %d" % (size, len(data)))


def stressThread(server, numMessages, wait):
    try:
        try:
            for i in range(numMessages):
                r = CallPipe(
                    CallNamedPipe,
                    (
                        "\\\\%s\\pipe\\PyPipeTest" % server,
                        "#" * 512,
                        1024,
                        NMPWAIT_WAIT_FOREVER,
                    ),
                )
        except:
            traceback.print_exc()
            print("Failed after %d messages" % i)
    finally:
        SetEvent(wait)


def stressTestClient(server, numThreads, numMessages):
    import _thread

    thread_waits = []
    for t_num in range(numThreads):
        # Note I could just wait on thread handles (after calling DuplicateHandle)
        # See the service itself for an example of waiting for the clients...
        wait = CreateEvent(None, 0, 0, None)
        thread_waits.append(wait)
        _thread.start_new_thread(stressThread, (server, numMessages, wait))
    # Wait for all threads to finish.
    WaitForMultipleObjects(thread_waits, 1, INFINITE)


def main():
    import getopt
    import sys

    server = "."
    thread_count = 0
    msg_count = 500
    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:t:m:vl")
        for o, a in opts:
            if o == "-s":
                server = a
            if o == "-m":
                msg_count = int(a)
            if o == "-t":
                thread_count = int(a)
            if o == "-v":
                global verbose
                verbose = 1
            if o == "-l":
                testLargeMessage(server)
        msg = " ".join(args).encode("mbcs")
    except getopt.error as msg:
        print(msg)
        my_name = os.path.split(sys.argv[0])[1]
        print(
            "Usage: %s [-v] [-s server] [-t thread_count=0] [-m msg_count=500] msg ..."
            % my_name
        )
        print("       -v = verbose")
        print(
            "       Specifying a value for -t will stress test using that many threads."
        )
        return
    testClient(server, msg)
    if thread_count > 0:
        print(
            "Spawning %d threads each sending %d messages..."
            % (thread_count, msg_count)
        )
        stressTestClient(server, thread_count, msg_count)


if __name__ == "__main__":
    main()
