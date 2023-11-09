"""Testing pasing object between multiple COM threads

Uses standard COM marshalling to pass objects between threads.  Even 
though Python generally seems to work when you just pass COM objects
between threads, it shouldnt.

This shows the "correct" way to do it.

It shows that although we create new threads to use the Python.Interpreter,
COM marshalls back all calls to that object to the main Python thread,
which must be running a message loop (as this sample does).

When this test is run in "free threaded" mode (at this stage, you must 
manually mark the COM objects as "ThreadingModel=Free", or run from a 
service which has marked itself as free-threaded), then no marshalling
is done, and the Python.Interpreter object start doing the "expected" thing
- ie, it reports being on the same thread as its caller!

Python.exe needs a good way to mark itself as FreeThreaded - at the moment
this is a pain in the but!

"""

import _thread
import traceback

import pythoncom
import win32api
import win32com.client
import win32event


def TestInterp(interp):
    if interp.Eval("1+1") != 2:
        raise ValueError("The interpreter returned the wrong result.")
    try:
        interp.Eval(1 + 1)
        raise ValueError("The interpreter did not raise an exception")
    except pythoncom.com_error as details:
        import winerror

        if details[0] != winerror.DISP_E_TYPEMISMATCH:
            raise ValueError(
                "The interpreter exception was not winerror.DISP_E_TYPEMISMATCH."
            )


def TestInterpInThread(stopEvent, cookie):
    try:
        DoTestInterpInThread(cookie)
    finally:
        win32event.SetEvent(stopEvent)


def CreateGIT():
    return pythoncom.CoCreateInstance(
        pythoncom.CLSID_StdGlobalInterfaceTable,
        None,
        pythoncom.CLSCTX_INPROC,
        pythoncom.IID_IGlobalInterfaceTable,
    )


def DoTestInterpInThread(cookie):
    try:
        pythoncom.CoInitialize()
        myThread = win32api.GetCurrentThreadId()
        GIT = CreateGIT()

        interp = GIT.GetInterfaceFromGlobal(cookie, pythoncom.IID_IDispatch)
        interp = win32com.client.Dispatch(interp)

        TestInterp(interp)
        interp.Exec("import win32api")
        print(
            "The test thread id is %d, Python.Interpreter's thread ID is %d"
            % (myThread, interp.Eval("win32api.GetCurrentThreadId()"))
        )
        interp = None
        pythoncom.CoUninitialize()
    except:
        traceback.print_exc()


def BeginThreadsSimpleMarshal(numThreads, cookie):
    """Creates multiple threads using simple (but slower) marshalling.

    Single interpreter object, but a new stream is created per thread.

    Returns the handles the threads will set when complete.
    """
    ret = []
    for i in range(numThreads):
        hEvent = win32event.CreateEvent(None, 0, 0, None)
        _thread.start_new(TestInterpInThread, (hEvent, cookie))
        ret.append(hEvent)
    return ret


def test(fn):
    print("The main thread is %d" % (win32api.GetCurrentThreadId()))
    GIT = CreateGIT()
    interp = win32com.client.Dispatch("Python.Interpreter")
    cookie = GIT.RegisterInterfaceInGlobal(interp._oleobj_, pythoncom.IID_IDispatch)

    events = fn(4, cookie)
    numFinished = 0
    while 1:
        try:
            rc = win32event.MsgWaitForMultipleObjects(
                events, 0, 2000, win32event.QS_ALLINPUT
            )
            if rc >= win32event.WAIT_OBJECT_0 and rc < win32event.WAIT_OBJECT_0 + len(
                events
            ):
                numFinished = numFinished + 1
                if numFinished >= len(events):
                    break
            elif rc == win32event.WAIT_OBJECT_0 + len(events):  # a message
                # This is critical - whole apartment model demo will hang.
                pythoncom.PumpWaitingMessages()
            else:  # Timeout
                print(
                    "Waiting for thread to stop with interfaces=%d, gateways=%d"
                    % (pythoncom._GetInterfaceCount(), pythoncom._GetGatewayCount())
                )
        except KeyboardInterrupt:
            break
    GIT.RevokeInterfaceFromGlobal(cookie)
    del interp
    del GIT


if __name__ == "__main__":
    test(BeginThreadsSimpleMarshal)
    win32api.Sleep(500)
    # Doing CoUninit here stop Pythoncom.dll hanging when DLLMain shuts-down the process
    pythoncom.CoUninitialize()
    if pythoncom._GetInterfaceCount() != 0 or pythoncom._GetGatewayCount() != 0:
        print(
            "Done with interfaces=%d, gateways=%d"
            % (pythoncom._GetInterfaceCount(), pythoncom._GetGatewayCount())
        )
    else:
        print("Done.")
