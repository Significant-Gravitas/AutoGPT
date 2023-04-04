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

import threading
import unittest

import pythoncom
import win32api
import win32com.client
import win32event

from .testServers import InterpCase

freeThreaded = 1


class ThreadInterpCase(InterpCase):
    def _testInterpInThread(self, stopEvent, interp):
        try:
            self._doTestInThread(interp)
        finally:
            win32event.SetEvent(stopEvent)

    def _doTestInThread(self, interp):
        pythoncom.CoInitialize()
        myThread = win32api.GetCurrentThreadId()

        if freeThreaded:
            interp = pythoncom.CoGetInterfaceAndReleaseStream(
                interp, pythoncom.IID_IDispatch
            )
            interp = win32com.client.Dispatch(interp)

        interp.Exec("import win32api")
        # print "The test thread id is %d, Python.Interpreter's thread ID is %d" % (myThread, interp.Eval("win32api.GetCurrentThreadId()"))
        pythoncom.CoUninitialize()

    def BeginThreadsSimpleMarshal(self, numThreads):
        """Creates multiple threads using simple (but slower) marshalling.

        Single interpreter object, but a new stream is created per thread.

        Returns the handles the threads will set when complete.
        """
        interp = win32com.client.Dispatch("Python.Interpreter")
        events = []
        threads = []
        for i in range(numThreads):
            hEvent = win32event.CreateEvent(None, 0, 0, None)
            events.append(hEvent)
            interpStream = pythoncom.CoMarshalInterThreadInterfaceInStream(
                pythoncom.IID_IDispatch, interp._oleobj_
            )
            t = threading.Thread(
                target=self._testInterpInThread, args=(hEvent, interpStream)
            )
            t.setDaemon(1)  # so errors dont cause shutdown hang
            t.start()
            threads.append(t)
        interp = None
        return threads, events

    #
    # NOTE - this doesnt quite work - Im not even sure it should, but Greg reckons
    # you should be able to avoid the marshal per thread!
    # I think that refers to CoMarshalInterface though...
    def BeginThreadsFastMarshal(self, numThreads):
        """Creates multiple threads using fast (but complex) marshalling.

        The marshal stream is created once, and each thread uses the same stream

        Returns the handles the threads will set when complete.
        """
        interp = win32com.client.Dispatch("Python.Interpreter")
        if freeThreaded:
            interp = pythoncom.CoMarshalInterThreadInterfaceInStream(
                pythoncom.IID_IDispatch, interp._oleobj_
            )
        events = []
        threads = []
        for i in range(numThreads):
            hEvent = win32event.CreateEvent(None, 0, 0, None)
            t = threading.Thread(target=self._testInterpInThread, args=(hEvent, interp))
            t.setDaemon(1)  # so errors dont cause shutdown hang
            t.start()
            events.append(hEvent)
            threads.append(t)
        return threads, events

    def _DoTestMarshal(self, fn, bCoWait=0):
        # print "The main thread is %d" % (win32api.GetCurrentThreadId())
        threads, events = fn(2)
        numFinished = 0
        while 1:
            try:
                if bCoWait:
                    rc = pythoncom.CoWaitForMultipleHandles(0, 2000, events)
                else:
                    # Specifying "bWaitAll" here will wait for messages *and* all events
                    # (which is pretty useless)
                    rc = win32event.MsgWaitForMultipleObjects(
                        events, 0, 2000, win32event.QS_ALLINPUT
                    )
                if (
                    rc >= win32event.WAIT_OBJECT_0
                    and rc < win32event.WAIT_OBJECT_0 + len(events)
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
        for t in threads:
            t.join(2)
            self.assertFalse(t.is_alive(), "thread failed to stop!?")
        threads = None  # threads hold references to args
        # Seems to be a leak here I can't locate :(
        # self.assertEqual(pythoncom._GetInterfaceCount(), 0)
        # self.assertEqual(pythoncom._GetGatewayCount(), 0)

    def testSimpleMarshal(self):
        self._DoTestMarshal(self.BeginThreadsSimpleMarshal)

    def testSimpleMarshalCoWait(self):
        self._DoTestMarshal(self.BeginThreadsSimpleMarshal, 1)


#    def testFastMarshal(self):
#        self._DoTestMarshal(self.BeginThreadsFastMarshal)

if __name__ == "__main__":
    unittest.main("testMarshal")
