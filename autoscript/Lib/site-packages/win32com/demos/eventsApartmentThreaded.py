# A sample originally provided by Richard Bell, and modified by Mark Hammond.

# This sample demonstrates how to use COM events in an aparment-threaded
# world.  In this world, COM itself ensures that all calls to and events
# from an object happen on the same thread that created the object, even
# if they originated from different threads.  For this cross-thread
# marshalling to work, this main thread *must* run a "message-loop" (ie,
# a loop fetching and dispatching Windows messages).  Without such message
# processing, dead-locks can occur.

# See also eventsFreeThreaded.py for how to do this in a free-threaded
# world where these marshalling considerations do not exist.

# NOTE: This example uses Internet Explorer, but it should not be considerd
# a "best-practices" for writing against IE events, but for working with
# events in general. For example:
# * The first OnDocumentComplete event is not a reliable indicator that the
#   URL has completed loading
# * As we are demonstrating the most efficient way of handling events, when
#   running this sample you will see an IE Windows briefly appear, but
#   vanish without ever being repainted.

import time

# sys.coinit_flags not set, so pythoncom initializes apartment-threaded.
import pythoncom
import win32api
import win32com.client
import win32event


class ExplorerEvents:
    def __init__(self):
        self.event = win32event.CreateEvent(None, 0, 0, None)

    def OnDocumentComplete(self, pDisp=pythoncom.Empty, URL=pythoncom.Empty):
        thread = win32api.GetCurrentThreadId()
        print("OnDocumentComplete event processed on thread %d" % thread)
        # Set the event our main thread is waiting on.
        win32event.SetEvent(self.event)

    def OnQuit(self):
        thread = win32api.GetCurrentThreadId()
        print("OnQuit event processed on thread %d" % thread)
        win32event.SetEvent(self.event)


def WaitWhileProcessingMessages(event, timeout=2):
    start = time.perf_counter()
    while True:
        # Wake 4 times a second - we can't just specify the
        # full timeout here, as then it would reset for every
        # message we process.
        rc = win32event.MsgWaitForMultipleObjects(
            (event,), 0, 250, win32event.QS_ALLEVENTS
        )
        if rc == win32event.WAIT_OBJECT_0:
            # event signalled - stop now!
            return True
        if (time.perf_counter() - start) > timeout:
            # Timeout expired.
            return False
        # must be a message.
        pythoncom.PumpWaitingMessages()


def TestExplorerEvents():
    iexplore = win32com.client.DispatchWithEvents(
        "InternetExplorer.Application", ExplorerEvents
    )

    thread = win32api.GetCurrentThreadId()
    print("TestExplorerEvents created IE object on thread %d" % thread)

    iexplore.Visible = 1
    try:
        iexplore.Navigate(win32api.GetFullPathName("..\\readme.html"))
    except pythoncom.com_error as details:
        print("Warning - could not open the test HTML file", details)

    # Wait for the event to be signalled while pumping messages.
    if not WaitWhileProcessingMessages(iexplore.event):
        print("Document load event FAILED to fire!!!")

    iexplore.Quit()
    #
    # Give IE a chance to shutdown, else it can get upset on fast machines.
    # Note, Quit generates events.  Although this test does NOT catch them
    # it is NECESSARY to pump messages here instead of a sleep so that the Quit
    # happens properly!
    if not WaitWhileProcessingMessages(iexplore.event):
        print("OnQuit event FAILED to fire!!!")

    iexplore = None


if __name__ == "__main__":
    TestExplorerEvents()
