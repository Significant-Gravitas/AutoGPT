# A sample originally provided by Richard Bell, and modified by Mark Hammond.

# This sample demonstrates how to use COM events in a free-threaded world.
# In this world, there is no need to marshall calls across threads, so
# no message loops are needed at all.  This means regular cross-thread
# sychronization can be used.  In this sample we just wait on win32 event
# objects.

# See also ieEventsApartmentThreaded.py for how to do this in an
# aparment-threaded world, where thread-marshalling complicates things.

# NOTE: This example uses Internet Explorer, but it should not be considerd
# a "best-practices" for writing against IE events, but for working with
# events in general. For example:
# * The first OnDocumentComplete event is not a reliable indicator that the
#   URL has completed loading
# * As we are demonstrating the most efficient way of handling events, when
#   running this sample you will see an IE Windows briefly appear, but
#   vanish without ever being repainted.

import sys

sys.coinit_flags = 0  # specify free threading


import pythoncom
import win32api
import win32com.client
import win32event


# The print statements indicate that COM has actually started another thread
# and will deliver the events to that thread (ie, the events do not actually
# fire on our main thread.
class ExplorerEvents:
    def __init__(self):
        # We reuse this event for all events.
        self.event = win32event.CreateEvent(None, 0, 0, None)

    def OnDocumentComplete(self, pDisp=pythoncom.Empty, URL=pythoncom.Empty):
        #
        # Caution:  Since the main thread and events thread(s) are different
        # it may be necessary to serialize access to shared data.  Because
        # this is a simple test case, that is not required here.  Your
        # situation may be different.   Caveat programmer.
        #
        thread = win32api.GetCurrentThreadId()
        print("OnDocumentComplete event processed on thread %d" % thread)
        # Set the event our main thread is waiting on.
        win32event.SetEvent(self.event)

    def OnQuit(self):
        thread = win32api.GetCurrentThreadId()
        print("OnQuit event processed on thread %d" % thread)
        win32event.SetEvent(self.event)


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

    # In this free-threaded example, we can simply wait until an event has
    # been set - we will give it 2 seconds before giving up.
    rc = win32event.WaitForSingleObject(iexplore.event, 2000)
    if rc != win32event.WAIT_OBJECT_0:
        print("Document load event FAILED to fire!!!")

    iexplore.Quit()
    # Now we can do the same thing to wait for exit!
    # Although Quit generates events, in this free-threaded world we
    # do *not* need to run any message pumps.

    rc = win32event.WaitForSingleObject(iexplore.event, 2000)
    if rc != win32event.WAIT_OBJECT_0:
        print("OnQuit event FAILED to fire!!!")

    iexplore = None
    print("Finished the IE event sample!")


if __name__ == "__main__":
    TestExplorerEvents()
