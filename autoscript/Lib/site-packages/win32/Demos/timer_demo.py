# -*- Mode: Python; tab-width: 4 -*-
#

# This module, and the timer.pyd core timer support, were written by
# Sam Rushing (rushing@nightmare.com)

import time

# Timers are based on Windows messages.  So we need
# to do the event-loop thing!
import timer
import win32event
import win32gui

# glork holds a simple counter for us.


class glork:
    def __init__(self, delay=1000, max=10):
        self.x = 0
        self.max = max
        self.id = timer.set_timer(delay, self.increment)
        # Could use the threading module, but this is
        # a win32 extension test after all! :-)
        self.event = win32event.CreateEvent(None, 0, 0, None)

    def increment(self, id, time):
        print("x = %d" % self.x)
        self.x = self.x + 1
        # if we've reached the max count,
        # kill off the timer.
        if self.x > self.max:
            # we could have used 'self.id' here, too
            timer.kill_timer(id)
            win32event.SetEvent(self.event)


# create a counter that will count from '1' thru '10', incrementing
# once a second, and then stop.


def demo(delay=1000, stop=10):
    g = glork(delay, stop)
    # Timers are message based - so we need
    # To run a message loop while waiting for our timers
    # to expire.
    start_time = time.time()
    while 1:
        # We can't simply give a timeout of 30 seconds, as
        # we may continouusly be recieving other input messages,
        # and therefore never expire.
        rc = win32event.MsgWaitForMultipleObjects(
            (g.event,),  # list of objects
            0,  # wait all
            500,  # timeout
            win32event.QS_ALLEVENTS,  # type of input
        )
        if rc == win32event.WAIT_OBJECT_0:
            # Event signalled.
            break
        elif rc == win32event.WAIT_OBJECT_0 + 1:
            # Message waiting.
            if win32gui.PumpWaitingMessages():
                raise RuntimeError("We got an unexpected WM_QUIT message!")
        else:
            # This wait timed-out.
            if time.time() - start_time > 30:
                raise RuntimeError("We timed out waiting for the timers to expire!")


if __name__ == "__main__":
    demo()
