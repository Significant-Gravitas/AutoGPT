import unittest

import pywintypes
import win32event


class TestWaitableTimer(unittest.TestCase):
    def testWaitableFire(self):
        h = win32event.CreateWaitableTimer(None, 0, None)
        dt = -160  # 160 ns.
        win32event.SetWaitableTimer(h, dt, 0, None, None, 0)
        rc = win32event.WaitForSingleObject(h, 1000)
        self.assertEqual(rc, win32event.WAIT_OBJECT_0)

    def testCreateWaitableTimerEx(self):
        h = win32event.CreateWaitableTimerEx(
            None,
            None,
            win32event.CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            win32event.TIMER_ALL_ACCESS,
        )
        dt = -160  # 160 ns.
        win32event.SetWaitableTimer(h, dt, 0, None, None, 0)
        rc = win32event.WaitForSingleObject(h, 1000)
        self.assertEqual(rc, win32event.WAIT_OBJECT_0)

    def testWaitableTrigger(self):
        h = win32event.CreateWaitableTimer(None, 0, None)
        # for the sake of this, pass a long that doesn't fit in an int.
        dt = -2000000000
        win32event.SetWaitableTimer(h, dt, 0, None, None, 0)
        rc = win32event.WaitForSingleObject(h, 10)  # 10 ms.
        self.assertEqual(rc, win32event.WAIT_TIMEOUT)

    def testWaitableError(self):
        h = win32event.CreateWaitableTimer(None, 0, None)
        h.close()
        self.assertRaises(
            pywintypes.error, win32event.SetWaitableTimer, h, -42, 0, None, None, 0
        )


class TestWaitFunctions(unittest.TestCase):
    def testMsgWaitForMultipleObjects(self):
        # this function used to segfault when called with an empty list
        res = win32event.MsgWaitForMultipleObjects([], 0, 0, 0)
        self.assertEqual(res, win32event.WAIT_TIMEOUT)

    def testMsgWaitForMultipleObjects2(self):
        # test with non-empty list
        event = win32event.CreateEvent(None, 0, 0, None)
        res = win32event.MsgWaitForMultipleObjects([event], 0, 0, 0)
        self.assertEqual(res, win32event.WAIT_TIMEOUT)

    def testMsgWaitForMultipleObjectsEx(self):
        # this function used to segfault when called with an empty list
        res = win32event.MsgWaitForMultipleObjectsEx([], 0, 0, 0)
        self.assertEqual(res, win32event.WAIT_TIMEOUT)

    def testMsgWaitForMultipleObjectsEx2(self):
        # test with non-empty list
        event = win32event.CreateEvent(None, 0, 0, None)
        res = win32event.MsgWaitForMultipleObjectsEx([event], 0, 0, 0)
        self.assertEqual(res, win32event.WAIT_TIMEOUT)


class TestEvent(unittest.TestCase):
    def assertSignaled(self, event):
        self.assertEqual(
            win32event.WaitForSingleObject(event, 0), win32event.WAIT_OBJECT_0
        )

    def assertNotSignaled(self, event):
        self.assertEqual(
            win32event.WaitForSingleObject(event, 0), win32event.WAIT_TIMEOUT
        )

    def testCreateEvent(self):
        event = win32event.CreateEvent(None, False, False, None)
        self.assertNotSignaled(event)
        event = win32event.CreateEvent(None, False, True, None)
        self.assertSignaled(event)
        self.assertNotSignaled(event)
        event = win32event.CreateEvent(None, True, True, None)
        self.assertSignaled(event)
        self.assertSignaled(event)

    def testSetEvent(self):
        event = win32event.CreateEvent(None, True, False, None)
        self.assertNotSignaled(event)
        res = win32event.SetEvent(event)
        self.assertEqual(res, None)
        self.assertSignaled(event)
        event.close()
        self.assertRaises(pywintypes.error, win32event.SetEvent, event)

    def testResetEvent(self):
        event = win32event.CreateEvent(None, True, True, None)
        self.assertSignaled(event)
        res = win32event.ResetEvent(event)
        self.assertEqual(res, None)
        self.assertNotSignaled(event)
        event.close()
        self.assertRaises(pywintypes.error, win32event.ResetEvent, event)


class TestMutex(unittest.TestCase):
    def testReleaseMutex(self):
        mutex = win32event.CreateMutex(None, True, None)
        res = win32event.ReleaseMutex(mutex)
        self.assertEqual(res, None)
        res = win32event.WaitForSingleObject(mutex, 0)
        self.assertEqual(res, win32event.WAIT_OBJECT_0)
        mutex.close()
        self.assertRaises(pywintypes.error, win32event.ReleaseMutex, mutex)


if __name__ == "__main__":
    unittest.main()
