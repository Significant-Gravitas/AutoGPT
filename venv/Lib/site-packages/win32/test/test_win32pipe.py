import threading
import time
import unittest

import pywintypes
import win32con
import win32event
import win32file
import win32pipe
import winerror
from pywin32_testutil import str2bytes  # py3k-friendly helper


class PipeTests(unittest.TestCase):
    pipename = "\\\\.\\pipe\\python_test_pipe"

    def _serverThread(self, pipe_handle, event, wait_time):
        # just do one connection and terminate.
        hr = win32pipe.ConnectNamedPipe(pipe_handle)
        self.assertTrue(
            hr in (0, winerror.ERROR_PIPE_CONNECTED), "Got error code 0x%x" % (hr,)
        )
        hr, got = win32file.ReadFile(pipe_handle, 100)
        self.assertEqual(got, str2bytes("foo\0bar"))
        time.sleep(wait_time)
        win32file.WriteFile(pipe_handle, str2bytes("bar\0foo"))
        pipe_handle.Close()
        event.set()

    def startPipeServer(self, event, wait_time=0):
        openMode = win32pipe.PIPE_ACCESS_DUPLEX
        pipeMode = win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT

        sa = pywintypes.SECURITY_ATTRIBUTES()
        sa.SetSecurityDescriptorDacl(1, None, 0)

        pipe_handle = win32pipe.CreateNamedPipe(
            self.pipename,
            openMode,
            pipeMode,
            win32pipe.PIPE_UNLIMITED_INSTANCES,
            0,
            0,
            2000,
            sa,
        )

        threading.Thread(
            target=self._serverThread, args=(pipe_handle, event, wait_time)
        ).start()

    def testCallNamedPipe(self):
        event = threading.Event()
        self.startPipeServer(event)

        got = win32pipe.CallNamedPipe(
            self.pipename, str2bytes("foo\0bar"), 1024, win32pipe.NMPWAIT_WAIT_FOREVER
        )
        self.assertEqual(got, str2bytes("bar\0foo"))
        event.wait(5)
        self.assertTrue(event.isSet(), "Pipe server thread didn't terminate")

    def testTransactNamedPipeBlocking(self):
        event = threading.Event()
        self.startPipeServer(event)
        open_mode = win32con.GENERIC_READ | win32con.GENERIC_WRITE

        hpipe = win32file.CreateFile(
            self.pipename,
            open_mode,
            0,  # no sharing
            None,  # default security
            win32con.OPEN_EXISTING,
            0,  # win32con.FILE_FLAG_OVERLAPPED,
            None,
        )

        # set to message mode.
        win32pipe.SetNamedPipeHandleState(
            hpipe, win32pipe.PIPE_READMODE_MESSAGE, None, None
        )

        hr, got = win32pipe.TransactNamedPipe(hpipe, str2bytes("foo\0bar"), 1024, None)
        self.assertEqual(got, str2bytes("bar\0foo"))
        event.wait(5)
        self.assertTrue(event.isSet(), "Pipe server thread didn't terminate")

    def testTransactNamedPipeBlockingBuffer(self):
        # Like testTransactNamedPipeBlocking, but a pre-allocated buffer is
        # passed (not really that useful, but it exercises the code path)
        event = threading.Event()
        self.startPipeServer(event)
        open_mode = win32con.GENERIC_READ | win32con.GENERIC_WRITE

        hpipe = win32file.CreateFile(
            self.pipename,
            open_mode,
            0,  # no sharing
            None,  # default security
            win32con.OPEN_EXISTING,
            0,  # win32con.FILE_FLAG_OVERLAPPED,
            None,
        )

        # set to message mode.
        win32pipe.SetNamedPipeHandleState(
            hpipe, win32pipe.PIPE_READMODE_MESSAGE, None, None
        )

        buffer = win32file.AllocateReadBuffer(1024)
        hr, got = win32pipe.TransactNamedPipe(
            hpipe, str2bytes("foo\0bar"), buffer, None
        )
        self.assertEqual(got, str2bytes("bar\0foo"))
        event.wait(5)
        self.assertTrue(event.isSet(), "Pipe server thread didn't terminate")

    def testTransactNamedPipeAsync(self):
        event = threading.Event()
        overlapped = pywintypes.OVERLAPPED()
        overlapped.hEvent = win32event.CreateEvent(None, 0, 0, None)
        self.startPipeServer(event, 0.5)
        open_mode = win32con.GENERIC_READ | win32con.GENERIC_WRITE

        hpipe = win32file.CreateFile(
            self.pipename,
            open_mode,
            0,  # no sharing
            None,  # default security
            win32con.OPEN_EXISTING,
            win32con.FILE_FLAG_OVERLAPPED,
            None,
        )

        # set to message mode.
        win32pipe.SetNamedPipeHandleState(
            hpipe, win32pipe.PIPE_READMODE_MESSAGE, None, None
        )

        buffer = win32file.AllocateReadBuffer(1024)
        hr, got = win32pipe.TransactNamedPipe(
            hpipe, str2bytes("foo\0bar"), buffer, overlapped
        )
        self.assertEqual(hr, winerror.ERROR_IO_PENDING)
        nbytes = win32file.GetOverlappedResult(hpipe, overlapped, True)
        got = buffer[:nbytes]
        self.assertEqual(got, str2bytes("bar\0foo"))
        event.wait(5)
        self.assertTrue(event.isSet(), "Pipe server thread didn't terminate")


if __name__ == "__main__":
    unittest.main()
