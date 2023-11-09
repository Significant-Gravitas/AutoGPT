# This is a simple serial port terminal demo.
#
# Its primary purpose is to demonstrate the native serial port access offered via
# win32file.

# It uses 3 threads:
# - The main thread, which cranks up the other 2 threads, then simply waits for them to exit.
# - The user-input thread - blocks waiting for a keyboard character, and when found sends it
#   out the COM port.  If the character is Ctrl+C, it stops, signalling the COM port thread to stop.
# - The COM port thread is simply listening for input on the COM port, and prints it to the screen.

# This demo uses userlapped IO, so that none of the read or write operations actually block (however,
# in this sample, the very next thing we do _is_ block - so it shows off the concepts even though it
# doesnt exploit them.

import msvcrt  # For the getch() function.
import sys
import threading

import win32con  # constants.
from win32event import *  # We use events and the WaitFor[Multiple]Objects functions.
from win32file import *  # The base COM port and file IO functions.


def FindModem():
    # Snoop over the comports, seeing if it is likely we have a modem.
    for i in range(1, 5):
        port = "COM%d" % (i,)
        try:
            handle = CreateFile(
                port,
                win32con.GENERIC_READ | win32con.GENERIC_WRITE,
                0,  # exclusive access
                None,  # no security
                win32con.OPEN_EXISTING,
                win32con.FILE_ATTRIBUTE_NORMAL,
                None,
            )
            # It appears that an available COM port will always success here,
            # just return 0 for the status flags.  We only care that it has _any_ status
            # flags (and therefore probably a real modem)
            if GetCommModemStatus(handle) != 0:
                return port
        except error:
            pass  # No port, or modem status failed.
    return None


# A basic synchronous COM port file-like object
class SerialTTY:
    def __init__(self, port):
        if type(port) == type(0):
            port = "COM%d" % (port,)
        self.handle = CreateFile(
            port,
            win32con.GENERIC_READ | win32con.GENERIC_WRITE,
            0,  # exclusive access
            None,  # no security
            win32con.OPEN_EXISTING,
            win32con.FILE_ATTRIBUTE_NORMAL | win32con.FILE_FLAG_OVERLAPPED,
            None,
        )
        # Tell the port we want a notification on each char.
        SetCommMask(self.handle, EV_RXCHAR)
        # Setup a 4k buffer
        SetupComm(self.handle, 4096, 4096)
        # Remove anything that was there
        PurgeComm(
            self.handle, PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR
        )
        # Setup for overlapped IO.
        timeouts = 0xFFFFFFFF, 0, 1000, 0, 1000
        SetCommTimeouts(self.handle, timeouts)
        # Setup the connection info.
        dcb = GetCommState(self.handle)
        dcb.BaudRate = CBR_115200
        dcb.ByteSize = 8
        dcb.Parity = NOPARITY
        dcb.StopBits = ONESTOPBIT
        SetCommState(self.handle, dcb)
        print("Connected to %s at %s baud" % (port, dcb.BaudRate))

    def _UserInputReaderThread(self):
        overlapped = OVERLAPPED()
        overlapped.hEvent = CreateEvent(None, 1, 0, None)
        try:
            while 1:
                ch = msvcrt.getch()
                if ord(ch) == 3:
                    break
                WriteFile(self.handle, ch, overlapped)
                # Wait for the write to complete.
                WaitForSingleObject(overlapped.hEvent, INFINITE)
        finally:
            SetEvent(self.eventStop)

    def _ComPortThread(self):
        overlapped = OVERLAPPED()
        overlapped.hEvent = CreateEvent(None, 1, 0, None)
        while 1:
            # XXX - note we could _probably_ just use overlapped IO on the win32file.ReadFile() statement
            # XXX but this tests the COM stuff!
            rc, mask = WaitCommEvent(self.handle, overlapped)
            if rc == 0:  # Character already ready!
                SetEvent(overlapped.hEvent)
            rc = WaitForMultipleObjects(
                [overlapped.hEvent, self.eventStop], 0, INFINITE
            )
            if rc == WAIT_OBJECT_0:
                # Some input - read and print it
                flags, comstat = ClearCommError(self.handle)
                rc, data = ReadFile(self.handle, comstat.cbInQue, overlapped)
                WaitForSingleObject(overlapped.hEvent, INFINITE)
                sys.stdout.write(data)
            else:
                # Stop the thread!
                # Just incase the user input thread uis still going, close it
                sys.stdout.close()
                break

    def Run(self):
        self.eventStop = CreateEvent(None, 0, 0, None)
        # Start the reader and writer threads.
        user_thread = threading.Thread(target=self._UserInputReaderThread)
        user_thread.start()
        com_thread = threading.Thread(target=self._ComPortThread)
        com_thread.start()
        user_thread.join()
        com_thread.join()


if __name__ == "__main__":
    print("Serial port terminal demo - press Ctrl+C to exit")
    if len(sys.argv) <= 1:
        port = FindModem()
        if port is None:
            print("No COM port specified, and no modem could be found")
            print("Please re-run this script with the name of a COM port (eg COM3)")
            sys.exit(1)
    else:
        port = sys.argv[1]

    tty = SerialTTY(port)
    tty.Run()
