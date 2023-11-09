import functools
import time
import io

import win32file
import win32pipe

cERROR_PIPE_BUSY = 0xe7
cSECURITY_SQOS_PRESENT = 0x100000
cSECURITY_ANONYMOUS = 0

MAXIMUM_RETRY_COUNT = 10


def check_closed(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError(
                'Can not reuse socket after connection was closed.'
            )
        return f(self, *args, **kwargs)
    return wrapped


class NpipeSocket:
    """ Partial implementation of the socket API over windows named pipes.
        This implementation is only designed to be used as a client socket,
        and server-specific methods (bind, listen, accept...) are not
        implemented.
    """

    def __init__(self, handle=None):
        self._timeout = win32pipe.NMPWAIT_USE_DEFAULT_WAIT
        self._handle = handle
        self._closed = False

    def accept(self):
        raise NotImplementedError()

    def bind(self, address):
        raise NotImplementedError()

    def close(self):
        self._handle.Close()
        self._closed = True

    @check_closed
    def connect(self, address, retry_count=0):
        try:
            handle = win32file.CreateFile(
                address,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                cSECURITY_ANONYMOUS | cSECURITY_SQOS_PRESENT,
                0
            )
        except win32pipe.error as e:
            # See Remarks:
            # https://msdn.microsoft.com/en-us/library/aa365800.aspx
            if e.winerror == cERROR_PIPE_BUSY:
                # Another program or thread has grabbed our pipe instance
                # before we got to it. Wait for availability and attempt to
                # connect again.
                retry_count = retry_count + 1
                if (retry_count < MAXIMUM_RETRY_COUNT):
                    time.sleep(1)
                    return self.connect(address, retry_count)
            raise e

        self.flags = win32pipe.GetNamedPipeInfo(handle)[0]

        self._handle = handle
        self._address = address

    @check_closed
    def connect_ex(self, address):
        return self.connect(address)

    @check_closed
    def detach(self):
        self._closed = True
        return self._handle

    @check_closed
    def dup(self):
        return NpipeSocket(self._handle)

    def getpeername(self):
        return self._address

    def getsockname(self):
        return self._address

    def getsockopt(self, level, optname, buflen=None):
        raise NotImplementedError()

    def ioctl(self, control, option):
        raise NotImplementedError()

    def listen(self, backlog):
        raise NotImplementedError()

    def makefile(self, mode=None, bufsize=None):
        if mode.strip('b') != 'r':
            raise NotImplementedError()
        rawio = NpipeFileIOBase(self)
        if bufsize is None or bufsize <= 0:
            bufsize = io.DEFAULT_BUFFER_SIZE
        return io.BufferedReader(rawio, buffer_size=bufsize)

    @check_closed
    def recv(self, bufsize, flags=0):
        err, data = win32file.ReadFile(self._handle, bufsize)
        return data

    @check_closed
    def recvfrom(self, bufsize, flags=0):
        data = self.recv(bufsize, flags)
        return (data, self._address)

    @check_closed
    def recvfrom_into(self, buf, nbytes=0, flags=0):
        return self.recv_into(buf, nbytes, flags), self._address

    @check_closed
    def recv_into(self, buf, nbytes=0):
        readbuf = buf
        if not isinstance(buf, memoryview):
            readbuf = memoryview(buf)

        err, data = win32file.ReadFile(
            self._handle,
            readbuf[:nbytes] if nbytes else readbuf
        )
        return len(data)

    def _recv_into_py2(self, buf, nbytes):
        err, data = win32file.ReadFile(self._handle, nbytes or len(buf))
        n = len(data)
        buf[:n] = data
        return n

    @check_closed
    def send(self, string, flags=0):
        err, nbytes = win32file.WriteFile(self._handle, string)
        return nbytes

    @check_closed
    def sendall(self, string, flags=0):
        return self.send(string, flags)

    @check_closed
    def sendto(self, string, address):
        self.connect(address)
        return self.send(string)

    def setblocking(self, flag):
        if flag:
            return self.settimeout(None)
        return self.settimeout(0)

    def settimeout(self, value):
        if value is None:
            # Blocking mode
            self._timeout = win32pipe.NMPWAIT_WAIT_FOREVER
        elif not isinstance(value, (float, int)) or value < 0:
            raise ValueError('Timeout value out of range')
        elif value == 0:
            # Non-blocking mode
            self._timeout = win32pipe.NMPWAIT_NO_WAIT
        else:
            # Timeout mode - Value converted to milliseconds
            self._timeout = value * 1000

    def gettimeout(self):
        return self._timeout

    def setsockopt(self, level, optname, value):
        raise NotImplementedError()

    @check_closed
    def shutdown(self, how):
        return self.close()


class NpipeFileIOBase(io.RawIOBase):
    def __init__(self, npipe_socket):
        self.sock = npipe_socket

    def close(self):
        super().close()
        self.sock = None

    def fileno(self):
        return self.sock.fileno()

    def isatty(self):
        return False

    def readable(self):
        return True

    def readinto(self, buf):
        return self.sock.recv_into(buf)

    def seekable(self):
        return False

    def writable(self):
        return False
