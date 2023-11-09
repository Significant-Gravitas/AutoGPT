import socket
import struct
import threading
import time

# websocket modules
from ._abnf import *
from ._exceptions import *
from ._handshake import *
from ._http import *
from ._logging import *
from ._socket import *
from ._ssl_compat import *
from ._utils import *

"""
_core.py
websocket - WebSocket client library for Python

Copyright 2022 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__all__ = ['WebSocket', 'create_connection']


class WebSocket:
    """
    Low level WebSocket interface.

    This class is based on the WebSocket protocol `draft-hixie-thewebsocketprotocol-76 <http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol-76>`_

    We can connect to the websocket server and send/receive data.
    The following example is an echo client.

    >>> import websocket
    >>> ws = websocket.WebSocket()
    >>> ws.connect("ws://echo.websocket.events")
    >>> ws.recv()
    'echo.websocket.events sponsored by Lob.com'
    >>> ws.send("Hello, Server")
    19
    >>> ws.recv()
    'Hello, Server'
    >>> ws.close()

    Parameters
    ----------
    get_mask_key: func
        A callable function to get new mask keys, see the
        WebSocket.set_mask_key's docstring for more information.
    sockopt: tuple
        Values for socket.setsockopt.
        sockopt must be tuple and each element is argument of sock.setsockopt.
    sslopt: dict
        Optional dict object for ssl socket options. See FAQ for details.
    fire_cont_frame: bool
        Fire recv event for each cont frame. Default is False.
    enable_multithread: bool
        If set to True, lock send method.
    skip_utf8_validation: bool
        Skip utf8 validation.
    """

    def __init__(self, get_mask_key=None, sockopt=None, sslopt=None,
                 fire_cont_frame=False, enable_multithread=True,
                 skip_utf8_validation=False, **_):
        """
        Initialize WebSocket object.

        Parameters
        ----------
        sslopt: dict
            Optional dict object for ssl socket options. See FAQ for details.
        """
        self.sock_opt = sock_opt(sockopt, sslopt)
        self.handshake_response = None
        self.sock = None

        self.connected = False
        self.get_mask_key = get_mask_key
        # These buffer over the build-up of a single frame.
        self.frame_buffer = frame_buffer(self._recv, skip_utf8_validation)
        self.cont_frame = continuous_frame(
            fire_cont_frame, skip_utf8_validation)

        if enable_multithread:
            self.lock = threading.Lock()
            self.readlock = threading.Lock()
        else:
            self.lock = NoLock()
            self.readlock = NoLock()

    def __iter__(self):
        """
        Allow iteration over websocket, implying sequential `recv` executions.
        """
        while True:
            yield self.recv()

    def __next__(self):
        return self.recv()

    def next(self):
        return self.__next__()

    def fileno(self):
        return self.sock.fileno()

    def set_mask_key(self, func):
        """
        Set function to create mask key. You can customize mask key generator.
        Mainly, this is for testing purpose.

        Parameters
        ----------
        func: func
            callable object. the func takes 1 argument as integer.
            The argument means length of mask key.
            This func must return string(byte array),
            which length is argument specified.
        """
        self.get_mask_key = func

    def gettimeout(self):
        """
        Get the websocket timeout (in seconds) as an int or float

        Returns
        ----------
        timeout: int or float
             returns timeout value (in seconds). This value could be either float/integer.
        """
        return self.sock_opt.timeout

    def settimeout(self, timeout):
        """
        Set the timeout to the websocket.

        Parameters
        ----------
        timeout: int or float
            timeout time (in seconds). This value could be either float/integer.
        """
        self.sock_opt.timeout = timeout
        if self.sock:
            self.sock.settimeout(timeout)

    timeout = property(gettimeout, settimeout)

    def getsubprotocol(self):
        """
        Get subprotocol
        """
        if self.handshake_response:
            return self.handshake_response.subprotocol
        else:
            return None

    subprotocol = property(getsubprotocol)

    def getstatus(self):
        """
        Get handshake status
        """
        if self.handshake_response:
            return self.handshake_response.status
        else:
            return None

    status = property(getstatus)

    def getheaders(self):
        """
        Get handshake response header
        """
        if self.handshake_response:
            return self.handshake_response.headers
        else:
            return None

    def is_ssl(self):
        try:
            return isinstance(self.sock, ssl.SSLSocket)
        except:
            return False

    headers = property(getheaders)

    def connect(self, url, **options):
        """
        Connect to url. url is websocket url scheme.
        ie. ws://host:port/resource
        You can customize using 'options'.
        If you set "header" list object, you can set your own custom header.

        >>> ws = WebSocket()
        >>> ws.connect("ws://echo.websocket.events",
                ...     header=["User-Agent: MyProgram",
                ...             "x-custom: header"])

        Parameters
        ----------
        header: list or dict
            Custom http header list or dict.
        cookie: str
            Cookie value.
        origin: str
            Custom origin url.
        connection: str
            Custom connection header value.
            Default value "Upgrade" set in _handshake.py
        suppress_origin: bool
            Suppress outputting origin header.
        host: str
            Custom host header string.
        timeout: int or float
            Socket timeout time. This value is an integer or float.
            If you set None for this value, it means "use default_timeout value"
        http_proxy_host: str
            HTTP proxy host name.
        http_proxy_port: str or int
            HTTP proxy port. Default is 80.
        http_no_proxy: list
            Whitelisted host names that don't use the proxy.
        http_proxy_auth: tuple
            HTTP proxy auth information. Tuple of username and password. Default is None.
        http_proxy_timeout: int or float
            HTTP proxy timeout, default is 60 sec as per python-socks.
        redirect_limit: int
            Number of redirects to follow.
        subprotocols: list
            List of available subprotocols. Default is None.
        socket: socket
            Pre-initialized stream socket.
        """
        self.sock_opt.timeout = options.get('timeout', self.sock_opt.timeout)
        self.sock, addrs = connect(url, self.sock_opt, proxy_info(**options),
                                   options.pop('socket', None))

        try:
            self.handshake_response = handshake(self.sock, url, *addrs, **options)
            for attempt in range(options.pop('redirect_limit', 3)):
                if self.handshake_response.status in SUPPORTED_REDIRECT_STATUSES:
                    url = self.handshake_response.headers['location']
                    self.sock.close()
                    self.sock, addrs = connect(url, self.sock_opt, proxy_info(**options),
                                               options.pop('socket', None))
                    self.handshake_response = handshake(self.sock, url, *addrs, **options)
            self.connected = True
        except:
            if self.sock:
                self.sock.close()
                self.sock = None
            raise

    def send(self, payload, opcode=ABNF.OPCODE_TEXT):
        """
        Send the data as string.

        Parameters
        ----------
        payload: str
            Payload must be utf-8 string or unicode,
            If the opcode is OPCODE_TEXT.
            Otherwise, it must be string(byte array).
        opcode: int
            Operation code (opcode) to send.
        """

        frame = ABNF.create_frame(payload, opcode)
        return self.send_frame(frame)

    def send_frame(self, frame):
        """
        Send the data frame.

        >>> ws = create_connection("ws://echo.websocket.events")
        >>> frame = ABNF.create_frame("Hello", ABNF.OPCODE_TEXT)
        >>> ws.send_frame(frame)
        >>> cont_frame = ABNF.create_frame("My name is ", ABNF.OPCODE_CONT, 0)
        >>> ws.send_frame(frame)
        >>> cont_frame = ABNF.create_frame("Foo Bar", ABNF.OPCODE_CONT, 1)
        >>> ws.send_frame(frame)

        Parameters
        ----------
        frame: ABNF frame
            frame data created by ABNF.create_frame
        """
        if self.get_mask_key:
            frame.get_mask_key = self.get_mask_key
        data = frame.format()
        length = len(data)
        if (isEnabledForTrace()):
            trace("++Sent raw: " + repr(data))
            trace("++Sent decoded: " + frame.__str__())
        with self.lock:
            while data:
                l = self._send(data)
                data = data[l:]

        return length

    def send_binary(self, payload):
        """
        Send a binary message (OPCODE_BINARY).

        Parameters
        ----------
        payload: bytes
            payload of message to send.
        """
        return self.send(payload, ABNF.OPCODE_BINARY)

    def ping(self, payload=""):
        """
        Send ping data.

        Parameters
        ----------
        payload: str
            data payload to send server.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.send(payload, ABNF.OPCODE_PING)

    def pong(self, payload=""):
        """
        Send pong data.

        Parameters
        ----------
        payload: str
            data payload to send server.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.send(payload, ABNF.OPCODE_PONG)

    def recv(self):
        """
        Receive string data(byte array) from the server.

        Returns
        ----------
        data: string (byte array) value.
        """
        with self.readlock:
            opcode, data = self.recv_data()
        if opcode == ABNF.OPCODE_TEXT:
            return data.decode("utf-8")
        elif opcode == ABNF.OPCODE_TEXT or opcode == ABNF.OPCODE_BINARY:
            return data
        else:
            return ''

    def recv_data(self, control_frame=False):
        """
        Receive data with operation code.

        Parameters
        ----------
        control_frame: bool
            a boolean flag indicating whether to return control frame
            data, defaults to False

        Returns
        -------
        opcode, frame.data: tuple
            tuple of operation code and string(byte array) value.
        """
        opcode, frame = self.recv_data_frame(control_frame)
        return opcode, frame.data

    def recv_data_frame(self, control_frame=False):
        """
        Receive data with operation code.

        If a valid ping message is received, a pong response is sent.

        Parameters
        ----------
        control_frame: bool
            a boolean flag indicating whether to return control frame
            data, defaults to False

        Returns
        -------
        frame.opcode, frame: tuple
            tuple of operation code and string(byte array) value.
        """
        while True:
            frame = self.recv_frame()
            if (isEnabledForTrace()):
                trace("++Rcv raw: " + repr(frame.format()))
                trace("++Rcv decoded: " + frame.__str__())
            if not frame:
                # handle error:
                # 'NoneType' object has no attribute 'opcode'
                raise WebSocketProtocolException(
                    "Not a valid frame %s" % frame)
            elif frame.opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY, ABNF.OPCODE_CONT):
                self.cont_frame.validate(frame)
                self.cont_frame.add(frame)

                if self.cont_frame.is_fire(frame):
                    return self.cont_frame.extract(frame)

            elif frame.opcode == ABNF.OPCODE_CLOSE:
                self.send_close()
                return frame.opcode, frame
            elif frame.opcode == ABNF.OPCODE_PING:
                if len(frame.data) < 126:
                    self.pong(frame.data)
                else:
                    raise WebSocketProtocolException(
                        "Ping message is too long")
                if control_frame:
                    return frame.opcode, frame
            elif frame.opcode == ABNF.OPCODE_PONG:
                if control_frame:
                    return frame.opcode, frame

    def recv_frame(self):
        """
        Receive data as frame from server.

        Returns
        -------
        self.frame_buffer.recv_frame(): ABNF frame object
        """
        return self.frame_buffer.recv_frame()

    def send_close(self, status=STATUS_NORMAL, reason=b""):
        """
        Send close data to the server.

        Parameters
        ----------
        status: int
            Status code to send. See STATUS_XXX.
        reason: str or bytes
            The reason to close. This must be string or UTF-8 bytes.
        """
        if status < 0 or status >= ABNF.LENGTH_16:
            raise ValueError("code is invalid range")
        self.connected = False
        self.send(struct.pack('!H', status) + reason, ABNF.OPCODE_CLOSE)

    def close(self, status=STATUS_NORMAL, reason=b"", timeout=3):
        """
        Close Websocket object

        Parameters
        ----------
        status: int
            Status code to send. See STATUS_XXX.
        reason: bytes
            The reason to close in UTF-8.
        timeout: int or float
            Timeout until receive a close frame.
            If None, it will wait forever until receive a close frame.
        """
        if self.connected:
            if status < 0 or status >= ABNF.LENGTH_16:
                raise ValueError("code is invalid range")

            try:
                self.connected = False
                self.send(struct.pack('!H', status) + reason, ABNF.OPCODE_CLOSE)
                sock_timeout = self.sock.gettimeout()
                self.sock.settimeout(timeout)
                start_time = time.time()
                while timeout is None or time.time() - start_time < timeout:
                    try:
                        frame = self.recv_frame()
                        if frame.opcode != ABNF.OPCODE_CLOSE:
                            continue
                        if isEnabledForError():
                            recv_status = struct.unpack("!H", frame.data[0:2])[0]
                            if recv_status >= 3000 and recv_status <= 4999:
                                debug("close status: " + repr(recv_status))
                            elif recv_status != STATUS_NORMAL:
                                error("close status: " + repr(recv_status))
                        break
                    except:
                        break
                self.sock.settimeout(sock_timeout)
                self.sock.shutdown(socket.SHUT_RDWR)
            except:
                pass

            self.shutdown()

    def abort(self):
        """
        Low-level asynchronous abort, wakes up other threads that are waiting in recv_*
        """
        if self.connected:
            self.sock.shutdown(socket.SHUT_RDWR)

    def shutdown(self):
        """
        close socket, immediately.
        """
        if self.sock:
            self.sock.close()
            self.sock = None
            self.connected = False

    def _send(self, data):
        return send(self.sock, data)

    def _recv(self, bufsize):
        try:
            return recv(self.sock, bufsize)
        except WebSocketConnectionClosedException:
            if self.sock:
                self.sock.close()
            self.sock = None
            self.connected = False
            raise


def create_connection(url, timeout=None, class_=WebSocket, **options):
    """
    Connect to url and return websocket object.

    Connect to url and return the WebSocket object.
    Passing optional timeout parameter will set the timeout on the socket.
    If no timeout is supplied,
    the global default timeout setting returned by getdefaulttimeout() is used.
    You can customize using 'options'.
    If you set "header" list object, you can set your own custom header.

    >>> conn = create_connection("ws://echo.websocket.events",
         ...     header=["User-Agent: MyProgram",
         ...             "x-custom: header"])

    Parameters
    ----------
    class_: class
        class to instantiate when creating the connection. It has to implement
        settimeout and connect. It's __init__ should be compatible with
        WebSocket.__init__, i.e. accept all of it's kwargs.
    header: list or dict
        custom http header list or dict.
    cookie: str
        Cookie value.
    origin: str
        custom origin url.
    suppress_origin: bool
        suppress outputting origin header.
    host: str
        custom host header string.
    timeout: int or float
        socket timeout time. This value could be either float/integer.
        If set to None, it uses the default_timeout value.
    http_proxy_host: str
        HTTP proxy host name.
    http_proxy_port: str or int
        HTTP proxy port. If not set, set to 80.
    http_no_proxy: list
        Whitelisted host names that don't use the proxy.
    http_proxy_auth: tuple
        HTTP proxy auth information. tuple of username and password. Default is None.
    http_proxy_timeout: int or float
        HTTP proxy timeout, default is 60 sec as per python-socks.
    enable_multithread: bool
        Enable lock for multithread.
    redirect_limit: int
        Number of redirects to follow.
    sockopt: tuple
        Values for socket.setsockopt.
        sockopt must be a tuple and each element is an argument of sock.setsockopt.
    sslopt: dict
        Optional dict object for ssl socket options. See FAQ for details.
    subprotocols: list
        List of available subprotocols. Default is None.
    skip_utf8_validation: bool
        Skip utf8 validation.
    socket: socket
        Pre-initialized stream socket.
    """
    sockopt = options.pop("sockopt", [])
    sslopt = options.pop("sslopt", {})
    fire_cont_frame = options.pop("fire_cont_frame", False)
    enable_multithread = options.pop("enable_multithread", True)
    skip_utf8_validation = options.pop("skip_utf8_validation", False)
    websock = class_(sockopt=sockopt, sslopt=sslopt,
                     fire_cont_frame=fire_cont_frame,
                     enable_multithread=enable_multithread,
                     skip_utf8_validation=skip_utf8_validation, **options)
    websock.settimeout(timeout if timeout is not None else getdefaulttimeout())
    websock.connect(url, **options)
    return websock
