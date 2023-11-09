import errno
import selectors
import socket

from ._exceptions import *
from ._ssl_compat import *
from ._utils import *

"""
_socket.py
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

DEFAULT_SOCKET_OPTION = [(socket.SOL_TCP, socket.TCP_NODELAY, 1)]
if hasattr(socket, "SO_KEEPALIVE"):
    DEFAULT_SOCKET_OPTION.append((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1))
if hasattr(socket, "TCP_KEEPIDLE"):
    DEFAULT_SOCKET_OPTION.append((socket.SOL_TCP, socket.TCP_KEEPIDLE, 30))
if hasattr(socket, "TCP_KEEPINTVL"):
    DEFAULT_SOCKET_OPTION.append((socket.SOL_TCP, socket.TCP_KEEPINTVL, 10))
if hasattr(socket, "TCP_KEEPCNT"):
    DEFAULT_SOCKET_OPTION.append((socket.SOL_TCP, socket.TCP_KEEPCNT, 3))

_default_timeout = None

__all__ = ["DEFAULT_SOCKET_OPTION", "sock_opt", "setdefaulttimeout", "getdefaulttimeout",
           "recv", "recv_line", "send"]


class sock_opt:

    def __init__(self, sockopt, sslopt):
        if sockopt is None:
            sockopt = []
        if sslopt is None:
            sslopt = {}
        self.sockopt = sockopt
        self.sslopt = sslopt
        self.timeout = None


def setdefaulttimeout(timeout):
    """
    Set the global timeout setting to connect.

    Parameters
    ----------
    timeout: int or float
        default socket timeout time (in seconds)
    """
    global _default_timeout
    _default_timeout = timeout


def getdefaulttimeout():
    """
    Get default timeout

    Returns
    ----------
    _default_timeout: int or float
        Return the global timeout setting (in seconds) to connect.
    """
    return _default_timeout


def recv(sock, bufsize):
    if not sock:
        raise WebSocketConnectionClosedException("socket is already closed.")

    def _recv():
        try:
            return sock.recv(bufsize)
        except SSLWantReadError:
            pass
        except socket.error as exc:
            error_code = extract_error_code(exc)
            if error_code != errno.EAGAIN and error_code != errno.EWOULDBLOCK:
                raise

        sel = selectors.DefaultSelector()
        sel.register(sock, selectors.EVENT_READ)

        r = sel.select(sock.gettimeout())
        sel.close()

        if r:
            return sock.recv(bufsize)

    try:
        if sock.gettimeout() == 0:
            bytes_ = sock.recv(bufsize)
        else:
            bytes_ = _recv()
    except TimeoutError:
        raise WebSocketTimeoutException("Connection timed out")
    except socket.timeout as e:
        message = extract_err_message(e)
        raise WebSocketTimeoutException(message)
    except SSLError as e:
        message = extract_err_message(e)
        if isinstance(message, str) and 'timed out' in message:
            raise WebSocketTimeoutException(message)
        else:
            raise

    if not bytes_:
        raise WebSocketConnectionClosedException(
            "Connection to remote host was lost.")

    return bytes_


def recv_line(sock):
    line = []
    while True:
        c = recv(sock, 1)
        line.append(c)
        if c == b'\n':
            break
    return b''.join(line)


def send(sock, data):
    if isinstance(data, str):
        data = data.encode('utf-8')

    if not sock:
        raise WebSocketConnectionClosedException("socket is already closed.")

    def _send():
        try:
            return sock.send(data)
        except SSLWantWriteError:
            pass
        except socket.error as exc:
            error_code = extract_error_code(exc)
            if error_code is None:
                raise
            if error_code != errno.EAGAIN and error_code != errno.EWOULDBLOCK:
                raise

        sel = selectors.DefaultSelector()
        sel.register(sock, selectors.EVENT_WRITE)

        w = sel.select(sock.gettimeout())
        sel.close()

        if w:
            return sock.send(data)

    try:
        if sock.gettimeout() == 0:
            return sock.send(data)
        else:
            return _send()
    except socket.timeout as e:
        message = extract_err_message(e)
        raise WebSocketTimeoutException(message)
    except Exception as e:
        message = extract_err_message(e)
        if isinstance(message, str) and "timed out" in message:
            raise WebSocketTimeoutException(message)
        else:
            raise
