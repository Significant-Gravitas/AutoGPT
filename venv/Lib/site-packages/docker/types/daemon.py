import socket

try:
    import requests.packages.urllib3 as urllib3
except ImportError:
    import urllib3

from ..errors import DockerException


class CancellableStream:
    """
    Stream wrapper for real-time events, logs, etc. from the server.

    Example:
        >>> events = client.events()
        >>> for event in events:
        ...   print(event)
        >>> # and cancel from another thread
        >>> events.close()
    """

    def __init__(self, stream, response):
        self._stream = stream
        self._response = response

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._stream)
        except urllib3.exceptions.ProtocolError:
            raise StopIteration
        except OSError:
            raise StopIteration

    next = __next__

    def close(self):
        """
        Closes the event streaming.
        """

        if not self._response.raw.closed:
            # find the underlying socket object
            # based on api.client._get_raw_response_socket

            sock_fp = self._response.raw._fp.fp

            if hasattr(sock_fp, 'raw'):
                sock_raw = sock_fp.raw

                if hasattr(sock_raw, 'sock'):
                    sock = sock_raw.sock

                elif hasattr(sock_raw, '_sock'):
                    sock = sock_raw._sock

            elif hasattr(sock_fp, 'channel'):
                # We're working with a paramiko (SSH) channel, which doesn't
                # support cancelable streams with the current implementation
                raise DockerException(
                    'Cancellable streams not supported for the SSH protocol'
                )
            else:
                sock = sock_fp._sock

            if hasattr(urllib3.contrib, 'pyopenssl') and isinstance(
                    sock, urllib3.contrib.pyopenssl.WrappedSocket):
                sock = sock.socket

            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
