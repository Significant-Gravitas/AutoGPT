"""SMTP connections that go only to checked addresses, within a deadline.

``smtplib`` resolves the server's name when it connects, so checking the
addresses a name resolves to says nothing about where the connection then goes:
the name can resolve to another address the second time. ``PinnedSMTP``
connects only to the addresses it is given and keeps the name for TLS, where the
server's certificate must be valid for it.

``smtplib`` blocks, so callers run it in a worker thread. A socket timeout bounds
one connect, read or write there, not the conversation: a server that sends a
little at a time is never idle for long. ``SMTPDeadline`` ends the conversation
from outside the thread.
"""

import contextlib
import ipaddress
import smtplib
import socket
import ssl
import threading
import time
from collections.abc import Sequence
from typing import BinaryIO


class SMTPDeadlinePassed(Exception):
    """The conversation was ended because its deadline passed."""


class SMTPDeadline:
    """How long one SMTP conversation may take.

    ``expire()`` may be called from any thread. It shuts the conversation's
    connection down, so a read or write waiting in the conversation's thread
    fails at once, and no connection is opened after it.
    """

    def __init__(self, seconds: float):
        self._ends_at = time.monotonic() + seconds
        self._lock = threading.Lock()
        self._expired = False
        self._connections: list[socket.socket] = []

    @property
    def expired(self) -> bool:
        return self._expired or time.monotonic() >= self._ends_at

    def timeout(self, limit: float) -> float:
        """``limit``, or the time left when that is less."""
        left = self._ends_at - time.monotonic()
        if self._expired or left <= 0:
            raise SMTPDeadlinePassed()
        return min(limit, left)

    def watch(self, connection: socket.socket) -> None:
        """Let ``expire()`` shut ``connection`` down; close it if that already ran."""
        with self._lock:
            if not self._expired:
                # A duplicate, because STARTTLS replaces the socket object
                # smtplib holds with an SSL one around the same connection.
                self._connections.append(connection.dup())
                return
        connection.close()
        raise SMTPDeadlinePassed()

    def expire(self) -> None:
        with self._lock:
            self._expired = True
            for connection in self._connections:
                with contextlib.suppress(OSError):  # the server closed it first
                    connection.shutdown(socket.SHUT_RDWR)

    def release(self) -> None:
        """Close what ``watch`` kept, once the conversation is over."""
        with self._lock:
            connections, self._connections = self._connections, []
        for connection in connections:
            connection.close()


def verified_tls_context() -> ssl.SSLContext:
    """TLS for STARTTLS that requires a valid certificate for the server's name;
    ``smtplib``'s own default checks neither the certificate nor the name."""
    return ssl.create_default_context()


class BoundedSMTP(smtplib.SMTP):
    """``smtplib.SMTP`` that caps the size of one server reply."""

    MAX_REPLY_BYTES = 1024 * 1024  # a real reply is a few KB

    def getreply(self):
        # smtplib opens the reply stream lazily, and again after STARTTLS.
        if self.file is None and self.sock is not None:
            self.file = _BoundedReplyStream(
                self.sock.makefile("rb"), self.MAX_REPLY_BYTES
            )
        if isinstance(self.file, _BoundedReplyStream):
            self.file.reply_bytes = 0
        return super().getreply()


class PinnedSMTP(BoundedSMTP):
    """``BoundedSMTP`` that connects only to ``addresses``, not to what ``host``
    resolves to. ``starttls()`` still sends ``host`` as the TLS server name and
    checks the certificate against it."""

    def __init__(
        self,
        host: str,
        port: int,
        addresses: Sequence[str],
        *,
        timeout: float,
        deadline: SMTPDeadline,
    ):
        # IP addresses only, since a name would be resolved again; and each once,
        # since getaddrinfo lists an address once per socket type.
        self._addresses = [
            str(ipaddress.ip_address(address)) for address in dict.fromkeys(addresses)
        ]
        self._deadline = deadline
        super().__init__(host, port, timeout=timeout)

    def _get_socket(self, host: str, port: int, timeout: float) -> socket.socket:
        errors: list[OSError] = []
        for address in self._addresses:
            attempt_timeout = self._deadline.timeout(timeout)
            try:
                connection = socket.create_connection(
                    (address, port), attempt_timeout, self.source_address
                )
            except OSError as e:
                errors.append(e)
                continue
            self._deadline.watch(connection)
            return connection
        raise errors[-1] if errors else OSError(f"No address to connect to for {host}")


class _BoundedReplyStream:
    def __init__(self, file: BinaryIO, max_reply_bytes: int):
        self._file = file
        self._max_reply_bytes = max_reply_bytes
        self.reply_bytes = 0

    def readline(self, size: int = -1) -> bytes:
        line = self._file.readline(size)
        self.reply_bytes += len(line)
        if self.reply_bytes > self._max_reply_bytes:
            # An OSError makes getreply() close the connection and raise
            # SMTPServerDisconnected, which the caller reports.
            raise OSError("SMTP reply too long")
        return line

    def close(self) -> None:
        self._file.close()
