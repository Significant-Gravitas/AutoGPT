# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

"""curio async I/O library query support"""

import socket
import curio
import curio.socket  # type: ignore

import dns._asyncbackend
import dns.exception
import dns.inet


def _maybe_timeout(timeout):
    if timeout:
        return curio.ignore_after(timeout)
    else:
        return dns._asyncbackend.NullContext()


# for brevity
_lltuple = dns.inet.low_level_address_tuple

# pylint: disable=redefined-outer-name


class DatagramSocket(dns._asyncbackend.DatagramSocket):
    def __init__(self, socket):
        super().__init__(socket.family)
        self.socket = socket

    async def sendto(self, what, destination, timeout):
        async with _maybe_timeout(timeout):
            return await self.socket.sendto(what, destination)
        raise dns.exception.Timeout(
            timeout=timeout
        )  # pragma: no cover  lgtm[py/unreachable-statement]

    async def recvfrom(self, size, timeout):
        async with _maybe_timeout(timeout):
            return await self.socket.recvfrom(size)
        raise dns.exception.Timeout(timeout=timeout)  # lgtm[py/unreachable-statement]

    async def close(self):
        await self.socket.close()

    async def getpeername(self):
        return self.socket.getpeername()

    async def getsockname(self):
        return self.socket.getsockname()


class StreamSocket(dns._asyncbackend.StreamSocket):
    def __init__(self, socket):
        self.socket = socket
        self.family = socket.family

    async def sendall(self, what, timeout):
        async with _maybe_timeout(timeout):
            return await self.socket.sendall(what)
        raise dns.exception.Timeout(timeout=timeout)  # lgtm[py/unreachable-statement]

    async def recv(self, size, timeout):
        async with _maybe_timeout(timeout):
            return await self.socket.recv(size)
        raise dns.exception.Timeout(timeout=timeout)  # lgtm[py/unreachable-statement]

    async def close(self):
        await self.socket.close()

    async def getpeername(self):
        return self.socket.getpeername()

    async def getsockname(self):
        return self.socket.getsockname()


class Backend(dns._asyncbackend.Backend):
    def name(self):
        return "curio"

    async def make_socket(
        self,
        af,
        socktype,
        proto=0,
        source=None,
        destination=None,
        timeout=None,
        ssl_context=None,
        server_hostname=None,
    ):
        if socktype == socket.SOCK_DGRAM:
            s = curio.socket.socket(af, socktype, proto)
            try:
                if source:
                    s.bind(_lltuple(source, af))
            except Exception:  # pragma: no cover
                await s.close()
                raise
            return DatagramSocket(s)
        elif socktype == socket.SOCK_STREAM:
            if source:
                source_addr = _lltuple(source, af)
            else:
                source_addr = None
            async with _maybe_timeout(timeout):
                s = await curio.open_connection(
                    destination[0],
                    destination[1],
                    ssl=ssl_context,
                    source_addr=source_addr,
                    server_hostname=server_hostname,
                )
            return StreamSocket(s)
        raise NotImplementedError(
            "unsupported socket " + f"type {socktype}"
        )  # pragma: no cover

    async def sleep(self, interval):
        await curio.sleep(interval)
