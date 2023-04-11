# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# This is a nullcontext for both sync and async.  3.7 has a nullcontext,
# but it is only for sync use.


class NullContext:
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    async def __aenter__(self):
        return self.enter_result

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass


# These are declared here so backends can import them without creating
# circular dependencies with dns.asyncbackend.


class Socket:  # pragma: no cover
    async def close(self):
        pass

    async def getpeername(self):
        raise NotImplementedError

    async def getsockname(self):
        raise NotImplementedError

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()


class DatagramSocket(Socket):  # pragma: no cover
    def __init__(self, family: int):
        self.family = family

    async def sendto(self, what, destination, timeout):
        raise NotImplementedError

    async def recvfrom(self, size, timeout):
        raise NotImplementedError


class StreamSocket(Socket):  # pragma: no cover
    async def sendall(self, what, timeout):
        raise NotImplementedError

    async def recv(self, size, timeout):
        raise NotImplementedError


class Backend:  # pragma: no cover
    def name(self):
        return "unknown"

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
        raise NotImplementedError

    def datagram_connection_required(self):
        return False

    async def sleep(self, interval):
        raise NotImplementedError
