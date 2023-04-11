# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

from typing import Dict

import dns.exception

# pylint: disable=unused-import

from dns._asyncbackend import (
    Socket,
    DatagramSocket,
    StreamSocket,
    Backend,
)  # noqa: F401  lgtm[py/unused-import]

# pylint: enable=unused-import

_default_backend = None

_backends: Dict[str, Backend] = {}

# Allow sniffio import to be disabled for testing purposes
_no_sniffio = False


class AsyncLibraryNotFoundError(dns.exception.DNSException):
    pass


def get_backend(name: str) -> Backend:
    """Get the specified asynchronous backend.

    *name*, a ``str``, the name of the backend.  Currently the "trio",
    "curio", and "asyncio" backends are available.

    Raises NotImplementError if an unknown backend name is specified.
    """
    # pylint: disable=import-outside-toplevel,redefined-outer-name
    backend = _backends.get(name)
    if backend:
        return backend
    if name == "trio":
        import dns._trio_backend

        backend = dns._trio_backend.Backend()
    elif name == "curio":
        import dns._curio_backend

        backend = dns._curio_backend.Backend()
    elif name == "asyncio":
        import dns._asyncio_backend

        backend = dns._asyncio_backend.Backend()
    else:
        raise NotImplementedError(f"unimplemented async backend {name}")
    _backends[name] = backend
    return backend


def sniff() -> str:
    """Attempt to determine the in-use asynchronous I/O library by using
    the ``sniffio`` module if it is available.

    Returns the name of the library, or raises AsyncLibraryNotFoundError
    if the library cannot be determined.
    """
    # pylint: disable=import-outside-toplevel
    try:
        if _no_sniffio:
            raise ImportError
        import sniffio

        try:
            return sniffio.current_async_library()
        except sniffio.AsyncLibraryNotFoundError:
            raise AsyncLibraryNotFoundError(
                "sniffio cannot determine " + "async library"
            )
    except ImportError:
        import asyncio

        try:
            asyncio.get_running_loop()
            return "asyncio"
        except RuntimeError:
            raise AsyncLibraryNotFoundError("no async library detected")


def get_default_backend() -> Backend:
    """Get the default backend, initializing it if necessary."""
    if _default_backend:
        return _default_backend

    return set_default_backend(sniff())


def set_default_backend(name: str) -> Backend:
    """Set the default backend.

    It's not normally necessary to call this method, as
    ``get_default_backend()`` will initialize the backend
    appropriately in many cases.  If ``sniffio`` is not installed, or
    in testing situations, this function allows the backend to be set
    explicitly.
    """
    global _default_backend
    _default_backend = get_backend(name)
    return _default_backend
