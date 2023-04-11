# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

try:
    import aioquic.quic.configuration  # type: ignore

    import dns.asyncbackend
    from dns._asyncbackend import NullContext
    from dns.quic._sync import SyncQuicManager, SyncQuicConnection, SyncQuicStream
    from dns.quic._asyncio import (
        AsyncioQuicManager,
        AsyncioQuicConnection,
        AsyncioQuicStream,
    )
    from dns.quic._common import AsyncQuicConnection, AsyncQuicManager

    have_quic = True

    def null_factory(
        *args,  # pylint: disable=unused-argument
        **kwargs  # pylint: disable=unused-argument
    ):
        return NullContext(None)

    def _asyncio_manager_factory(
        context, *args, **kwargs  # pylint: disable=unused-argument
    ):
        return AsyncioQuicManager(*args, **kwargs)

    # We have a context factory and a manager factory as for trio we need to have
    # a nursery.

    _async_factories = {"asyncio": (null_factory, _asyncio_manager_factory)}

    try:
        import trio
        from dns.quic._trio import (  # pylint: disable=ungrouped-imports
            TrioQuicManager,
            TrioQuicConnection,
            TrioQuicStream,
        )

        def _trio_context_factory():
            return trio.open_nursery()

        def _trio_manager_factory(context, *args, **kwargs):
            return TrioQuicManager(context, *args, **kwargs)

        _async_factories["trio"] = (_trio_context_factory, _trio_manager_factory)
    except ImportError:
        pass

    def factories_for_backend(backend=None):
        if backend is None:
            backend = dns.asyncbackend.get_default_backend()
        return _async_factories[backend.name()]

except ImportError:
    have_quic = False

    from typing import Any

    class AsyncQuicStream:  # type: ignore
        pass

    class AsyncQuicConnection:  # type: ignore
        async def make_stream(self) -> Any:
            raise NotImplementedError

    class SyncQuicStream:  # type: ignore
        pass

    class SyncQuicConnection:  # type: ignore
        def make_stream(self) -> Any:
            raise NotImplementedError
