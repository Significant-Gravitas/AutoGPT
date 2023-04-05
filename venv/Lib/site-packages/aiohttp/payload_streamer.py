"""
Payload implemenation for coroutines as data provider.

As a simple case, you can upload data from file::

   @aiohttp.streamer
   async def file_sender(writer, file_name=None):
      with open(file_name, 'rb') as f:
          chunk = f.read(2**16)
          while chunk:
              await writer.write(chunk)

              chunk = f.read(2**16)

Then you can use `file_sender` like this:

    async with session.post('http://httpbin.org/post',
                            data=file_sender(file_name='huge_file')) as resp:
        print(await resp.text())

..note:: Coroutine must accept `writer` as first argument

"""

import types
import warnings
from typing import Any, Awaitable, Callable, Dict, Tuple

from .abc import AbstractStreamWriter
from .payload import Payload, payload_type

__all__ = ("streamer",)


class _stream_wrapper:
    def __init__(
        self,
        coro: Callable[..., Awaitable[None]],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        self.coro = types.coroutine(coro)
        self.args = args
        self.kwargs = kwargs

    async def __call__(self, writer: AbstractStreamWriter) -> None:
        await self.coro(writer, *self.args, **self.kwargs)  # type: ignore[operator]


class streamer:
    def __init__(self, coro: Callable[..., Awaitable[None]]) -> None:
        warnings.warn(
            "@streamer is deprecated, use async generators instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.coro = coro

    def __call__(self, *args: Any, **kwargs: Any) -> _stream_wrapper:
        return _stream_wrapper(self.coro, args, kwargs)


@payload_type(_stream_wrapper)
class StreamWrapperPayload(Payload):
    async def write(self, writer: AbstractStreamWriter) -> None:
        await self._value(writer)


@payload_type(streamer)
class StreamPayload(StreamWrapperPayload):
    def __init__(self, value: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(value(), *args, **kwargs)

    async def write(self, writer: AbstractStreamWriter) -> None:
        await self._value(writer)
