"""Bounded Codex response retention for a CLI retry after a disconnect."""

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from aiohttp import web
from pydantic import BaseModel, ConfigDict

MAX_REPLAY_ENTRIES = 128
MAX_REPLAY_BYTES = 8 * 1024 * 1024
REPLAY_TTL_SECONDS = 1800.0


class RecordingResponse(web.StreamResponse):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.recording: bytearray | None = bytearray()
        self._disconnected = False

    async def prepare(self, request):
        if request.transport is None or request.transport.is_closing():
            self._disconnected = True
            return None
        try:
            return await super().prepare(request)
        except ConnectionError:
            self._disconnected = True
            return None

    async def write(self, data: bytes) -> None:
        if self.recording is not None:
            if len(self.recording) + len(data) <= MAX_REPLAY_BYTES:
                self.recording.extend(data)
            else:
                self.recording = None
        if not self._disconnected:
            try:
                await super().write(data)
            except ConnectionError:
                self._disconnected = True

    async def write_eof(self, data: bytes = b"") -> None:
        if data:
            await self.write(data)
        if not self._disconnected:
            try:
                await super().write_eof()
            except ConnectionError:
                self._disconnected = True


class _Replay(BaseModel):
    model_config = ConfigDict(frozen=True)

    body: bytes
    status: int
    headers: dict[str, str]
    expires: float


class ResponseReplays:
    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Task[web.StreamResponse]] = {}
        self._completed: OrderedDict[str, _Replay] = OrderedDict()
        self._bytes = 0

    async def response(
        self,
        key: str,
        produce: Callable[[], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        self._prune()
        replay = self._completed.get(key)
        if replay is not None:
            return web.Response(
                body=replay.body, status=replay.status, headers=replay.headers
            )
        task = self._pending.get(key)
        if task is not None:
            await asyncio.shield(task)
            replay = self._completed.get(key)
            if replay is not None:
                return web.Response(
                    body=replay.body, status=replay.status, headers=replay.headers
                )
            return await self.response(key, produce)
        task = asyncio.create_task(self._produce(key, produce))
        self._pending[key] = task
        return await asyncio.shield(task)

    async def _produce(
        self, key: str, produce: Callable[[], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        try:
            response = await produce()
            if (
                isinstance(response, RecordingResponse)
                and response.recording is not None
            ):
                body = bytes(response.recording)
                if response.status == 200:
                    replay = _Replay(
                        body=body,
                        status=response.status,
                        headers={
                            k: v
                            for k, v in response.headers.items()
                            if k.lower() not in {"transfer-encoding", "content-length"}
                        },
                        expires=time.monotonic() + REPLAY_TTL_SECONDS,
                    )
                    self._completed[key] = replay
                    self._bytes += len(body)
                    self._prune()
            elif isinstance(response, web.Response) and response.status == 200:
                body = response.body
                if isinstance(body, bytes) and len(body) <= MAX_REPLAY_BYTES:
                    self._completed[key] = _Replay(
                        body=body,
                        status=response.status,
                        headers=dict(response.headers),
                        expires=time.monotonic() + REPLAY_TTL_SECONDS,
                    )
                    self._bytes += len(body)
                    self._prune()
            return response
        finally:
            self._pending.pop(key, None)

    def _prune(self) -> None:
        now = time.monotonic()
        while self._completed:
            first = next(iter(self._completed.values()))
            if (
                first.expires > now
                and len(self._completed) <= MAX_REPLAY_ENTRIES
                and self._bytes <= MAX_REPLAY_BYTES
            ):
                break
            _, replay = self._completed.popitem(last=False)
            self._bytes -= len(replay.body)

    async def close(self) -> None:
        tasks = list(self._pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._completed.clear()
        self._bytes = 0
