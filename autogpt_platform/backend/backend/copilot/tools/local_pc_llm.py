from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .local_pc_shim import LocalPCShim

_LLM_STREAM_IDLE_TIMEOUT_SECONDS = 30.0
_LLM_STREAM_TIMEOUT_SECONDS = 300.0
_LLM_STREAM_MAX_OUTPUT_BYTES = 4 * 1024 * 1024


class LocalLLMError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class LocalLLMProxy:
    def __init__(self, shim: LocalPCShim) -> None:
        self._shim = shim

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        await self._shim._await_capacity("LOCAL_LLM_COMPLETION")
        msg_id = str(uuid.uuid4())
        queue = self._shim._register_stream(msg_id)
        envelope = {
            "type": "LOCAL_LLM_COMPLETION",
            "id": msg_id,
            "ts": time.time(),
            "payload": {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
            },
        }
        deadline = asyncio.get_running_loop().time() + _LLM_STREAM_TIMEOUT_SECONDS
        try:
            try:
                await asyncio.wait_for(
                    self._shim._ws.send_text(json.dumps(envelope)),
                    timeout=min(
                        _LLM_STREAM_IDLE_TIMEOUT_SECONDS, _LLM_STREAM_TIMEOUT_SECONDS
                    ),
                )
            except TimeoutError as exc:
                raise LocalLLMError(
                    "LOCAL_LLM_TIMEOUT", "Local LLM request timed out"
                ) from exc
            except Exception as exc:
                raise LocalLLMError(
                    "LOCAL_LLM_FAILED", "Failed to send Local LLM request"
                ) from exc
            async for delta in _read_stream(queue, deadline):
                yield delta
        finally:
            self._shim._cleanup_stream(msg_id)

    async def complete_blocking(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        return "".join(
            [
                delta
                async for delta in self.complete(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            ]
        )


async def _read_stream(
    queue: asyncio.Queue[dict], deadline: float
) -> AsyncIterator[str]:
    received_bytes = 0
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        try:
            frame = await asyncio.wait_for(
                queue.get(), timeout=min(_LLM_STREAM_IDLE_TIMEOUT_SECONDS, remaining)
            )
        except TimeoutError as exc:
            raise LocalLLMError(
                "LOCAL_LLM_TIMEOUT", "Local LLM stream timed out"
            ) from exc
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            raise LocalLLMError("LOCAL_LLM_FAILED", "Invalid Local LLM response")
        if frame.get("type") == "LOCAL_LLM_COMPLETION_RESPONSE":
            return
        if frame.get("type") == "ERROR":
            raise LocalLLMError(
                str(payload.get("code") or "LOCAL_LLM_FAILED"),
                str(payload.get("message") or "Local LLM completion failed"),
                (
                    payload.get("details")
                    if isinstance(payload.get("details"), dict)
                    else None
                ),
            )
        delta = payload.get("delta") or ""
        if not isinstance(delta, str):
            raise LocalLLMError("LOCAL_LLM_FAILED", "Invalid Local LLM response")
        received_bytes += len(delta.encode())
        if received_bytes > _LLM_STREAM_MAX_OUTPUT_BYTES:
            raise LocalLLMError(
                "LOCAL_LLM_STREAM_LIMIT", "Local LLM output exceeded its size limit"
            )
        if delta:
            yield delta
