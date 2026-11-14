from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing

MAX_OUTPUT_BYTES = 256 * 1024
MAX_CHUNK_BYTES = 8192
IDLE_TIMEOUT_SECONDS = 30.0
TOTAL_TIMEOUT_SECONDS = 300.0
MAX_LINE_BYTES = 64 * 1024


async def bounded_chunks(stream: AsyncGenerator[dict, None]) -> AsyncGenerator[dict, None]:
    """The model's chunks, within the output limits. The first may take as
    long as the whole request, since Ollama sends nothing until it has loaded
    the model and read the prompt, which can take minutes; after it, each
    chunk must come within IDLE_TIMEOUT_SECONDS of the one before."""
    output_bytes = 0
    started = False
    async with aclosing(stream):
        while True:
            limit = IDLE_TIMEOUT_SECONDS if started else TOTAL_TIMEOUT_SECONDS
            try:
                async with asyncio.timeout(limit):
                    chunk = await anext(stream)
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise TimeoutError(f"The local model sent nothing for {limit:g} seconds") from None
            started = True
            if not isinstance(chunk, dict):
                raise ValueError("Invalid local model response")
            message = chunk.get("message") or {}
            if not isinstance(message, dict):
                raise ValueError("Invalid local model message")
            delta = message.get("content") or ""
            if not isinstance(delta, str):
                raise ValueError("Invalid local model content")
            chunk_bytes = len(delta.encode("utf-8"))
            output_bytes += chunk_bytes
            if chunk_bytes > MAX_CHUNK_BYTES or output_bytes > MAX_OUTPUT_BYTES:
                raise ValueError("Local model output exceeded its size limit")
            yield chunk


async def read_json_lines(chunks: AsyncIterator[bytes]) -> AsyncGenerator[dict, None]:
    pending = bytearray()
    async for chunk in chunks:
        for segment in chunk.splitlines(keepends=True):
            if len(pending) + len(segment) > MAX_LINE_BYTES:
                raise ValueError("Local model response line exceeded its size limit")
            pending.extend(segment)
            if segment.endswith(b"\n"):
                if pending.strip():
                    yield _parse_line(pending)
                pending.clear()
    if pending.strip():
        yield _parse_line(pending)


def _parse_line(line: bytearray) -> dict:
    try:
        value = json.loads(line)
    except (ValueError, UnicodeError):
        raise ValueError("Invalid local model JSON response") from None
    if not isinstance(value, dict):
        raise ValueError("Invalid local model JSON response")
    return value
