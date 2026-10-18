"""Server-sent events: reading an upstream's stream and writing the CLI's."""

import json
from collections.abc import AsyncIterable, AsyncIterator, Mapping

from pydantic import BaseModel, ConfigDict

PING = {"type": "ping"}


class SseFrame(BaseModel):
    """One upstream event: its ``event:`` name and joined ``data:`` lines."""

    model_config = ConfigDict(frozen=True)

    event: str | None = None
    data: str

    def encoded(self) -> bytes:
        """The frame as it is relayed: the same fields, one per line."""
        lines = [f"event: {self.event}"] if self.event else []
        lines.extend(f"data: {line}" for line in self.data.split("\n"))
        return ("\n".join(lines) + "\n\n").encode()

    def payload(self) -> object | None:
        """The decoded JSON data, or ``None`` for a non-JSON frame."""
        try:
            return json.loads(self.data)
        except ValueError:
            return None


def encode_event(event: Mapping[str, object]) -> bytes:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event['type']}\ndata: {payload}\n\n".encode()


async def read_frames(lines: AsyncIterable[str]) -> AsyncIterator[SseFrame]:
    """Group decoded lines into frames; comments and unknown fields are dropped."""
    event: str | None = None
    data: list[str] = []
    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            if data:
                yield SseFrame(event=event, data="\n".join(data))
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield SseFrame(event=event, data="\n".join(data))
