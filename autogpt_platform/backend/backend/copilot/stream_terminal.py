"""Per-turn terminal records and recovery of unpublished failures."""

import logging
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from redis.exceptions import RedisError

from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamRecoveredError,
)
from backend.data.redis_client import AsyncRedisClient
from backend.data.redis_helpers import as_str

logger = logging.getLogger(__name__)


class TurnTerminal(BaseModel):
    """The committed outcome, with optional public error details."""

    model_config = ConfigDict(strict=True)

    turn_id: str
    status: Literal["completed", "failed"]
    error_message: str = ""
    error_code: str = ""

    def as_error(
        self, *, include_suppressed_errors: bool = False
    ) -> StreamRecoveredError | None:
        """Keep suppressed failures quiet except for internal outcome waiters."""
        if self.status != "failed":
            return None
        if not self.error_message and not include_suppressed_errors:
            return None
        return StreamRecoveredError(
            turn_id=self.turn_id,
            errorText=self.error_message or "Turn ended without completing.",
            code=self.error_code or None,
        )


async def recover_terminal_error(
    redis: AsyncRedisClient,
    meta_key: str,
    stream_key: str,
    turn_id: str,
    after: str,
    *,
    replay_count: int,
    parse_chunk: Callable[[dict[str, str]], StreamBaseResponse | None],
    include_suppressed_errors: bool = False,
) -> StreamRecoveredError | None:
    """Recover a matching failure unless its error was already consumed."""
    terminal = await get_terminal_meta(redis, meta_key, turn_id)
    if terminal is None:
        return None
    error = terminal.as_error(include_suppressed_errors=include_suppressed_errors)
    if error is None or after == "0-0":
        return error
    try:
        if await _error_was_consumed(
            redis, stream_key, after, replay_count, parse_chunk
        ):
            return None
        return error
    except RedisError:
        logger.warning(f"Could not check terminal error cursor for turn {turn_id}")
        return None


async def get_terminal_meta(
    redis: AsyncRedisClient, meta_key: str, turn_id: str
) -> TurnTerminal | None:
    """Prefer the immutable turn record; fall back only to its own session hash."""
    try:
        raw = await redis.get(terminal_key(meta_key, turn_id))
        if raw is not None:
            terminal = TurnTerminal.model_validate_json(raw)
            if terminal.turn_id == turn_id:
                return terminal
    except ValidationError:
        logger.warning(f"Invalid terminal record for turn {turn_id}")
    except RedisError:
        logger.warning(
            f"Could not read terminal record for turn {turn_id}", exc_info=True
        )
    try:
        meta = await redis.hgetall(meta_key)
        terminal = TurnTerminal.model_validate(
            {as_str(key): as_str(value) for key, value in (meta or {}).items()}
        )
        return terminal if terminal.turn_id == turn_id else None
    except ValidationError:
        return None
    except RedisError:
        logger.warning(
            f"Could not recover terminal error for turn {turn_id}", exc_info=True
        )
        return None


def terminal_key(meta_key: str, turn_id: str) -> str:
    """Colocate the turn record with the ownership-guarded session hash."""
    return f"terminal:{meta_key}:{turn_id}"


async def _error_was_consumed(
    redis: AsyncRedisClient,
    stream_key: str,
    after: str,
    replay_count: int,
    parse_chunk: Callable[[dict[str, str]], StreamBaseResponse | None],
) -> bool:
    """Search backwards through the consumed prefix with exclusive page cursors."""
    upper = after
    while True:
        entries = (
            await redis.xrevrange(stream_key, max=upper, count=replay_count)
        ) or []
        for _, fields in entries:
            if fields is None:
                continue
            decoded = {as_str(k) or "": as_str(v) or "" for k, v in fields.items()}
            if isinstance(parse_chunk(decoded), StreamError):
                return True
        if len(entries) < replay_count:
            return False
        upper = f"({as_str(entries[-1][0])}"
