"""Where a pai session keeps its model history between turns.

Same bucket and path convention as the CLI/baseline transcript
(``cli-sessions/<user>/<session>``, see ``copilot.transcript``) with a
``.pai.json`` suffix, so the engines never read each other's files. One file
holds the messages, the held calls the last run ended on, and the chat-row
watermark it covers, so there is no meta file to fall out of step with it.
"""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from backend.copilot.transcript import (
    _CLI_SESSION_STORAGE_PREFIX,
    _build_path_from_parts,
    _sanitize_id,
)
from backend.util.workspace_storage import get_workspace_storage

from .history import HeldToolCall, sanitize_for_storage

logger = logging.getLogger(__name__)

HISTORY_SUFFIX = ".pai.json"
_FORMAT_VERSION = 1
# Bounded like the baseline's transcript upload: a hung bucket must not hold
# the turn's finish; the write carries on in the background.
UPLOAD_TIMEOUT_S = 5
_background_uploads: set[asyncio.Task[None]] = set()


class StoredHistory(BaseModel):
    """The file's content."""

    version: int = _FORMAT_VERSION
    # Next chat-row sequence this history does not cover (``transcript``'s
    # ``next_uncovered_sequence``), so rows written by another engine later
    # can be folded in.
    watermark: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    held: list[HeldToolCall] = Field(default_factory=list)


class LoadedHistory(BaseModel):
    """A decoded history; ``messages`` are Pydantic AI messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    watermark: int
    messages: list[ModelMessage]
    held: list[HeldToolCall]


def history_path_parts(user_id: str, session_id: str) -> tuple[str, str, str]:
    return (
        _CLI_SESSION_STORAGE_PREFIX,
        _sanitize_id(user_id),
        f"{_sanitize_id(session_id)}{HISTORY_SUFFIX}",
    )


def encode_history(
    messages: list[ModelMessage], held: list[HeldToolCall], watermark: int
) -> bytes:
    dumped = ModelMessagesTypeAdapter.dump_python(
        sanitize_for_storage(messages), mode="json"
    )
    return (
        StoredHistory(watermark=watermark, messages=dumped, held=held)
        .model_dump_json()
        .encode("utf-8")
    )


def decode_history(content: bytes) -> LoadedHistory | None:
    try:
        stored = StoredHistory.model_validate_json(content)
        messages = ModelMessagesTypeAdapter.validate_python(stored.messages)
    except (ValidationError, ValueError):
        logger.warning("[PAI] Stored history is unreadable; rebuilding from rows")
        return None
    if stored.version != _FORMAT_VERSION:
        return None
    return LoadedHistory(
        watermark=stored.watermark, messages=messages, held=stored.held
    )


async def download_history(
    user_id: str, session_id: str
) -> tuple[bool, LoadedHistory | None]:
    """``(upload_safe, history)``: like the baseline, a failed read (unknown
    bucket state) makes this turn's upload unsafe; a missing or corrupt file
    does not."""
    try:
        storage = await get_workspace_storage()
        path = _build_path_from_parts(history_path_parts(user_id, session_id), storage)
        content = await storage.retrieve(path)
    except FileNotFoundError:
        return True, None
    except Exception:
        logger.warning("[PAI] History download failed", exc_info=True)
        return False, None
    return True, decode_history(content)


async def upload_history(
    user_id: str,
    session_id: str,
    messages: list[ModelMessage],
    held: list[HeldToolCall],
    watermark: int,
) -> None:
    """Store the history, waiting at most ``UPLOAD_TIMEOUT_S`` for it."""
    task = asyncio.create_task(
        _store(user_id, session_id, encode_history(messages, held, watermark))
    )
    _background_uploads.add(task)
    task.add_done_callback(_background_uploads.discard)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=UPLOAD_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.info("[PAI] History upload still running in the background")
    except Exception:
        logger.warning("[PAI] History upload failed", exc_info=True)


async def _store(user_id: str, session_id: str, content: bytes) -> None:
    storage = await get_workspace_storage()
    workspace_id, file_id, filename = history_path_parts(user_id, session_id)
    await storage.store(
        workspace_id=workspace_id, file_id=file_id, filename=filename, content=content
    )
