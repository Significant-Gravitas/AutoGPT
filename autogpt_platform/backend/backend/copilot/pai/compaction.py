"""Context compaction for a pai run, as a Pydantic AI history processor.

The work is the baseline's ``_compress_session_messages`` (LLM summary of
older turns, then truncation and middle-out deletion), run on the flattened
history before each model request. Pydantic AI writes the processed history
back into the run, so a compaction happens once and the stored history is the
compacted one.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence

from pydantic_ai.messages import ModelMessage

from backend.copilot.baseline.service import _compress_session_messages
from backend.util.prompt import get_compression_target

from .history import chat_rows_to_messages, messages_to_chat_rows

logger = logging.getLogger(__name__)

HistoryProcessor = Callable[[list[ModelMessage]], Awaitable[list[ModelMessage]]]

# Below this many characters per target token no tokenizer puts a history
# over the target, so the count (and the flatten) can be skipped.
_CHARS_PER_TOKEN_FLOOR = 2


def compaction_processor(model: str, *, always_check: bool) -> HistoryProcessor:
    """A processor compacting toward *model*'s window.

    ``always_check`` skips the cheap size guard: the local transport probes
    its real (often small) window inside the compressor itself.
    """
    target = get_compression_target(model)

    async def process(messages: list[ModelMessage]) -> list[ModelMessage]:
        if not always_check and _size(messages) < target * _CHARS_PER_TOKEN_FLOOR:
            return messages
        return await compact(messages, model)

    return process


async def compact(messages: list[ModelMessage], model: str) -> list[ModelMessage]:
    """The compacted history, or *messages* itself when nothing was cut."""
    tail = messages[-1] if messages else None
    if tail is None or tail.kind != "request":
        return messages
    rows = messages_to_chat_rows(messages)
    compressed = await _compress_session_messages(rows, model)
    if compressed is rows:
        return messages
    rebuilt = chat_rows_to_messages(compressed)
    last = rebuilt[-1] if rebuilt else None
    if last is None or last.kind != "request":
        logger.warning("[PAI] Compaction did not end on a request; keeping history")
        return messages
    # The run's instructions ride on the last request.
    last.instructions = tail.instructions
    logger.info(f"[PAI] Context compacted: {len(messages)} -> {len(rebuilt)} messages")
    return rebuilt


def _size(messages: Sequence[ModelMessage]) -> int:
    return sum(len(str(part)) for message in messages for part in message.parts)
