"""Background embedding generation for ``UserWorkspaceFile`` rows.

Mirrors ``backend.api.features.library.embeddings``: fire-and-forget
scheduling so user-facing file writes don't pay the OpenAI embedding
latency. Failures are logged, not raised — a missing embedding only
degrades search quality, it never breaks correctness.

Only the user-visible ``name`` (and the path stem, when it diverges) is
embedded — file *contents* are intentionally out of scope.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict

from prisma.enums import ContentType

from backend.api.features.search.content_handlers import build_workspace_file_text
from backend.api.features.search.embeddings import (
    EMBEDDING_MODEL,
    delete_content_embedding,
    generate_embedding,
    get_content_embedding,
    store_content_embedding,
)

logger = logging.getLogger(__name__)

# See ``library/embeddings.py`` for why we hold a strong ref to the task.
_background_tasks: set[asyncio.Task[None]] = set()

_EMBED_TTL_SECONDS = 86_400
_EMBED_CACHE_SIZE = 256
# Finished vectors, oldest first. Keyed by the text, which is all that is
# embedded: every installed skill is "SKILL.md SKILL", so a hire of 8 skills
# asked for one vector 8 times.
_embedded: OrderedDict[tuple[str, str], tuple[float, list[float]]] = OrderedDict()
_embedded_lock = threading.Lock()
# Requests in flight, per event loop: callers wanting the same text share one
# request, and a request for other text never waits behind it.
_in_flight: dict[
    tuple[asyncio.AbstractEventLoop, str, str], asyncio.Future[list[float]]
] = {}


async def _run_embedding(file_id: str, user_id: str, name: str, path: str) -> None:
    try:
        searchable_text = build_workspace_file_text(name, path)
        if not searchable_text:
            logger.debug(
                "Skipping workspace file embedding for %s: empty searchable text",
                file_id,
            )
            return
        existing = await get_content_embedding(
            ContentType.WORKSPACE_FILE, file_id, user_id
        )
        if existing and existing.get("searchableText") == searchable_text:
            return
        await store_content_embedding(
            content_type=ContentType.WORKSPACE_FILE,
            content_id=file_id,
            embedding=await _embed(EMBEDDING_MODEL, searchable_text),
            searchable_text=searchable_text,
            metadata={"name": name, "path": path},
            user_id=user_id,
        )
    except Exception as e:
        logger.warning(
            "Failed to ensure workspace file embedding for %s: %s", file_id, e
        )


async def _embed(model: str, text: str) -> list[float]:
    """*text*'s vector, requested at most once per process a day."""
    with _embedded_lock:
        hit = _embedded.get((model, text))
    if hit is not None and time.monotonic() - hit[0] < _EMBED_TTL_SECONDS:
        return hit[1]
    key = (asyncio.get_running_loop(), model, text)
    request = _in_flight.get(key)
    if request is None:
        request = asyncio.ensure_future(_request_embedding(model, text))
        _in_flight[key] = request
        request.add_done_callback(
            lambda done: _in_flight.pop(key) if _in_flight.get(key) is done else None
        )
    return await asyncio.shield(request)


async def _request_embedding(model: str, text: str) -> list[float]:
    vector = await generate_embedding(text)
    with _embedded_lock:
        _embedded[(model, text)] = (time.monotonic(), vector)
        _embedded.move_to_end((model, text))
        while len(_embedded) > _EMBED_CACHE_SIZE:
            _embedded.popitem(last=False)
    return vector


def schedule_workspace_file_embedding(
    file_id: str, user_id: str, name: str, path: str
) -> asyncio.Task[None]:
    """Schedule a fire-and-forget (re-)embed of a workspace file."""
    task = asyncio.create_task(_run_embedding(file_id, user_id, name, path))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def delete_workspace_file_embedding(file_id: str, user_id: str) -> None:
    """Best-effort embedding cleanup when a workspace file is deleted."""
    try:
        await delete_content_embedding(
            ContentType.WORKSPACE_FILE, file_id, user_id=user_id
        )
    except Exception as e:
        logger.warning(
            "Failed to delete workspace file embedding for %s: %s", file_id, e
        )
