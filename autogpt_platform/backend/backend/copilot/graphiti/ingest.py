"""Async episode ingestion with per-user serialization.

graphiti-core requires sequential ``add_episode()`` calls within the same
group_id.  This module provides a per-user asyncio.Queue that serializes
ingestion while keeping it fire-and-forget from the caller's perspective.
"""

import asyncio
import logging
import weakref
from datetime import datetime, timezone

from graphiti_core.nodes import EpisodeType

from .client import derive_group_id, get_graphiti_client
from .memory_model import MemoryEnvelope, MemoryKind, MemoryStatus, SourceKind

logger = logging.getLogger(__name__)


# The CoPilot executor runs one asyncio loop per worker thread, and
# asyncio.Queue / asyncio.Lock / asyncio.Task are all bound to the loop they
# were first used on. A process-wide worker registry would hand a loop-1-bound
# Queue to a coroutine running on loop 2 → RuntimeError "Future attached to a
# different loop". Scope the registry per running loop so each loop has its
# own queues, workers, and lock. Entries auto-clean when the loop is GC'd.
class _LoopIngestState:
    __slots__ = ("user_queues", "user_workers", "workers_lock")

    def __init__(self) -> None:
        self.user_queues: dict[str, asyncio.Queue] = {}
        self.user_workers: dict[str, asyncio.Task] = {}
        self.workers_lock = asyncio.Lock()


_loop_state: (
    "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopIngestState]"
) = weakref.WeakKeyDictionary()


def _get_loop_state() -> _LoopIngestState:
    loop = asyncio.get_running_loop()
    state = _loop_state.get(loop)
    if state is None:
        state = _LoopIngestState()
        _loop_state[loop] = state
    return state


# Idle workers are cleaned up after this many seconds of inactivity.
_WORKER_IDLE_TIMEOUT = 60

CUSTOM_EXTRACTION_INSTRUCTIONS = """
- Do not extract "User", "Assistant", "AI", "System", "CoPilot", or "human" as entity nodes.
- Do not extract software tool names, block names, API endpoint names, or internal system identifiers as entities.
- Do not extract action descriptions like "the assistant created..." as facts. Extract only the underlying user intent or real-world information.
- Focus on real-world entities: people, companies, products, projects, concepts, and preferences.
- Use canonical names: if the speaker says "my company" and context reveals it is "Acme Corp", use "Acme Corp".
"""


async def _ingestion_worker(user_id: str, queue: asyncio.Queue) -> None:
    """Process episodes sequentially for a single user.

    Exits after ``_WORKER_IDLE_TIMEOUT`` seconds of inactivity so that
    idle workers don't leak memory indefinitely.
    """
    # Snapshot the loop-local state at task start so cleanup always runs
    # against the same state dict the worker was registered in, even if the
    # worker is cancelled from another task.
    state = _get_loop_state()
    try:
        while True:
            try:
                payload = await asyncio.wait_for(
                    queue.get(), timeout=_WORKER_IDLE_TIMEOUT
                )
            except asyncio.TimeoutError:
                break  # idle — clean up below

            try:
                group_id = derive_group_id(user_id)
                client = await get_graphiti_client(group_id)
                await client.add_episode(**payload)
            except Exception:
                logger.warning(
                    "Graphiti ingestion failed for user %s",
                    user_id[:12],
                    exc_info=True,
                )
            finally:
                queue.task_done()
    except asyncio.CancelledError:
        logger.debug("Ingestion worker cancelled for user %s", user_id[:12])
        raise
    finally:
        # Clean up so the next message re-creates the worker.
        state.user_queues.pop(user_id, None)
        state.user_workers.pop(user_id, None)


async def enqueue_conversation_turn(
    user_id: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str = "",
) -> None:
    """Enqueue a conversation turn for async background ingestion.

    This returns almost immediately — the actual graphiti-core
    ``add_episode()`` call (which triggers LLM entity extraction)
    runs in a background worker task.

    If ``assistant_msg`` is provided and contains substantive findings
    (not just acknowledgments), a separate derived-finding episode is
    queued with ``source_kind=assistant_derived`` and ``status=tentative``.
    """
    if not user_id:
        return

    try:
        group_id = derive_group_id(user_id)
    except ValueError:
        logger.warning("Invalid user_id for ingestion: %s", user_id[:12])
        return

    user_display_name = await _resolve_user_name(user_id)

    episode_name = f"conversation_{session_id}"

    # User's own words only, in graphiti's expected "Speaker: content" format.
    # Assistant response is excluded from extraction
    # (Zep Cloud approach: ignore_roles=["assistant"]).
    episode_body_for_graphiti = f"{user_display_name}: {user_msg}"

    source_description = f"User message in session {session_id}"

    queue = await _ensure_worker(user_id)

    try:
        queue.put_nowait(
            {
                "name": episode_name,
                "episode_body": episode_body_for_graphiti,
                "source": EpisodeType.message,
                "source_description": source_description,
                "reference_time": datetime.now(timezone.utc),
                "group_id": group_id,
                "custom_extraction_instructions": CUSTOM_EXTRACTION_INSTRUCTIONS,
            }
        )
    except asyncio.QueueFull:
        logger.warning(
            "Graphiti ingestion queue full for user %s — dropping episode",
            user_id[:12],
        )
        return

    # --- Derived-finding lane ---
    # If the assistant response is substantive, distill it into a
    # structured finding with tentative status.
    if assistant_msg and _is_finding_worthy(assistant_msg):
        finding = _distill_finding(assistant_msg)
        if finding:
            envelope = MemoryEnvelope(
                content=finding,
                source_kind=SourceKind.assistant_derived,
                memory_kind=MemoryKind.finding,
                status=MemoryStatus.tentative,
                provenance=f"session:{session_id}",
            )
            try:
                queue.put_nowait(
                    {
                        "name": f"finding_{session_id}",
                        "episode_body": envelope.model_dump_json(),
                        "source": EpisodeType.json,
                        "source_description": f"Assistant-derived finding in session {session_id}",
                        "reference_time": datetime.now(timezone.utc),
                        "group_id": group_id,
                        "custom_extraction_instructions": CUSTOM_EXTRACTION_INSTRUCTIONS,
                    }
                )
            except asyncio.QueueFull:
                pass  # user canonical episode already queued — finding is best-effort


async def enqueue_episode(
    user_id: str,
    session_id: str,
    *,
    name: str,
    episode_body: str,
    source_description: str = "Conversation memory",
    is_json: bool = False,
) -> bool:
    """Enqueue an arbitrary episode for background ingestion.

    Used by ``MemoryStoreTool`` so that explicit memory-store calls go
    through the same per-user serialization queue as conversation turns.

    Args:
        is_json: When ``True``, ingest as ``EpisodeType.json`` (for
            structured ``MemoryEnvelope`` payloads).  Otherwise uses
            ``EpisodeType.text``.

    Returns ``True`` if the episode was queued, ``False`` if it was dropped.
    """
    if not user_id:
        return False

    try:
        group_id = derive_group_id(user_id)
    except ValueError:
        logger.warning("Invalid user_id for episode ingestion: %s", user_id[:12])
        return False

    queue = await _ensure_worker(user_id)

    source = EpisodeType.json if is_json else EpisodeType.text

    try:
        queue.put_nowait(
            {
                "name": name,
                "episode_body": episode_body,
                "source": source,
                "source_description": source_description,
                "reference_time": datetime.now(timezone.utc),
                "group_id": group_id,
                "custom_extraction_instructions": CUSTOM_EXTRACTION_INSTRUCTIONS,
            }
        )
        return True
    except asyncio.QueueFull:
        logger.warning(
            "Graphiti ingestion queue full for user %s — dropping episode",
            user_id[:12],
        )
        return False


async def _ensure_worker(user_id: str) -> asyncio.Queue:
    """Create a queue and worker for *user_id* if one doesn't exist.

    Returns the queue directly so callers don't need to look it up from
    the state dict (which avoids a TOCTOU race if the worker times out
    and cleans up between this call and the put_nowait).
    """
    state = _get_loop_state()
    async with state.workers_lock:
        if user_id not in state.user_queues:
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            state.user_queues[user_id] = q
            state.user_workers[user_id] = asyncio.create_task(
                _ingestion_worker(user_id, q),
                name=f"graphiti-ingest-{user_id[:12]}",
            )
        return state.user_queues[user_id]


async def _resolve_user_name(user_id: str) -> str:
    """Get the user's display name from BusinessUnderstanding, or fall back to 'User'."""
    try:
        from backend.data.db_accessors import understanding_db

        understanding = await understanding_db().get_business_understanding(user_id)
        if understanding and understanding.user_name:
            return understanding.user_name
    except Exception:
        logger.debug("Could not resolve user name for %s", user_id[:12])
    return "User"


# --- Derived-finding distillation ---

# Phrases that indicate workflow chatter, not substantive findings.
_CHATTER_PREFIXES = (
    "done",
    "got it",
    "sure, i",
    "sure!",
    "ok",
    "okay",
    "i've created",
    "i've updated",
    "i've sent",
    "i'll ",
    "let me ",
    "a sign-in button",
    "please click",
)

# Minimum length for an assistant message to be considered finding-worthy.
_MIN_FINDING_LENGTH = 150


def _is_finding_worthy(assistant_msg: str) -> bool:
    """Heuristic gate: is this assistant response worth distilling into a finding?

    Skips short acknowledgments, workflow chatter, and UI prompts.
    Only passes through responses that likely contain substantive
    factual content (research results, analysis, conclusions).
    """
    if len(assistant_msg) < _MIN_FINDING_LENGTH:
        return False

    lower = assistant_msg.lower().strip()
    for prefix in _CHATTER_PREFIXES:
        if lower.startswith(prefix):
            return False

    return True


def _distill_finding(assistant_msg: str) -> str | None:
    """Extract the core finding from an assistant response.

    For now, uses a simple truncation approach. Phase 3+ could use
    a lightweight LLM call for proper distillation.
    """
    # Take the first 500 chars as the finding content.
    # Strip markdown formatting artifacts.
    content = assistant_msg.strip()
    if len(content) > 500:
        content = content[:500] + "..."
    return content if content else None
