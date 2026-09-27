"""How recalled memory is written into a prompt or a tool result.

The recall policy (``recall.py``) decides what may be read back; this module
writes out what it returns. A fact line always says when the fact holds, so
an ended or retired fact never reads as a current one.
"""

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from pydantic import BaseModel, ValidationError

from .memory_model import MemoryStatus
from .recall import fact_status, is_live

# Scope of an episode that is not a ``MemoryEnvelope`` (plain conversation).
GLOBAL_SCOPE = "real:global"
# Episode bodies are cut to this many characters when rendered.
EPISODE_DISPLAY_CHARS = 500

_RETIRED_STATUSES = frozenset(
    status.value
    for status in (
        MemoryStatus.superseded,
        MemoryStatus.contradicted,
        MemoryStatus.retracted,
    )
)


def render(fact: EntityEdge) -> str:
    """One recall line for ``fact``: its text and when it holds.

    A live fact shows its valid-time interval (``fact_validity``), which may
    already have ended: recall still returns a fact that held in the past,
    and its end date keeps it from reading as current. A retired fact
    (expired, or in a status recall skips) is labelled with how and when it
    was retired instead; recall never returns one, the label keeps any other
    caller honest.
    """
    text = fact_text(fact)
    if is_live(fact):
        valid_from, valid_to = fact_validity(fact)
        return f"{text} (valid: {valid_from} — {valid_to})"
    retired_at = str(fact.expired_at) if fact.expired_at else "at an unknown time"
    return f"{text} ({_retired_label(fact)} {retired_at})"


def fact_text(fact: EntityEdge) -> str:
    """The fact sentence, or the relation name when extraction left none."""
    return fact.fact or fact.name


def fact_validity(fact: EntityEdge) -> tuple[str, str]:
    """``(valid_from, valid_to)`` in valid time.

    ``valid_to`` is "present" only while no ``invalid_at`` is known; a past
    one is shown as it is, so an ended fact reads as history, not as now.
    """
    valid_from = str(fact.valid_at) if fact.valid_at else "unknown"
    valid_to = str(fact.invalid_at) if fact.invalid_at else "present"
    return valid_from, valid_to


def render_episode(episode: EpisodicNode) -> str:
    """``[created_at] body``, the body cut to ``EPISODE_DISPLAY_CHARS``."""
    return f"[{episode.created_at}] {episode.content[:EPISODE_DISPLAY_CHARS]}"


def episode_scope(episode: EpisodicNode) -> str:
    """The ``MemoryEnvelope`` scope an episode was stored under.

    An episode that is not an envelope (plain conversation, or JSON that is
    not an object) belongs to ``GLOBAL_SCOPE``. Reads the full body: an
    envelope cut to display length is no longer valid JSON.
    """
    try:
        return _EnvelopeScope.model_validate_json(episode.content).scope
    except ValidationError:
        return GLOBAL_SCOPE


class _EnvelopeScope(BaseModel):
    scope: str = GLOBAL_SCOPE


def _retired_label(fact: EntityEdge) -> str:
    status = fact_status(fact)
    return status if status in _RETIRED_STATUSES else "expired"
