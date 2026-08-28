import hashlib
import logging
import re

from backend.api.features.experts.learned_notes_db import LearnedNoteCandidate
from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_store import MemoryStoreTool
from backend.data.db_accessors import expert_learned_notes_db
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

_EXPLICIT_CORRECTION_PATTERNS = (
    re.compile(r"\bfrom now on\b", re.I),
    re.compile(r"\bremember (?:that|this)\b", re.I),
    re.compile(r"\buse your (?:own )?judg(?:e)?ment\b", re.I),
    re.compile(r"\b(?:do not|don['’]t|stop) ask(?:ing)? me\b", re.I),
    re.compile(r"\bi prefer (?:that|you|we)\b", re.I),
    re.compile(r"\byou should (?:always|never)\b", re.I),
    re.compile(r"^\s*(?:always|never)\b", re.I),
)


def explicit_correction(message: str | None) -> str | None:
    if not message:
        return None
    text = " ".join(message.strip().split())
    if not text or len(text) > 2_000:
        return None
    if not any(pattern.search(text) for pattern in _EXPLICIT_CORRECTION_PATTERNS):
        return None
    return text


async def capture_explicit_correction(
    *,
    user_id: str | None,
    session: ChatSession,
    message: str | None,
) -> bool:
    correction = explicit_correction(message)
    if user_id is None or correction is None:
        return False
    try:
        if not await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False):
            return False
        if session.expert_id:
            await expert_learned_notes_db().promote_learned_notes(
                user_id,
                session.expert_id,
                [
                    LearnedNoteCandidate(
                        text=correction,
                        source_session_id=session.session_id,
                    )
                ],
            )
            return True
        digest = hashlib.sha256(correction.casefold().encode()).hexdigest()[:12]
        result = await MemoryStoreTool()._execute(
            user_id,
            session,
            name=f"Founder correction {digest}",
            content=correction,
            source_description="Explicit founder correction",
            source_kind="user_asserted",
            memory_kind="rule",
            rule={
                "instruction": correction,
                "actor": session.expert_id or "AutoPilot",
                "trigger": "Future matching work",
            },
        )
        return result.type.value == "memory_store"
    except Exception:
        logger.warning("Could not retain explicit founder correction", exc_info=True)
        return False
