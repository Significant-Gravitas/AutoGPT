"""Link a published skill version into the owner's Graphiti memory.

The skill registry stays the canonical store for the procedure; memory
only records a *reference* — that the scope learned a named skill, which
version, and where the evidence came from. No procedure steps are written
to the graph, so a later restore or revocation cannot leave stale
instructions there: version validity is always decided by the registry at
retrieval time. Provenance is stamped deterministically on the edges as
``assistant_derived`` so it never reads as a user-asserted fact.
"""

from __future__ import annotations

import logging

from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.graphiti.ingest import enqueue_episode
from backend.copilot.graphiti.memory_model import (
    MemoryEnvelope,
    MemoryKind,
    MemoryStatus,
    SourceKind,
)
from backend.data.skill_versions import SkillVersionRecord

logger = logging.getLogger(__name__)


def memory_scope_for(version: SkillVersionRecord) -> str:
    return f"expert:{version.expert_id}" if version.expert_id else "real:global"


def build_reference_envelope(version: SkillVersionRecord) -> MemoryEnvelope:
    source_refs = ", ".join(
        f"{s.get('source_kind')}:{s.get('source_ref')}@{s.get('revision')}"
        for s in version.sources
    )
    return MemoryEnvelope(
        content=(
            f"Learned skill '{version.skill_name}' (version {version.version}) "
            f"is available to this scope; load it with read_skill."
        ),
        source_kind=SourceKind.assistant_derived,
        scope=memory_scope_for(version),
        memory_kind=MemoryKind.procedure,
        status=MemoryStatus.active,
        provenance=f"skill_version:{version.id}; sources: {source_refs}",
    )


async def link_version_to_memory(user_id: str, version: SkillVersionRecord) -> bool:
    """Best-effort; returns whether an episode was queued. Never raises."""
    try:
        if not await is_enabled_for_user(user_id):
            return False
        envelope = build_reference_envelope(version)
        # ``dream_`` name prefix: the dream pass must not treat our own
        # write as new user activity worth a paid consolidation pass.
        return await enqueue_episode(
            user_id,
            f"skill_learning_{version.id}",
            name=f"dream_skill_{version.id}",
            episode_body=envelope.model_dump_json(),
            source_description="dream-pass skill learning",
            is_json=True,
            edge_metadata={
                "status": envelope.status.value,
                "source_kind": envelope.source_kind.value,
                "scope": envelope.scope,
                "confidence": None,
                "provenance": envelope.provenance,
            },
            expert_id=version.expert_id,
        )
    except Exception:
        logger.warning(
            "Skill learning memory link failed for user %s", user_id[:12], exc_info=True
        )
        return False
