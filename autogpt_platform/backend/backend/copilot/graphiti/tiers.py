"""Tiered memory plumbing — group derivation, membership, governance, labels.

Three tiers share one FalkorDB-per-group substrate:

- ``personal`` → ``user_<id>``   (private; exists today)
- ``team``     → ``team_<id>``   (visible to ACTIVE members of that team)
- ``org``      → ``org_<id>``    (visible to every org member)

This module is the single place that:

- resolves which groups a read (warm context / explicit search) may touch,
  never returning a team the user is not an ACTIVE member of;
- resolves the target group + write governance (active vs. tentative) for an
  explicit shared-tier ``memory_store``;
- produces the provenance labels rendered into the prompt context so the LLM
  can weigh "org memory" / "team memory (<name>)" against plain personal facts.

Storage does NOT resolve conflicts across tiers — provenance labelling hands
that judgement to the model.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from itertools import zip_longest
from typing import TYPE_CHECKING, Iterable, TypeVar, cast

from backend.data.tenancy import get_user_team_ids

from .client import derive_group_id, derive_org_group_id, derive_team_group_id

if TYPE_CHECKING:
    from prisma.types import OrgMemberWhereInput, TeamMemberWhereInput

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Provenance labels rendered into <temporal_context> / search results.
# Personal facts stay unlabelled (label is None) so the common case reads
# cleanly; shared-tier facts are prefixed so the model knows the source.
ORG_LABEL = "org memory"


def team_label(name: str | None) -> str:
    """Render a team's provenance label, including its name when known."""
    return f"team memory ({name})" if name else "team memory"


class MemoryTier(str, Enum):
    personal = "personal"
    team = "team"
    org = "org"


class TierError(Exception):
    """Raised when a shared-tier request is ambiguous or unauthorized.

    Carries a user-facing ``message`` safe to surface to the LLM/user.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class TierTarget:
    """A single group a read fans out to, plus its provenance label."""

    group_id: str
    tier: MemoryTier
    label: str | None  # None → personal (rendered unlabelled)
    team_id: str | None = None


# ---------------------------------------------------------------------------
# Prisma-backed membership + governance (mock at this boundary in tests)
# ---------------------------------------------------------------------------


async def resolve_team_name(team_id: str) -> str | None:
    """Best-effort team display name for provenance labels."""
    from backend.data.db import prisma

    try:
        team = await prisma.team.find_unique(where={"id": team_id})
    except Exception:
        logger.debug("Failed to resolve team name for %s", team_id, exc_info=True)
        return None
    return team.name if team else None


async def resolve_team_names(team_ids: list[str]) -> dict[str, str]:
    """Batch team-id → name lookup for labelling a fan-out of teams."""
    if not team_ids:
        return {}
    from backend.data.db import prisma

    try:
        teams = await prisma.team.find_many(where={"id": {"in": team_ids}})
    except Exception:
        logger.debug("Failed to batch-resolve team names", exc_info=True)
        return {}
    return {t.id: t.name for t in teams}


async def is_org_admin(user_id: str, org_id: str) -> bool:
    """True if *user_id* is an ACTIVE admin or owner of *org_id*."""
    from backend.data.db import prisma

    member = await prisma.orgmember.find_first(
        where=cast(
            "OrgMemberWhereInput",
            {"userId": user_id, "orgId": org_id, "status": "ACTIVE"},
        )
    )
    return bool(member and (member.isAdmin or member.isOwner))


async def get_team_membership(user_id: str, team_id: str, org_id: str):
    """Return the ACTIVE TeamMember row for (user, team) within *org_id*.

    Scoped to the org so a team id from a different org can never be
    used to smuggle a write/read into the wrong tenant.
    """
    from backend.data.db import prisma

    return await prisma.teammember.find_first(
        where=cast(
            "TeamMemberWhereInput",
            {
                "userId": user_id,
                "teamId": team_id,
                "status": "ACTIVE",
                "Team": {"is": {"orgId": org_id}},
            },
        )
    )


async def hold_buffer_enabled(org_id: str) -> bool:
    """Whether the org's shared-write hold buffer is on (default TRUE).

    Reads ``Organization.settings["memory"]["holdBuffer"]``. When on,
    non-admin shared-tier writes land ``tentative`` for later admin
    review; when off, all permitted writes land ``active``. Any missing
    setting or read failure defaults to ON (the safer, review-gated
    behavior).
    """
    from backend.data.db import prisma

    try:
        org = await prisma.organization.find_unique(where={"id": org_id})
    except Exception:
        logger.debug("Failed to read org settings for %s", org_id, exc_info=True)
        return True
    if org is None:
        return True

    settings = org.settings
    if isinstance(settings, str):
        import json

        try:
            settings = json.loads(settings)
        except (json.JSONDecodeError, TypeError):
            return True
    if not isinstance(settings, dict):
        return True
    memory = settings.get("memory")
    if not isinstance(memory, dict):
        return True
    return bool(memory.get("holdBuffer", True))


async def resolve_store_team(
    user_id: str,
    org_id: str,
    session_team_id: str | None,
    explicit_team_id: str | None,
):
    """Resolve + authorize the target team for a ``tier="team"`` write.

    Precedence: explicit ``team_id`` arg → session's team → the user's
    single team (only when they have exactly one). Raises ``TierError``
    with a clear message when the target is ambiguous, absent, or the
    user is not an ACTIVE member of it. Returns the TeamMember row so
    the caller can read ``teamId`` and ``isAdmin`` without a second query.
    """
    target = explicit_team_id or session_team_id
    if not target:
        active_ids = await get_user_team_ids(user_id, org_id)
        if len(active_ids) == 1:
            target = active_ids[0]
        elif not active_ids:
            raise TierError(
                "You are not an active member of any team in this organization, "
                "so team memory is unavailable."
            )
        else:
            raise TierError(
                "You belong to multiple teams — specify which one with the "
                "'team_id' argument when storing to team memory."
            )

    membership = await get_team_membership(user_id, target, org_id)
    if membership is None:
        raise TierError(
            "You are not an active member of the specified team, so you cannot "
            "store to its team memory."
        )
    return membership


# ---------------------------------------------------------------------------
# Read target resolution
# ---------------------------------------------------------------------------


async def resolve_warm_targets(
    user_id: str,
    organization_id: str | None,
    session_team_id: str | None,
) -> list[TierTarget]:
    """Groups the session-start warm-context prefetch may read.

    Personal always; org when the session carries an organization; the
    session's team ONLY when the session is team-tagged AND the user is
    an ACTIVE member of it (untagged sessions skip the team tier).
    """
    targets = [TierTarget(derive_group_id(user_id), MemoryTier.personal, None)]

    if organization_id:
        targets.append(
            TierTarget(derive_org_group_id(organization_id), MemoryTier.org, ORG_LABEL)
        )

        if session_team_id:
            active_ids = await get_user_team_ids(user_id, organization_id)
            if session_team_id in active_ids:
                name = await resolve_team_name(session_team_id)
                targets.append(
                    TierTarget(
                        derive_team_group_id(session_team_id),
                        MemoryTier.team,
                        team_label(name),
                        session_team_id,
                    )
                )

    return targets


async def resolve_search_targets(
    user_id: str,
    organization_id: str | None,
    tier: str,
) -> list[TierTarget]:
    """Groups an explicit ``memory_search`` may read for the given tier.

    ``all`` (default) unions personal + org + EVERY ACTIVE team the user
    belongs to; ``personal``/``org``/``team`` restrict to that tier.
    Team tiers only ever include the user's ACTIVE memberships. Raises
    ``TierError`` when an org/team tier is requested without an org on
    the session.
    """
    tier = tier or "all"
    include_personal = tier in ("all", "personal")
    include_org = tier in ("all", "org")
    include_team = tier in ("all", "team")

    if tier in ("org", "team") and not organization_id:
        raise TierError(
            "This session is not attached to an organization, so org/team "
            "memory is unavailable. Use tier='personal' or attach the session "
            "to an organization."
        )

    targets: list[TierTarget] = []

    if include_personal:
        targets.append(TierTarget(derive_group_id(user_id), MemoryTier.personal, None))

    if include_org and organization_id:
        targets.append(
            TierTarget(derive_org_group_id(organization_id), MemoryTier.org, ORG_LABEL)
        )

    if include_team and organization_id:
        active_ids = await get_user_team_ids(user_id, organization_id)
        names = await resolve_team_names(active_ids)
        for team_id in active_ids:
            targets.append(
                TierTarget(
                    derive_team_group_id(team_id),
                    MemoryTier.team,
                    team_label(names.get(team_id)),
                    team_id,
                )
            )

    return targets


# ---------------------------------------------------------------------------
# Budget merge — personal keeps >= half; shared tiers share the rest
# ---------------------------------------------------------------------------


def _round_robin(lists: list[list[T]]) -> Iterable[T]:
    """Fairly interleave already-ranked per-tier lists.

    Each input list is assumed rerank-ordered (best first); round-robin
    interleaving keeps every shared tier's top hits near the front so the
    shared budget is shared "by rerank score" without a cross-tier score
    (graphiti's search API does not expose comparable scores across
    separate group queries).
    """
    sentinel: object = object()
    for group in zip_longest(*lists, fillvalue=sentinel):
        for item in group:
            if item is not sentinel:
                yield item  # type: ignore[misc]


def merge_tiered(
    personal: list[T],
    shared: list[tuple[str | None, list[T]]],
    total: int,
) -> list[tuple[T, str | None]]:
    """Merge personal + labelled shared results under a total budget.

    Personal keeps at least ``total // 2`` (the floor) and absorbs any
    budget the shared tiers don't use; the remainder is filled by
    round-robin over the shared tiers, each labelled with its provenance.
    Returns ``[(item, label)]`` in render order — personal first
    (``label`` None), then interleaved shared.
    """
    if total <= 0:
        return []

    half = total // 2
    shared_available = sum(len(items) for _, items in shared)
    personal_take = min(len(personal), max(half, total - shared_available))

    merged: list[tuple[T, str | None]] = [
        (item, None) for item in personal[:personal_take]
    ]
    remaining = total - len(merged)
    if remaining <= 0 or not shared:
        return merged

    labelled: list[list[tuple[T, str | None]]] = [
        [(item, label) for item in items] for label, items in shared if items
    ]
    for pair in _round_robin(labelled):
        if remaining <= 0:
            break
        merged.append(pair)
        remaining -= 1

    return merged
