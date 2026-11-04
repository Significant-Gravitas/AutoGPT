"""One slot for personal work: a personal organization's default team.

The database sends every row a personal organization writes without a team to
its default team (trigger ``a0_normalize_personal_team``, migration
20261104140000). A caller that builds ``(personal org, None)`` by hand and then
matches rows on the exact pair would find nothing, so exact-match checks read
the pair through ``canonical_personal_scope`` first.
"""

from prisma import Prisma
from prisma.models import Team

from backend.data.db import prisma


async def canonical_personal_scope(
    organization_id: str | None,
    team_id: str | None,
    client: Prisma | None = None,
) -> tuple[str | None, str | None]:
    """The scope the database stores for ``(organization_id, team_id)``.

    A personal organization without a team is its default team; any other
    scope is returned as given. Matches ``personal_default_team_id`` in the
    migration: a live personal organization's live default team, oldest first.
    """
    if organization_id is None or team_id is not None:
        return organization_id, team_id
    default_team = await Team.prisma(client or prisma).find_first(
        where={
            "orgId": organization_id,
            "isDefault": True,
            "archivedAt": None,
            "Org": {"is": {"isPersonal": True, "deletedAt": None}},
        },
        order=[{"createdAt": "asc"}, {"id": "asc"}],
    )
    return organization_id, default_team.id if default_team else None
