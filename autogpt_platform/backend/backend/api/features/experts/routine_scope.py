"""Personal routine identity is durable and independent of navigation."""

import prisma.models
import prisma.types

from backend.data.memory_scope import get_personal_memory_organization
from backend.data.tenancy import ResourceAccess, has_live_resource_access

# One class for a missing routine everywhere, so the RPC layer answers it at
# once instead of retrying, and the API maps it to a 404.
from backend.util.exceptions import RoutineNotFoundError


async def personal_routine_scope(
    user_id: str, organization_id: str | None, access: ResourceAccess = "view"
) -> tuple[str, str]:
    org_id = organization_id or await get_personal_memory_organization(user_id)
    team = await prisma.models.Team.prisma().find_first(
        where={"orgId": org_id, "isDefault": True, "archivedAt": None}
    )
    if team is None or not await has_live_resource_access(
        user_id, org_id, team.id, access
    ):
        raise RoutineNotFoundError("Routine organization is unavailable")
    return org_id, team.id


async def personal_owner_where(
    user_id: str, organization_id: str | None
) -> prisma.types.ExpertRoutineWhereInput:
    org_id, _ = await personal_routine_scope(user_id, organization_id)
    where: prisma.types.ExpertRoutineWhereInput = {
        "userId": user_id,
        "expertId": None,
        "organizationId": org_id,
    }
    personal = await get_personal_memory_organization(user_id)
    if org_id == personal:
        del where["organizationId"]
        where["OR"] = [{"organizationId": org_id}, {"organizationId": None}]
    return where


async def saved_personal_org(row: prisma.models.ExpertRoutine) -> str | None:
    if row.expertId is not None or row.userId is None:
        return None
    return row.organizationId or await get_personal_memory_organization(row.userId)


async def validate_personal_session(
    user_id: str, organization_id: str, session_id: str | None
) -> None:
    if session_id is None:
        return
    session = await prisma.models.ChatSession.prisma().find_first(
        where={"id": session_id, "userId": user_id, "expertId": None}
    )
    if session is None:
        raise ValueError("The routine's chat was not found in this organization.")
    session_org = session.organizationId or await get_personal_memory_organization(
        user_id
    )
    if session_org != organization_id:
        raise ValueError("The routine's chat was not found in this organization.")
