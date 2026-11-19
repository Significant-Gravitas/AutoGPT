import asyncio

import prisma.models

from backend.api.features.experts.credentials import _seed_if_needed, _user_credentials
from backend.integrations.service_identity import service_for_credential


async def expert_credential_providers(
    user_id: str, experts: list[prisma.models.Expert]
) -> dict[str, list[str]]:
    """Service icon ids for each owned expert, one per live granted credential.

    One entry per grant, so ``len`` is the credential count and a service
    granted twice appears twice; the roster dedupes for its logos.
    """
    owned = [expert for expert in experts if expert.ownerUserId == user_id]
    if not owned:
        return {}
    semaphore = asyncio.Semaphore(8)

    async def seed(expert: prisma.models.Expert) -> None:
        async with semaphore:
            await _seed_if_needed(user_id, expert)

    await asyncio.gather(*(seed(expert) for expert in owned))
    grants, credentials = await asyncio.gather(
        prisma.models.ExpertCredential.prisma().find_many(
            where={"expertId": {"in": [expert.id for expert in owned]}},
            # Grant order is what "first-seen" means for the card's logos.
            order=[{"createdAt": "asc"}, {"id": "asc"}],
        ),
        _user_credentials(user_id),
    )
    # One service icon id per grant: the provider slug for a block
    # credential, the catalog icon for an MCP one.
    icon_by_id = {credential.id: _icon_for(credential) for credential in credentials}
    providers: dict[str, list[str]] = {}
    for grant in grants:
        icon = icon_by_id.get(grant.credentialId)
        if icon is not None:
            providers.setdefault(grant.expertId, []).append(icon)
    return providers


def _icon_for(credential) -> str:
    identity = service_for_credential(credential)
    return identity.icon or identity.service
