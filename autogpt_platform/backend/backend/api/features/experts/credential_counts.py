import asyncio

import prisma.models

from backend.api.features.experts.credentials import _seed_if_needed, _user_credentials


async def expert_credential_providers(
    user_id: str, experts: list[prisma.models.Expert]
) -> dict[str, list[str]]:
    """The provider of every live credential each owned expert may use.

    One entry per grant, so ``len`` is the credential count and a provider
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
    provider_by_id = {credential.id: credential.provider for credential in credentials}
    providers: dict[str, list[str]] = {}
    for grant in grants:
        provider = provider_by_id.get(grant.credentialId)
        if provider is not None:
            providers.setdefault(grant.expertId, []).append(provider)
    return providers
