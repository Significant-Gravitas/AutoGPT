"""Per-expert credential grants — which of the owner's integrations an expert may use.

Deny-by-default: an expert reaches only the credentials granted to it here.
That rule cannot apply retroactively, so an expert whose ``credentialsSeededAt``
is still null gets its allow-list seeded on first read from the credentials its
installed workflows actually resolve to. Once stamped, the list is the user's
to curate — a revoked credential stays revoked, and installing a workflow grants
what that workflow needs so adding one never produces a silently broken expert.

System credentials (the platform's own LLM keys, built from settings rather than
stored per user) are never granted and never filtered: every expert may use them,
or no expert could run a single LLM block.
"""

import logging
from datetime import datetime, timezone

import prisma.models

from backend.api.features.experts.models import ExpertCredentialRef
from backend.data.model import Credentials
from backend.integrations.credentials_store import is_system_credential
from backend.util.exceptions import ExpertNotFoundError

logger = logging.getLogger(__name__)

_WORKFLOW_INCLUDE = {"Workflows": {"include": {"LibraryAgent": True}}}


async def _owned_expert(user_id: str, expert_id: str) -> prisma.models.Expert:
    row = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        },
        include=_WORKFLOW_INCLUDE,  # type: ignore[arg-type]
    )
    if row is None:
        raise ExpertNotFoundError(f"Expert #{expert_id} not found")
    return row


async def _user_credentials(user_id: str) -> list[Credentials]:
    from backend.integrations.creds_manager import IntegrationCredentialsManager

    return await IntegrationCredentialsManager().store.get_all_creds(user_id)


async def _derive_from_workflows(
    user_id: str, expert: prisma.models.Expert
) -> tuple[dict[str, str], bool]:
    """Credential id → provider for everything this expert's workflows resolve to.

    Returns ``(derived, is_complete)``. ``is_complete`` is False when any
    workflow failed to resolve, which keeps the seed pending: a transient
    graph-load failure must not be frozen in as "this expert reaches nothing",
    because enforcement would then block every run it has until the user
    noticed and re-granted by hand.
    """
    from backend.copilot.tools.utils import match_user_credentials_to_graph
    from backend.data.graph import get_graph

    derived: dict[str, str] = {}
    is_complete = True
    for workflow in expert.Workflows or []:
        library_agent = workflow.LibraryAgent
        if library_agent is None:
            continue
        try:
            graph = await get_graph(
                library_agent.agentGraphId,
                library_agent.agentGraphVersion,
                user_id,
                include_subgraphs=True,
            )
            # A workflow whose graph is gone resolves to nothing and always
            # will; that is a complete answer, not a failure to retry.
            if graph is None:
                continue
            matched, _ = await match_user_credentials_to_graph(user_id, graph)
        except Exception:
            logger.warning(
                f"Could not derive credentials for workflow #{workflow.id} on "
                f"expert #{expert.id}; leaving the seed pending",
                exc_info=True,
            )
            is_complete = False
            continue
        for meta in matched.values():
            if not is_system_credential(meta.id):
                derived[meta.id] = str(meta.provider)
    return derived, is_complete


async def _seed_if_needed(user_id: str, expert: prisma.models.Expert) -> None:
    """Seed the allow-list from the expert's workflows, once.

    An incomplete derivation writes what it found but leaves the expert
    unstamped, so the next read retries the workflows that failed. Re-deriving
    is safe because it only ever adds grants — see ``_stamp_seeded`` for the one
    case where that is not good enough.
    """
    if expert.credentialsSeededAt is not None:
        return
    derived, is_complete = await _derive_from_workflows(user_id, expert)
    if derived:
        await prisma.models.ExpertCredential.prisma().create_many(
            data=[
                {
                    "expertId": expert.id,
                    "credentialId": credential_id,
                    "provider": provider,
                }
                for credential_id, provider in derived.items()
            ],
            skip_duplicates=True,
        )
    if not is_complete:
        return
    # Stamped even when nothing was derived: an expert with no workflows has
    # legitimately been offered nothing, and re-deriving on every read would
    # pay for a graph load per workflow on every header render.
    await _stamp_seeded(expert.id)


async def _stamp_seeded(expert_id: str) -> None:
    """Mark the allow-list as curated, so seeding never runs again.

    Re-seeding is additive, so a pending seed can never undo a *grant* — but it
    can undo a *revoke* by re-deriving the credential the user just removed.
    Revoking therefore finalizes the seed even if derivation never completed:
    an expert missing a grant is a visible, fixable problem, while one that
    silently regains access the user revoked is a broken promise.
    """
    await prisma.models.Expert.prisma().update_many(
        where={"id": expert_id, "credentialsSeededAt": None},
        data={"credentialsSeededAt": datetime.now(timezone.utc)},
    )


async def _grants(expert_id: str) -> list[prisma.models.ExpertCredential]:
    return await prisma.models.ExpertCredential.prisma().find_many(
        where={"expertId": expert_id}, order={"createdAt": "asc"}
    )


def _to_refs(
    grants: list[prisma.models.ExpertCredential],
    credentials: list[Credentials],
) -> list[ExpertCredentialRef]:
    """Grants joined against the owner's live credentials.

    A grant whose credential no longer exists is dropped rather than rendered:
    the row is inert (enforcement matches on id, and the id resolves to
    nothing), and showing a logo for a deleted credential would claim access
    the expert does not have.
    """
    by_id = {c.id: c for c in credentials}
    refs: list[ExpertCredentialRef] = []
    for grant in grants:
        credential = by_id.get(grant.credentialId)
        if credential is None:
            continue
        refs.append(
            ExpertCredentialRef(
                credential_id=credential.id,
                provider=str(credential.provider),
                title=credential.title or str(credential.provider),
                type=str(credential.type),
            )
        )
    return refs


async def list_expert_credentials(
    user_id: str, expert_id: str
) -> list[ExpertCredentialRef]:
    expert = await _owned_expert(user_id, expert_id)
    await _seed_if_needed(user_id, expert)
    grants, credentials = await _grants(expert_id), await _user_credentials(user_id)
    return _to_refs(grants, credentials)


async def grant_expert_credentials(
    user_id: str, expert_id: str, credential_ids: list[str]
) -> list[ExpertCredentialRef]:
    """Grant credentials to an expert. Unknown or system ids are rejected.

    Rejecting rather than ignoring: a silently dropped id would leave the user
    looking at a management list that disagrees with what they just added.
    """
    expert = await _owned_expert(user_id, expert_id)
    credentials = await _user_credentials(user_id)
    by_id = {c.id: c for c in credentials}

    unknown = [
        credential_id
        for credential_id in credential_ids
        if credential_id not in by_id or is_system_credential(credential_id)
    ]
    if unknown:
        raise ValueError(f"Not your credentials: {', '.join(sorted(unknown))}")

    # Seed before granting so the seed cannot later overwrite an explicit add.
    await _seed_if_needed(user_id, expert)
    if credential_ids:
        await prisma.models.ExpertCredential.prisma().create_many(
            data=[
                {
                    "expertId": expert_id,
                    "credentialId": credential_id,
                    "provider": str(by_id[credential_id].provider),
                }
                for credential_id in credential_ids
            ],
            skip_duplicates=True,
        )
    return _to_refs(await _grants(expert_id), credentials)


async def revoke_expert_credential(
    user_id: str, expert_id: str, credential_id: str
) -> list[ExpertCredentialRef]:
    expert = await _owned_expert(user_id, expert_id)
    # Seed first: revoking from a never-seeded expert must remove that one
    # credential, not leave an empty list that the next read seeds straight back.
    await _seed_if_needed(user_id, expert)
    await prisma.models.ExpertCredential.prisma().delete_many(
        where={"expertId": expert_id, "credentialId": credential_id}
    )
    # Finalize even if the seed was left pending by a failed derivation, or the
    # next read would re-derive exactly the credential just revoked.
    await _stamp_seeded(expert_id)
    return _to_refs(await _grants(expert_id), await _user_credentials(user_id))


async def expert_allowed_credential_ids(user_id: str, expert_id: str) -> list[str]:
    """The credential ids *expert_id* may use. Enforcement's source of truth.

    Raises ``ExpertNotFoundError`` for an expert the user does not own, so a
    bad scope fails the run rather than falling through to unrestricted access.
    """
    expert = await _owned_expert(user_id, expert_id)
    await _seed_if_needed(user_id, expert)
    return [grant.credentialId for grant in await _grants(expert_id)]


def filter_credentials_for_expert(
    credentials: list[Credentials], allowed_ids: set[str]
) -> list[Credentials]:
    """Drop credentials an expert has not been granted, keeping system ones."""
    return [
        credential
        for credential in credentials
        if is_system_credential(credential.id) or credential.id in allowed_ids
    ]
