"""What stands between a hired expert and its scheduled work.

A workflow "needs setup" when its roster cadence could not become a schedule
at install time, almost always because the graph needs a credential the
expert cannot reach. This names that gap per workflow so the Team page can
offer the one fix that clears it: connect a service, or allow the expert to
use one the user already has.
"""

import logging
from enum import Enum

import prisma.models
import prisma.types

from backend.api.features.experts.credentials import (
    _user_credentials,
    expert_allowed_credential_ids,
    filter_credentials_for_expert,
)
from backend.api.features.experts.models import ExpertSetupItem
from backend.data.model import Credentials

logger = logging.getLogger(__name__)

_INCLUDE: prisma.types.ExpertInclude = {
    "Workflows": {
        "include": {
            "LibraryAgent": {"include": {"AgentGraph": True}},
            "StoreListingVersion": True,
        }
    }
}


async def list_setup_items(user_id: str) -> list[ExpertSetupItem]:
    experts = await prisma.models.Expert.prisma().find_many(
        where={"ownerUserId": user_id, "isTemplate": False, "isArchived": False},
        include=_INCLUDE,
        order={"createdAt": "asc"},
    )
    pending = [
        (expert, workflow)
        for expert in experts
        for workflow in expert.Workflows or []
        if workflow.scheduleCron and workflow.scheduleId is None
    ]
    if not pending:
        return []

    credentials = await _user_credentials(user_id)
    reachable_by_expert: dict[str, list[Credentials]] = {}
    items: list[ExpertSetupItem] = []
    for expert, workflow in pending:
        if expert.id not in reachable_by_expert:
            allowed = set(await expert_allowed_credential_ids(user_id, expert.id))
            reachable_by_expert[expert.id] = filter_credentials_for_expert(
                credentials, allowed
            )
        items.extend(
            await _workflow_items(
                user_id, expert, workflow, credentials, reachable_by_expert[expert.id]
            )
        )
    return items


async def _workflow_items(
    user_id: str,
    expert: prisma.models.Expert,
    workflow: prisma.models.ExpertWorkflow,
    credentials: list[Credentials],
    reachable: list[Credentials],
) -> list[ExpertSetupItem]:
    # Imported here: the matcher lives beside the executor, which imports this
    # package at module load.
    from backend.copilot.tools.utils import find_matching_credential
    from backend.data.graph import get_graph

    def item(**overrides) -> ExpertSetupItem:
        return ExpertSetupItem(
            expert_id=expert.id,
            expert_name=expert.name,
            expert_avatar_url=expert.avatarUrl,
            workflow_id=workflow.id,
            workflow_name=_workflow_name(workflow),
            library_agent_id=workflow.libraryAgentId,
            **overrides,
        )

    library_agent = workflow.LibraryAgent
    if library_agent is None:
        return [item(providers=[], resolution="workflow")]
    try:
        graph = await get_graph(
            library_agent.agentGraphId,
            library_agent.agentGraphVersion,
            user_id,
            include_subgraphs=True,
        )
    except Exception:
        logger.warning(
            f"Could not load graph for workflow #{workflow.id} on expert "
            f"#{expert.id}; reporting it as needing its schedule",
            exc_info=True,
        )
        graph = None
    if graph is None:
        return [item(providers=[], resolution="workflow")]

    items: list[ExpertSetupItem] = []
    for field_info, _, is_required in graph.regular_credentials_inputs.values():
        # An optional credential never blocks a schedule: the scheduler skips
        # the node instead.
        if not is_required:
            continue
        if find_matching_credential(reachable, field_info):
            continue
        candidate = find_matching_credential(credentials, field_info)
        items.append(
            item(
                providers=sorted(_provider_slug(p) for p in field_info.provider),
                resolution="allow" if candidate else "connect",
                credential_id=candidate.id if candidate else None,
            )
        )
    # Everything is reachable yet the schedule is still missing: whatever
    # failed was not a credential, so hand the user the workflow itself.
    return items or [item(providers=[], resolution="workflow")]


def _workflow_name(workflow: prisma.models.ExpertWorkflow) -> str | None:
    listing = workflow.StoreListingVersion
    if listing is not None:
        return listing.name
    library_agent = workflow.LibraryAgent
    if library_agent is None:
        return None
    graph = library_agent.AgentGraph
    return library_agent.name or (graph.name if graph else None)


def _provider_slug(provider: object) -> str:
    return provider.value if isinstance(provider, Enum) else str(provider)
