"""Publishing one of your own experts as a marketplace template.

Admin-only and deliberately small: no submission, no review queue, no version
history. An admin publishes an expert they own and the template appears; publish
it again and the same template is refreshed in place. ``publishedFromExpertId``
is what makes "in place" possible, and its uniqueness is what stops a second
template from ever appearing for the same expert.

Every workflow must resolve to a store listing version first. A template whose
workflow pointed at a private graph would install nothing for whoever hired it,
so an agent that was never published is a 400 telling the admin to publish that
agent — not a template that quietly arrives broken.

The built package is stored on the template as bytes, so downloading it and
hiring it both read exactly what was published rather than rebuilding from an
expert that has since moved on.
"""

import logging
from typing import cast

import prisma.models
import prisma.types
from fastapi.concurrency import run_in_threadpool
from prisma import Base64

from backend.api.features.experts.expert_zip import zip_from_package
from backend.api.features.experts.models import (
    decode_voice_preferences,
    encode_day_one,
    encode_voice_preferences,
)
from backend.api.features.experts.package_export import build_expert_package
from backend.api.features.experts.package_model import ExpertPackage
from backend.data.db import transaction
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


class UnpublishedWorkflowsError(ValueError):
    """An expert cannot become a template while any of its agents is private."""

    def __init__(self, workflows: list[str]):
        super().__init__("Publish this agent first")
        self.workflows = workflows


async def publish_expert(row: prisma.models.Expert) -> prisma.models.Expert:
    """Create or refresh the marketplace template for *row*, the admin's own
    expert as loaded by ``get_owned_expert_row``."""
    version_ids = await _listing_version_ids(row)
    package = await build_expert_package(row)
    # Deflating up to 50 skills is CPU, not IO: on the event loop it would stall
    # every other request this worker is serving. The download route offloads the
    # same call for the same reason.
    package_bytes = await run_in_threadpool(zip_from_package, package)
    fields = _template_fields(row, package, package_bytes)
    async with transaction() as tx:
        existing = await tx.expert.find_first(where={"publishedFromExpertId": row.id})
        if existing:
            await tx.expert.update(
                where={"id": existing.id},
                data=cast(prisma.types.ExpertUpdateInput, fields),
            )
            template_id = existing.id
        else:
            created = await tx.expert.create(
                data={
                    **fields,
                    "isTemplate": True,
                    "publishedFromExpertId": row.id,
                }
            )
            template_id = created.id
        # Replaced rather than merged: a workflow the admin removed from the
        # expert must disappear from the template too.
        await tx.expertworkflow.delete_many(where={"expertId": template_id})
        for workflow in row.Workflows or []:
            await tx.expertworkflow.create(
                data={
                    "expertId": template_id,
                    "storeListingVersionId": version_ids[workflow.id],
                    "scheduleCron": workflow.scheduleCron,
                }
            )
        return await tx.expert.find_unique_or_raise(where={"id": template_id})


async def published_template(expert_id: str) -> prisma.models.Expert | None:
    """The template this expert was published as, if it ever was."""
    return await prisma.models.Expert.prisma().find_first(
        where={"publishedFromExpertId": expert_id, "isArchived": False}
    )


def _template_fields(
    row: prisma.models.Expert, package: ExpertPackage, package_bytes: bytes
) -> prisma.types.ExpertCreateInput:
    """The template's own columns, written the way ``seed._upsert_template``
    writes the roster's, so a published template and a seeded one are the same
    kind of row."""
    manifest = package.manifest
    description, samples = decode_voice_preferences(row.voicePreferences)
    return {
        "name": manifest.identity.name,
        "role": manifest.identity.role,
        "tagline": manifest.identity.tagline,
        "bio": manifest.identity.bio,
        # As-is: the expert's picture already lives in our own media, and a
        # template row references it the same way a hire does.
        "avatarUrl": row.avatarUrl,
        "color": manifest.identity.color,
        "categories": list(manifest.identity.categories),
        "identity": manifest.soul.identity,
        "voicePreferences": encode_voice_preferences(description, samples),
        "boundaries": manifest.soul.boundaries,
        "dayOne": SafeJson(encode_day_one(manifest.day_one)),
        "toolProfile": SafeJson(manifest.tool_profile),
        "skills": [card.slug for card in manifest.skills],
        "isArchived": False,
        "publishedPackage": Base64.encode(package_bytes),
    }


async def _listing_version_ids(row: prisma.models.Expert) -> dict[str, str]:
    """A store listing version per workflow row, or the names of the ones that
    have none. Checked before anything is written."""
    resolved: dict[str, str] = {}
    unpublished: list[str] = []
    for workflow in row.Workflows or []:
        version_id = workflow.storeListingVersionId or await _published_version_id(
            workflow
        )
        if version_id is None:
            unpublished.append(_workflow_name(workflow))
            continue
        resolved[workflow.id] = version_id
    if unpublished:
        raise UnpublishedWorkflowsError(unpublished)
    return resolved


def _workflow_name(row: prisma.models.ExpertWorkflow) -> str:
    """What to call the agent in the message telling the admin to publish it.
    A user-created agent's name is on the graph, not on the library row."""
    agent = row.LibraryAgent
    if agent is None:
        return "An agent"
    graph = agent.AgentGraph
    return agent.name or (graph.name if graph else None) or "An agent"


async def _published_version_id(row: prisma.models.ExpertWorkflow) -> str | None:
    """The listing for the admin's own library agent, for an agent they
    published after installing it on the expert."""
    agent = row.LibraryAgent
    if agent is None:
        return None
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={
            "agentGraphId": agent.agentGraphId,
            "isDeleted": False,
            "hasApprovedVersion": True,
        }
    )
    return listing.activeVersionId if listing else None
