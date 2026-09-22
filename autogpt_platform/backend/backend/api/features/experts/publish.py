"""Publishing one of your own experts as a marketplace template.

Admin-only and deliberately small: no submission, no review queue, no version
history. An admin publishes an expert they own and the template appears; publish
it again and the same template is refreshed in place. ``publishedFromExpertId``
is what makes "in place" possible, and its uniqueness is what stops a second
template from ever appearing for the same expert.

Every workflow must resolve to a store listing version a hire could install —
approved, available, not deleted, on a listing that is not deleted — before
anything is written. A template whose workflow pointed at a private graph, or
at a version the library would refuse, would install nothing for whoever hired
it, so such an agent is a 400 naming it — not a template that quietly arrives
broken.

The built package is stored beside the template as bytes, so downloading it and
hiring it both read exactly what was published rather than rebuilding from an
expert that has since moved on. Its workflow references are the same versions
the template's rows point at, so Download → Import lands on the marketplace
agent that Hire installs.
"""

import logging
from typing import cast

import prisma.models
import prisma.types
from fastapi.concurrency import run_in_threadpool
from prisma import Base64

from backend.api.features.experts.expert_zip import zip_from_package
from backend.api.features.experts.listing_versions import (
    installable_active_version,
    installable_version,
)
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


async def publish_expert(
    row: prisma.models.Expert, *, user_id: str
) -> prisma.models.Expert:
    """Create or refresh the marketplace template for *row*, the admin's own
    expert as loaded by ``get_owned_expert_row``. *user_id* is that admin: the
    package is built for its owner, exactly as a download of it would be."""
    versions = await _installable_versions(row)
    # Packaged with the references the template rows will carry, so the file
    # names the same marketplace version a hire installs — not whatever the
    # expert's row happened to point at when the agent was installed.
    package = await build_expert_package(_with_versions(row, versions), user_id=user_id)
    # Deflating up to 50 skills is CPU, not IO: on the event loop it would stall
    # every other request this worker is serving. The download route offloads the
    # same call for the same reason.
    package_bytes = await run_in_threadpool(zip_from_package, package)
    fields = _template_fields(row, package)
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
                data={**fields, "publishedFromExpertId": row.id}
            )
            template_id = created.id
        await tx.expertpublishedpackage.upsert(
            where={"expertId": template_id},
            data={
                "create": {
                    "expertId": template_id,
                    "package": Base64.encode(package_bytes),
                },
                "update": {"package": Base64.encode(package_bytes)},
            },
        )
        # Replaced rather than merged: a workflow the admin removed from the
        # expert must disappear from the template too.
        await tx.expertworkflow.delete_many(where={"expertId": template_id})
        for workflow in row.Workflows or []:
            await tx.expertworkflow.create(
                data={
                    "expertId": template_id,
                    "storeListingVersionId": versions[workflow.id].id,
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
    row: prisma.models.Expert, package: ExpertPackage
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
        # By name, as `store_user_skill` records a skill on every other expert
        # row; the folder slug is the archive's business, not the column's.
        "skills": [card.name for card in manifest.skills],
        # Both halves of the write, so a republish restores the row's defining
        # columns rather than assuming they still hold — the same reason
        # `isArchived` is here rather than only on the create.
        "isTemplate": True,
        "isArchived": False,
    }


def _with_versions(
    row: prisma.models.Expert,
    versions: dict[str, prisma.models.StoreListingVersion],
) -> prisma.models.Expert:
    """*row* with each workflow pointed at the version it resolved to, so the
    exporter packages that reference (id, slug, creator) and not the stale
    or missing one the row was installed with."""
    return row.model_copy(
        update={
            "Workflows": [
                workflow.model_copy(
                    update={
                        "storeListingVersionId": versions[workflow.id].id,
                        "StoreListingVersion": versions[workflow.id],
                    }
                )
                for workflow in row.Workflows or []
            ]
        }
    )


async def _installable_versions(
    row: prisma.models.Expert,
) -> dict[str, prisma.models.StoreListingVersion]:
    """An installable store listing version per workflow row, or the names of
    the ones that have none. Checked before anything is written."""
    resolved: dict[str, prisma.models.StoreListingVersion] = {}
    unpublished: list[str] = []
    for workflow in row.Workflows or []:
        version = await _installable_workflow_version(workflow)
        if version is None:
            unpublished.append(_workflow_name(workflow))
            continue
        resolved[workflow.id] = version
    if unpublished:
        raise UnpublishedWorkflowsError(unpublished)
    return resolved


async def _installable_workflow_version(
    row: prisma.models.ExpertWorkflow,
) -> prisma.models.StoreListingVersion | None:
    """The version a hire of the template would install for this workflow.

    The row's own reference first, held to the library's installability test
    rather than trusted: an agent whose version was since deleted or hidden
    must block the publish, not ship in a template nobody can hire. Failing
    that, the admin's own listing for the agent's graph — for an agent they
    published after installing it on the expert — through the same test.
    """
    if row.storeListingVersionId:
        version = await installable_version(row.storeListingVersionId)
        if version is not None:
            return version
    agent = row.LibraryAgent
    if agent is None:
        return None
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={"agentGraphId": agent.agentGraphId, "isDeleted": False}
    )
    return await installable_active_version(listing)


def _workflow_name(row: prisma.models.ExpertWorkflow) -> str:
    """What to call the agent in the message telling the admin to publish it.
    A user-created agent's name is on the graph, not on the library row."""
    agent = row.LibraryAgent
    if agent is None:
        return "An agent"
    graph = agent.AgentGraph
    return agent.name or (graph.name if graph else None) or "An agent"
