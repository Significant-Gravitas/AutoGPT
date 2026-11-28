"""The workflows and experts a page draws on, loaded once per read for its owner."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.api.features.experts.models import Expert
from backend.api.features.library.model import LibraryAgent
from backend.data.collections.model import CollectionInfo
from backend.data.db_accessors import collections_db, experts_db, library_db
from backend.util.exceptions import NotFoundError
from backend.util.feature_flag import Flag, is_feature_enabled

from .design import PageDesign

logger = logging.getLogger(__name__)


class PageSources(BaseModel):
    """Each id the design names: its record, None when it is missing, deleted
    or another user's (the page never says which), or in ``failed`` when it
    could not be read just now."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflows: dict[str, LibraryAgent | None] = Field(default_factory=dict)
    experts: dict[str, Expert | None] = Field(default_factory=dict)
    collections: dict[str, CollectionInfo | None] = Field(default_factory=dict)
    failed: set[str] = Field(default_factory=set)


async def load_sources(
    user_id: str, design: PageDesign, *, extra_workflows: Sequence[str] = ()
) -> PageSources:
    """Load every source *design* names, through readers scoped to *user_id*."""
    workflow_ids = list(dict.fromkeys([*design.workflow_ids(), *extra_workflows]))
    expert_ids = design.expert_ids()
    collection_ids = design.collection_ids()
    loaded = await asyncio.gather(
        *(_load_workflow(user_id, ref) for ref in workflow_ids),
        *(_load_expert(user_id, ref) for ref in expert_ids),
        *(_load_collection(user_id, ref) for ref in collection_ids),
        return_exceptions=True,
    )
    sources = PageSources()
    targets: list[dict[str, Any]] = [
        *(sources.workflows for _ in workflow_ids),
        *(sources.experts for _ in expert_ids),
        *(sources.collections for _ in collection_ids),
    ]
    for ref, target, result in zip(
        [*workflow_ids, *expert_ids, *collection_ids], targets, loaded
    ):
        if isinstance(result, BaseException):
            logger.warning(f"Page source {ref} could not be loaded: {result!r}")
            sources.failed.add(ref)
            target[ref] = None
        else:
            target[ref] = result
    return sources


async def _load_workflow(user_id: str, library_agent_id: str) -> LibraryAgent | None:
    try:
        return await library_db().get_library_agent(library_agent_id, user_id)
    except NotFoundError:
        return None


async def _load_collection(user_id: str, collection_id: str) -> CollectionInfo | None:
    if not await is_feature_enabled(Flag.COLLECTIONS, user_id, default=False):
        return None
    found = await collections_db().get_collection(user_id, collection_id)
    # By id only: a section never names a collection by its name.
    return found if found is not None and found.id == collection_id else None


async def _load_expert(user_id: str, expert_id: str) -> Expert | None:
    expert = await experts_db().get_expert(user_id, expert_id)
    if expert is None or expert.is_archived or expert.is_template:
        return None
    return expert
