"""The workflows and experts a page draws on, loaded once per read for its owner."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.api.features.experts.models import Expert
from backend.api.features.library.model import LibraryAgent
from backend.data.db_accessors import experts_db, library_db
from backend.util.exceptions import NotFoundError

from .design import PageDesign

logger = logging.getLogger(__name__)


class PageSources(BaseModel):
    """Each id the design names: its record, None when it is missing, deleted
    or another user's (the page never says which), or in ``failed`` when it
    could not be read just now."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflows: dict[str, LibraryAgent | None] = Field(default_factory=dict)
    experts: dict[str, Expert | None] = Field(default_factory=dict)
    failed: set[str] = Field(default_factory=set)


async def load_sources(
    user_id: str, design: PageDesign, *, extra_workflows: Sequence[str] = ()
) -> PageSources:
    """Load every source *design* names, through readers scoped to *user_id*."""
    workflow_ids = list(dict.fromkeys([*design.workflow_ids(), *extra_workflows]))
    expert_ids = design.expert_ids()
    loaded = await asyncio.gather(
        *(_load_workflow(user_id, ref) for ref in workflow_ids),
        *(_load_expert(user_id, ref) for ref in expert_ids),
        return_exceptions=True,
    )
    sources = PageSources()
    for ref, result in zip([*workflow_ids, *expert_ids], loaded):
        target: dict[str, Any] = (
            sources.workflows if ref in workflow_ids else sources.experts
        )
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


async def _load_expert(user_id: str, expert_id: str) -> Expert | None:
    expert = await experts_db().get_expert(user_id, expert_id)
    if expert is None or expert.is_archived or expert.is_template:
        return None
    return expert
