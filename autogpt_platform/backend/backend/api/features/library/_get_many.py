import asyncio
import logging

import prisma.models
from pydantic import BaseModel

import backend.api.features.library.model as library_model
import backend.data.graph as graph_db
from backend.data.includes import library_agent_include

from ._schedule_info import _fetch_schedule_info
from .db import _fetch_marketplace_details

logger = logging.getLogger(__name__)

MAX_LIBRARY_AGENTS_PER_BATCH = 50


async def get_library_agents_by_ids(
    ids: list[str], user_id: str
) -> list[library_model.LibraryAgent]:
    """What ``get_library_agent`` returns, for several agents in one read.

    Workflow card grids need the full view (credentials and trigger info for
    the run button). Fetching it card by card multiplied every scheduler read
    by the number of cards: one AutoPilot page load sent 205 at once. Here the
    schedule info is read once for the whole batch.

    Results keep the order of ``ids``. Unknown, deleted, foreign or unparseable
    agents are left out rather than failing the batch, so one bad card can't
    blank the grid.
    """
    unique_ids = list(dict.fromkeys(ids))
    if not unique_ids:
        return []
    rows = await prisma.models.LibraryAgent.prisma().find_many(
        where={"id": {"in": unique_ids}, "userId": user_id, "isDeleted": False},
        include=library_agent_include(user_id),
    )
    schedule_info, details = await asyncio.gather(
        _fetch_schedule_info(user_id),
        asyncio.gather(*(_load_details(row) for row in rows)),
    )

    by_id: dict[str, library_model.LibraryAgent] = {}
    for detail in details:
        if detail is None:
            continue
        try:
            by_id[detail.row.id] = library_model.LibraryAgent.from_db(
                detail.row,
                sub_graphs=detail.sub_graphs,
                store_listing=detail.store_listing,
                profile=detail.profile,
                schedule_info=schedule_info,
            )
        except Exception as e:
            logger.error(f"Error parsing LibraryAgent #{detail.row.id}: {e}")
    return [by_id[agent_id] for agent_id in unique_ids if agent_id in by_id]


class _AgentDetails(BaseModel):
    row: prisma.models.LibraryAgent
    store_listing: prisma.models.StoreListing | None
    profile: prisma.models.Profile | None
    sub_graphs: list[prisma.models.AgentGraph]


async def _load_details(row: prisma.models.LibraryAgent) -> _AgentDetails | None:
    if not row.AgentGraph:
        return None
    try:
        (store_listing, profile), sub_graphs = await asyncio.gather(
            _fetch_marketplace_details(row.AgentGraph.id),
            graph_db.get_sub_graphs(row.AgentGraph),
        )
    except Exception as e:
        logger.error(f"Error loading LibraryAgent #{row.id}: {e}")
        return None
    return _AgentDetails(
        row=row, store_listing=store_listing, profile=profile, sub_graphs=sub_graphs
    )
