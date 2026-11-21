"""Stored workflow pages: one per library entry, with the design it replaced."""

from datetime import datetime
from typing import Any

from prisma.models import LibraryAgent, WorkflowPage
from pydantic import BaseModel

from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson


class WorkflowPageRecord(BaseModel):
    library_agent_id: str
    design: dict[str, Any]
    previous_design: dict[str, Any] | None = None
    updated_at: datetime


async def get_workflow_page(
    user_id: str, library_agent_id: str
) -> WorkflowPageRecord | None:
    row = await WorkflowPage.prisma().find_first(
        where={"libraryAgentId": library_agent_id, "userId": user_id}
    )
    return _record(row) if row else None


async def save_workflow_page(
    user_id: str, library_agent_id: str, design: dict[str, Any]
) -> WorkflowPageRecord:
    """Store *design* for the user's library entry, keeping the one it replaces."""
    owned = await LibraryAgent.prisma().find_first(
        where={"id": library_agent_id, "userId": user_id, "isDeleted": False}
    )
    if owned is None:
        raise NotFoundError(f"Library agent #{library_agent_id} not found")
    current = await get_workflow_page(user_id, library_agent_id)
    row = await WorkflowPage.prisma().upsert(
        where={"libraryAgentId": library_agent_id},
        data={
            "create": {
                "libraryAgentId": library_agent_id,
                "userId": user_id,
                "design": SafeJson(design),
            },
            "update": {
                "design": SafeJson(design),
                **({"previousDesign": SafeJson(current.design)} if current else {}),
            },
        },
    )
    return _record(row)


async def restore_workflow_page(
    user_id: str, library_agent_id: str
) -> WorkflowPageRecord:
    """Swap the design and the one before it, so a restore can be undone."""
    current = await get_workflow_page(user_id, library_agent_id)
    if current is None:
        raise NotFoundError(f"No page for library agent #{library_agent_id}")
    if current.previous_design is None:
        raise ValueError("This page has no earlier design to restore")
    row = await WorkflowPage.prisma().update(
        where={"libraryAgentId": library_agent_id},
        data={
            "design": SafeJson(current.previous_design),
            "previousDesign": SafeJson(current.design),
        },
    )
    if row is None:
        raise NotFoundError(f"No page for library agent #{library_agent_id}")
    return _record(row)


async def delete_workflow_page(user_id: str, library_agent_id: str) -> bool:
    deleted = await WorkflowPage.prisma().delete_many(
        where={"libraryAgentId": library_agent_id, "userId": user_id}
    )
    return deleted > 0


def _record(row: WorkflowPage) -> WorkflowPageRecord:
    return WorkflowPageRecord(
        library_agent_id=row.libraryAgentId,
        design=_as_dict(row.design),
        previous_design=_as_dict(row.previousDesign) if row.previousDesign else None,
        updated_at=row.updatedAt,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
