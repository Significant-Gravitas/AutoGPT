"""Stored pages: the user's own, each with the design it replaced and a revision."""

from datetime import datetime
from typing import Any

from prisma.models import LibraryAgent, Page
from pydantic import BaseModel

from backend.data.db import transaction
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

MAX_PAGES_PER_USER = 50


class StaleRevisionError(ValueError):
    """The page changed after the version a save was based on."""


class PageRecord(BaseModel):
    id: str
    title: str
    design: dict[str, Any]
    previous_design: dict[str, Any] | None = None
    revision: int
    pinned_library_agent_id: str | None = None
    chat_session_id: str | None = None
    updated_at: datetime


async def list_pages(user_id: str) -> list[PageRecord]:
    rows = await Page.prisma().find_many(
        where={"userId": user_id}, order={"updatedAt": "desc"}, take=MAX_PAGES_PER_USER
    )
    return [_record(row) for row in rows]


async def get_page(user_id: str, page_id: str) -> PageRecord | None:
    row = await Page.prisma().find_first(where={"id": page_id, "userId": user_id})
    return _record(row) if row else None


async def get_pinned_page(user_id: str, library_agent_id: str) -> PageRecord | None:
    row = await Page.prisma().find_first(
        where={"pinnedLibraryAgentId": library_agent_id, "userId": user_id}
    )
    return _record(row) if row else None


async def create_page(
    user_id: str,
    title: str,
    design: dict[str, Any],
    *,
    chat_session_id: str | None = None,
) -> PageRecord:
    if await Page.prisma().count(where={"userId": user_id}) >= MAX_PAGES_PER_USER:
        raise ValueError(
            f"You have the most pages one account can keep ({MAX_PAGES_PER_USER})"
        )
    row = await Page.prisma().create(
        data={
            "userId": user_id,
            "title": title,
            "design": SafeJson(design),
            **({"chatSessionId": chat_session_id} if chat_session_id else {}),
        }
    )
    return _record(row)


async def save_page_design(
    user_id: str, page_id: str, title: str, design: dict[str, Any], *, based_on: int
) -> PageRecord:
    """Store *design* over revision *based_on*, keeping the one it replaces."""
    current = await get_page(user_id, page_id)
    if current is None:
        raise NotFoundError(f"Page #{page_id} not found")
    updated = await Page.prisma().update_many(
        where={"id": page_id, "userId": user_id, "revision": based_on},
        data={
            "title": title,
            "design": SafeJson(design),
            "previousDesign": SafeJson(current.design),
            "revision": {"increment": 1},
        },
    )
    if updated == 0:
        raise StaleRevisionError(
            f"This page changed since revision {based_on}; it is at revision {current.revision} now"
        )
    saved = await get_page(user_id, page_id)
    assert saved is not None
    return saved


async def restore_previous(user_id: str, page_id: str) -> PageRecord:
    """Swap the design and the one before it, so a restore can be undone."""
    current = await get_page(user_id, page_id)
    if current is None:
        raise NotFoundError(f"Page #{page_id} not found")
    if current.previous_design is None:
        raise ValueError("This page has no earlier version to restore")
    return await save_page_design(
        user_id,
        page_id,
        str(current.previous_design.get("title") or current.title),
        current.previous_design,
        based_on=current.revision,
    )


async def set_page_pin(
    user_id: str, page_id: str, library_agent_id: str | None
) -> PageRecord:
    """Pin the page to the user's workflow, moving the pin off any other page,
    or unpin it with None."""
    async with transaction() as tx:
        page = await Page.prisma(tx).find_first(
            where={"id": page_id, "userId": user_id}
        )
        if page is None:
            raise NotFoundError(f"Page #{page_id} not found")
        if library_agent_id is not None:
            owned = await LibraryAgent.prisma(tx).find_first(
                where={"id": library_agent_id, "userId": user_id, "isDeleted": False}
            )
            if owned is None:
                raise NotFoundError(f"Library agent #{library_agent_id} not found")
            await Page.prisma(tx).update_many(
                where={"pinnedLibraryAgentId": library_agent_id, "userId": user_id},
                data={"pinnedLibraryAgentId": None},
            )
        row = await Page.prisma(tx).update(
            where={"id": page_id}, data={"pinnedLibraryAgentId": library_agent_id}
        )
    assert row is not None
    return _record(row)


async def delete_page(user_id: str, page_id: str) -> bool:
    deleted = await Page.prisma().delete_many(where={"id": page_id, "userId": user_id})
    return deleted > 0


def _record(row: Page) -> PageRecord:
    return PageRecord(
        id=row.id,
        title=row.title,
        design=_as_dict(row.design),
        previous_design=_as_dict(row.previousDesign) if row.previousDesign else None,
        revision=row.revision,
        pinned_library_agent_id=row.pinnedLibraryAgentId,
        chat_session_id=row.chatSessionId,
        updated_at=row.updatedAt,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
