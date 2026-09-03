"""
V2 External API - Pagination

One envelope and one set of query parameters for every list endpoint:
`?limit=&cursor=` in, `{"items", "next_cursor", "total_count"}` out.

The cursor is opaque. It carries a version and a kind, so a cursor from another
endpoint, a stale format, or a hand-made one is rejected rather than silently
read as page 1 — which is the failure a later move to keyset pagination would
otherwise hit.
"""

import base64
import binascii
import json
from typing import Annotated, Any, Generic, Optional, Sequence, TypeVar

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from starlette import status

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

T = TypeVar("T")

CURSOR_VERSION = 1

# An offset this deep is a forged cursor, not a caller walking a list. Without a
# bound, Prisma raises a DataError on the skip and the caller gets a 500.
MAX_PAGE = 1_000_000


class Page(BaseModel, Generic[T]):
    """The response envelope of every v2 list endpoint."""

    items: list[T]
    next_cursor: Optional[str] = Field(
        description="Pass as `cursor` to fetch the next page. `null` on the last page."
    )
    total_count: Optional[int] = Field(
        description=(
            "Items matching the request across all pages, or `null` where the "
            "underlying source cannot report one. Always present."
        )
    )


class PageRequest(BaseModel):
    """The `limit`/`cursor` query parameters of every v2 list endpoint."""

    limit: int
    cursor: Optional[str] = None
    position: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def _decode_cursor(self) -> "PageRequest":
        # Decoded here rather than where it is read, so a malformed cursor is a
        # 400 even on an endpoint that ignores the value.
        self.position = _decode(self.cursor)
        return self

    @property
    def page(self) -> int:
        """1-indexed page, for the offset-paginated sources."""
        if not self.position:
            return 1
        page = self.position.get("p")
        if self.position.get("k") != "p" or not isinstance(page, int):
            raise _wrong_cursor("a page")
        if not 1 <= page <= MAX_PAGE:
            raise _malformed_cursor()
        return page

    @property
    def token(self) -> Optional[str]:
        """Opaque position, for the keyset-paginated sources."""
        if not self.position:
            return None
        token = self.position.get("t")
        if self.position.get("k") != "t" or not isinstance(token, str):
            raise _wrong_cursor("a keyset token")
        return token

    def paged(self, items: Sequence[T], total_count: int) -> Page[T]:
        """For an offset-paginated source that reports a total."""
        return Page[T](
            items=list(items),
            next_cursor=(
                encode_page_cursor(self.page + 1)
                if self.page * self.limit < total_count
                else None
            ),
            total_count=total_count,
        )

    def slice(self, items: Sequence[T]) -> Page[T]:
        """For a source that returns everything it has; paginate in memory."""
        start = (self.page - 1) * self.limit
        return self.paged(items[start : start + self.limit], len(items))

    def keyset(
        self,
        items: Sequence[T],
        next_token: Optional[str],
        total_count: Optional[int] = None,
    ) -> Page[T]:
        """For a keyset-paginated source, which may not be able to report a total."""
        return Page[T](
            items=list(items),
            next_cursor=encode_token_cursor(next_token) if next_token else None,
            total_count=total_count,
        )

    def uncounted(self, items: Sequence[T]) -> Page[T]:
        """For a source that answers in one shot: no second page, no total."""
        if self.position:
            raise _wrong_cursor("no cursor — this endpoint returns a single page")
        return Page[T](items=list(items), next_cursor=None, total_count=None)


def page_request(
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_PAGE_SIZE,
            description=f"Items per page (max {MAX_PAGE_SIZE})",
        ),
    ] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[
        Optional[str],
        Query(description="`next_cursor` from the previous response"),
    ] = None,
) -> PageRequest:
    return PageRequest(limit=limit, cursor=cursor)


def encode_page_cursor(page: int) -> str:
    return _encode({"k": "p", "p": page})


def encode_token_cursor(token: str) -> str:
    return _encode({"k": "t", "t": token})


def _encode(payload: dict[str, Any]) -> str:
    raw = json.dumps({"v": CURSOR_VERSION, **payload}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode(cursor: Optional[str]) -> dict[str, Any]:
    if not cursor:
        return {}
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        )
    except (binascii.Error, UnicodeDecodeError, ValueError, TypeError):
        raise _malformed_cursor()
    if not isinstance(payload, dict) or payload.get("v") != CURSOR_VERSION:
        raise _malformed_cursor()
    return payload


def _malformed_cursor() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Malformed cursor. Pass back the `next_cursor` of a previous response.",
    )


def _wrong_cursor(expected: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"This endpoint expects {expected}. "
            "Cursors are not interchangeable between endpoints."
        ),
    )
