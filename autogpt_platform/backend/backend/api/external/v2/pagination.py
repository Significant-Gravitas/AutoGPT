"""
V2 External API - Pagination

One envelope and one set of query parameters for every list endpoint:
`?limit=&cursor=` in, `{"items": [...], "next_cursor": ...}` out.

The cursor is opaque on purpose. Most sources underneath are page-based and one
is keyset-based; encoding which is which inside the token means a source can
switch without changing the contract clients wrote against.
"""

import base64
import binascii
import json
from typing import Annotated, Generic, Optional, Sequence, TypeVar

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field
from starlette import status

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """The response envelope of every v2 list endpoint."""

    items: list[T]
    next_cursor: Optional[str] = Field(
        default=None,
        description="Pass as `cursor` to fetch the next page. `null` on the last page.",
    )


class PageRequest(BaseModel):
    """The `limit`/`cursor` query parameters of every v2 list endpoint."""

    limit: int
    cursor: Optional[str] = None

    @property
    def page(self) -> int:
        """1-indexed page, for the offset-paginated sources."""
        page = _decode(self.cursor).get("p", 1)
        if not isinstance(page, int) or page < 1:
            raise _malformed_cursor()
        return page

    @property
    def token(self) -> Optional[str]:
        """Opaque position, for the keyset-paginated sources."""
        token = _decode(self.cursor).get("t")
        if token is not None and not isinstance(token, str):
            raise _malformed_cursor()
        return token

    def paged(self, items: Sequence[T], total_count: int) -> Page[T]:
        return Page[T](
            items=list(items),
            next_cursor=(
                encode_page_cursor(self.page + 1)
                if self.page * self.limit < total_count
                else None
            ),
        )

    def keyset(self, items: Sequence[T], next_token: Optional[str]) -> Page[T]:
        return Page[T](
            items=list(items),
            next_cursor=encode_token_cursor(next_token) if next_token else None,
        )

    def unpaginated(self, items: Sequence[T]) -> Page[T]:
        """For sources that return everything they have in one call."""
        return Page[T](items=list(items), next_cursor=None)

    def slice(self, items: Sequence[T]) -> Page[T]:
        """For sources that return everything they have; paginate in memory."""
        start = (self.page - 1) * self.limit
        return self.paged(items[start : start + self.limit], len(items))


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
    return _encode({"p": page})


def encode_token_cursor(token: str) -> str:
    return _encode({"t": token})


def _encode(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode(cursor: Optional[str]) -> dict:
    if not cursor:
        return {}
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        )
    except (binascii.Error, UnicodeDecodeError, ValueError, TypeError):
        raise _malformed_cursor()
    if not isinstance(payload, dict):
        raise _malformed_cursor()
    return payload


def _malformed_cursor() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Malformed cursor. Pass back the `next_cursor` of a previous response.",
    )
