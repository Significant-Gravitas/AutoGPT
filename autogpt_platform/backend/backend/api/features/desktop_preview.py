"""Authenticated gateway to desktop streams; never cache the credential redirect."""

from typing import Annotated

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, HTTPException, Query, Security
from fastapi.responses import RedirectResponse

from backend.util.desktop_preview import resolve_preview, stream_is_served

router = APIRouter()


@router.get(
    "/desktop-preview",
    response_class=RedirectResponse,
    responses={
        404: {"description": "Preview unavailable"},
        410: {"description": "Desktop stream ended"},
    },
)
async def open_desktop_preview(
    token: Annotated[str, Query(max_length=8192)],
    user_id: Annotated[str, Security(get_user_id)],
) -> RedirectResponse:
    """Redirect the desktop's owner to its live stream.

    The stream URL carries the desktop's password, so it is only ever
    disclosed here, to the user the token was issued to, and marked
    ``no-store`` so no cache or referrer keeps a copy.  A stream the box no
    longer serves answers 410, so the page can say so instead of showing a
    dead frame.
    """
    preview = resolve_preview(user_id, token)
    if not preview:
        raise HTTPException(status_code=404, detail="Preview unavailable")
    if not await stream_is_served(preview):
        raise HTTPException(status_code=410, detail="Desktop stream ended")
    return RedirectResponse(
        preview.url,
        status_code=307,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )
