"""Authenticated gateway to desktop streams; never cache the credential redirect."""

from typing import Annotated

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, HTTPException, Query, Security
from fastapi.responses import RedirectResponse

from backend.util.desktop_preview import resolve_preview_link

router = APIRouter()


@router.get("/desktop-preview", response_class=RedirectResponse)
async def open_desktop_preview(
    token: Annotated[str, Query(max_length=8192)],
    user_id: Annotated[str, Security(get_user_id)],
) -> RedirectResponse:
    """Redirect the desktop's owner to its live stream.

    The stream URL carries the desktop's password, so it is only ever
    disclosed here, to the user the token was issued to, and marked
    ``no-store`` so no cache or referrer keeps a copy.
    """
    url = resolve_preview_link(user_id, token)
    if not url:
        raise HTTPException(status_code=404, detail="Preview unavailable")
    return RedirectResponse(
        url,
        status_code=307,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )
