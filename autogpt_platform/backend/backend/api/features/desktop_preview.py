"""Authenticated gateway to desktop previews; never cache the credential redirect."""

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
    url = resolve_preview_link(user_id, token)
    if not url:
        raise HTTPException(status_code=404, detail="Preview unavailable")
    return RedirectResponse(
        url,
        status_code=307,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )
