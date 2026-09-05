"""Owner-bound, encrypted links for E2B desktop credentials."""

from typing import Literal
from urllib.parse import urlencode

from cryptography.fernet import InvalidToken
from pydantic import BaseModel, ValidationError

from backend.util.encryption import JSONCryptor
from backend.util.settings import Config

PREVIEW_LINK_TTL = 86400


class DesktopPreview(BaseModel):
    purpose: Literal["e2b-desktop-preview"]
    user_id: str
    url: str


def create_preview_link(user_id: str, live_url: str) -> str:
    if not user_id:
        raise ValueError("Live view requires an authenticated user")
    token = JSONCryptor().encrypt(
        DesktopPreview(
            purpose="e2b-desktop-preview", user_id=user_id, url=live_url
        ).model_dump()
    )
    base_url = Config().frontend_base_url.rstrip("/")
    return f"{base_url}/api/proxy/api/desktop-preview?{urlencode({'token': token})}"


def resolve_preview_link(user_id: str, token: str) -> str | None:
    try:
        data = JSONCryptor().fernet.decrypt(token.encode(), ttl=PREVIEW_LINK_TTL)
        preview = DesktopPreview.model_validate_json(data)
    except (InvalidToken, ValidationError):
        return None
    if preview.user_id != user_id:
        return None
    return preview.url
