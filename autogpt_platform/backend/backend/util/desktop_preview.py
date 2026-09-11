"""Owner-bound, encrypted links for E2B desktop streams.

A desktop's live URL carries its VNC password, so anyone holding the URL
controls the desktop and every login in it.  The URL therefore never leaves
the backend: callers get a link to ``/api/desktop-preview`` carrying an
encrypted, owner-bound, expiring token, and the endpoint redirects the owner
to the real URL with ``Cache-Control: no-store``.  A chat message, a share
link or a copied card holds the token, never the credential.
"""

from typing import Literal
from urllib.parse import urlencode

from cryptography.fernet import InvalidToken
from pydantic import BaseModel, ValidationError

from backend.util.encryption import JSONCryptor
from backend.util.settings import Config

# The token is bound to the owner and only redeemable with the owner's
# session, so its lifetime is about how long a stored chat card keeps
# working, not about limiting exposure of the credential behind it.
PREVIEW_LINK_TTL = 86400


class DesktopPreview(BaseModel):
    purpose: Literal["e2b-desktop-preview"]
    user_id: str
    url: str


def create_preview_link(user_id: str, live_url: str) -> str:
    """An owner-bound link to *live_url* that discloses nothing about it."""
    if not user_id:
        raise ValueError("Live view requires an authenticated user")
    base_url = Config().frontend_base_url.rstrip("/")
    if not base_url:
        raise ValueError("Live view requires FRONTEND_BASE_URL to be configured")
    token = JSONCryptor().encrypt(
        DesktopPreview(
            purpose="e2b-desktop-preview", user_id=user_id, url=live_url
        ).model_dump()
    )
    return f"{base_url}/api/proxy/api/desktop-preview?{urlencode({'token': token})}"


def resolve_preview_link(user_id: str, token: str) -> str | None:
    """The live URL behind *token*, only for the user it was issued to."""
    try:
        data = JSONCryptor().fernet.decrypt(token.encode(), ttl=PREVIEW_LINK_TTL)
        preview = DesktopPreview.model_validate_json(data)
    except (InvalidToken, ValidationError):
        return None
    if not user_id or preview.user_id != user_id:
        return None
    return preview.url
