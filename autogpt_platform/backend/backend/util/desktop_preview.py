"""Owner-bound, encrypted links for E2B desktop streams.

A desktop's live URL carries its VNC password, so anyone holding the URL
controls the desktop and every login in it.  The URL therefore never leaves
the backend: callers get a link to ``/api/desktop-preview`` carrying an
encrypted, owner-bound, expiring token, and the endpoint redirects the owner
to the real URL with ``Cache-Control: no-store``.  A chat message, a share
link or a copied card holds the token, never the credential.

A link outlives the stream it points at: every pause of the box stops the
stream, and the next open restarts it under a new password.  The token names
the owner's Redis key holding the live password, so the endpoint can tell a
stopped stream apart and say so rather than send the browser to a dead page.
"""

import logging
from typing import Literal
from urllib.parse import parse_qs, urlencode, urlsplit

from cryptography.fernet import InvalidToken
from pydantic import BaseModel, ValidationError

from backend.data.redis_client import get_redis_async
from backend.util.encryption import JSONCryptor

logger = logging.getLogger(__name__)

# The token is bound to the owner and only redeemable with the owner's
# session, so its lifetime is about how long a stored chat card keeps
# working, not about limiting exposure of the credential behind it.
PREVIEW_LINK_TTL = 86400


class DesktopPreview(BaseModel):
    purpose: Literal["e2b-desktop-preview"]
    user_id: str
    url: str
    # Absent on links issued before the key was carried; those are not checked.
    stream_key: str | None = None


def create_preview_link(
    user_id: str, live_url: str, stream_key: str | None = None
) -> str:
    """An owner-bound link to *live_url* that discloses nothing about it.

    Origin-relative: the owner redeems it on whatever origin they are on, so
    a preview deployment, a custom domain or a plain-http local stack still
    sends the session cookie, and the link never leaks the platform's host.
    *stream_key* is the Redis key that holds the stream's live password.
    """
    if not user_id:
        raise ValueError("Live view requires an authenticated user")
    token = JSONCryptor().encrypt(
        DesktopPreview(
            purpose="e2b-desktop-preview",
            user_id=user_id,
            url=live_url,
            stream_key=stream_key,
        ).model_dump()
    )
    return f"/api/proxy/api/desktop-preview?{urlencode({'token': token})}"


def resolve_preview(user_id: str, token: str) -> DesktopPreview | None:
    """The preview behind *token*, only for the user it was issued to."""
    try:
        data = JSONCryptor().fernet.decrypt(token.encode(), ttl=PREVIEW_LINK_TTL)
        preview = DesktopPreview.model_validate_json(data)
    except (InvalidToken, ValidationError):
        return None
    if not user_id or preview.user_id != user_id:
        return None
    return preview


async def stream_is_served(preview: DesktopPreview) -> bool:
    """Whether the box still serves the stream under the link's password.

    The key holds the live password while the stream runs, an empty marker
    once a pause stopped it, a new password after a restart, and nothing once
    the box's running limit has passed (``backend.copilot.computer``).
    """
    if preview.stream_key is None:
        return True
    try:
        redis = await get_redis_async()
        raw = await redis.get(preview.stream_key)
    except Exception as exc:
        # The stream's own password still guards it; only the notice is lost.
        logger.warning(f"Could not read the desktop stream state: {exc}")
        return True
    current = raw.decode() if isinstance(raw, bytes) else raw
    password = parse_qs(urlsplit(preview.url).query).get("password", [""])[0]
    return bool(current) and current == password
