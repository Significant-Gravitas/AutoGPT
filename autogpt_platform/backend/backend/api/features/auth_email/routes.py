"""Backend endpoint for Better Auth transactional emails.

Better Auth runs in the Next.js frontend (Vercel), which deliberately holds no
email/Postmark credentials. Rather than a frontend SMTP transport, its
``sendResetPassword`` / ``sendVerificationEmail`` hooks POST here with a
short-lived frontend service token — signed with the same Better Auth JWKS
key the backend already trusts for user tokens — and this endpoint forwards
the message to the notification service, which owns the Postmark credential
(the REST API pod holds none). The subject/body are built server-side from a
fixed set of ``type``s (never free-form input), so even a captured token can
only trigger our own templated auth emails, and the action link is restricted
to our own hosts.
"""

import asyncio
import html
import logging
import re
from typing import Literal
from urllib.parse import urlparse

from autogpt_libs.auth import requires_frontend_service
from fastapi import APIRouter, HTTPException, Security, status
from pydantic import BaseModel, EmailStr

from backend.util.clients import get_notification_manager_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

auth_email_router = APIRouter(prefix="/auth-email")

# Module-level so tests can override the exact dependency instance.
requires_auth_email_service = requires_frontend_service("auth-email:send")

_SUBJECTS: dict[str, str] = {
    "reset_password": "Reset your AutoGPT Platform password",
    "verify_email": "Verify your AutoGPT Platform email",
    "change_email": "Confirm your new AutoGPT Platform email",
}

_ACTIONS: dict[str, str] = {
    "reset_password": "reset your password",
    "verify_email": "verify your email",
    "change_email": "confirm your new email address",
}


class AuthEmailRequest(BaseModel):
    type: Literal["reset_password", "verify_email", "change_email"]
    to: EmailStr
    url: str


def _origin_of(url: str) -> str | None:
    """Return the scheme://host[:port] origin of an http(s) URL, or None.
    Built from hostname/port (not netloc) so embedded credentials in a crafted
    URL can't smuggle a different effective host past the allowlist."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _trusted_frontend_origins() -> tuple[set[str], list[str]]:
    """The set of frontend origins auth-email links may target: the configured
    frontend_base_url plus any trusted_frontend_origins. Returns (exact origins,
    regex patterns)."""
    entries: list[str] = []
    frontend_origin = _origin_of(settings.config.frontend_base_url)
    if frontend_origin:
        entries.append(frontend_origin)
    entries.extend(settings.config.trusted_frontend_origins)

    exact = {e for e in entries if not e.startswith("regex:")}
    patterns = [e[len("regex:") :] for e in entries if e.startswith("regex:")]
    return exact, patterns


def _url_origin_allowed(url: str) -> bool:
    """Only send links pointing at a trusted frontend origin (frontend_base_url
    or an entry in trusted_frontend_origins). Blocks a token-holder from
    sending phishing links to arbitrary domains. Self-hosting works with just
    frontend_base_url; there is no hardcoded provider wildcard."""
    origin = _origin_of(url)
    if origin is None:
        return False
    exact, patterns = _trusted_frontend_origins()
    if origin in exact:
        return True
    return any(re.fullmatch(pattern, origin) for pattern in patterns)


@auth_email_router.post(
    "/send",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Security(requires_auth_email_service)],
    summary="Send a Better Auth transactional email via the backend mailer",
    tags=["auth-email"],
)
async def send_auth_email(request: AuthEmailRequest) -> None:
    if not _url_origin_allowed(request.url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="url must point at a trusted frontend origin.",
        )

    subject = _SUBJECTS[request.type]
    action = _ACTIONS[request.type]
    # Escape the (host-validated) URL before embedding it in HTML — a path or
    # query on an allowed host could still carry markup-breaking characters.
    safe_url = html.escape(request.url, quote=True)
    body = (
        f"<p>Click the link below to {action} for the AutoGPT Platform:</p>"
        f'<p><a href="{safe_url}">{safe_url}</a></p>'
        "<p>If you didn't request this, you can safely ignore this email.</p>"
    )

    # The blocking RPC to the notification service runs off the event loop; a
    # delivery failure there surfaces as a 5xx so a misconfigured mailer fails
    # loudly instead of dropping the auth email.
    await asyncio.to_thread(
        get_notification_manager_client().send_transactional_email,
        request.to,
        subject,
        body,
    )
    # Don't log the recipient address — auth emails go to arbitrary users and
    # the address is PII we don't want in application logs.
    logger.info("Sent %s auth email", request.type)
