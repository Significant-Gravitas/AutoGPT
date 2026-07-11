"""Backend endpoint for Better Auth transactional emails.

Better Auth runs in the Next.js frontend (Vercel), which deliberately holds no
email/Postmark credentials. Rather than a frontend SMTP transport, its
``sendResetPassword`` / ``sendVerificationEmail`` hooks POST here with a shared
``AUTH_EMAIL_TOKEN``, and the backend sends the message through its existing
Postmark mailer. The subject/body are built server-side from a fixed set of
``type``s (never free-form input), so a leaked token can only trigger our own
templated auth emails, and the action link is restricted to our own hosts.
"""

import asyncio
import hmac
import logging
from typing import Annotated, Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, EmailStr

from backend.notifications.email import EmailSender
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

auth_email_router = APIRouter(prefix="/auth-email")

_email_sender = EmailSender()

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


def require_auth_email_token(
    x_auth_email_token: Annotated[str | None, Header()] = None,
) -> None:
    expected = settings.secrets.auth_email_token
    if not expected:
        # Disabled unless the shared secret is configured on this backend.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth email endpoint is not configured.",
        )
    if not x_auth_email_token or not hmac.compare_digest(x_auth_email_token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def _url_host_allowed(url: str) -> bool:
    """Only send links pointing at our own frontend: the configured
    frontend_base_url host in prod, or a *.vercel.app preview host. Blocks a
    token-holder from sending phishing links to arbitrary domains."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname
    frontend = settings.config.frontend_base_url
    if frontend:
        frontend_host = urlparse(frontend).hostname
        if frontend_host and host == frontend_host:
            return True
    return host.endswith(".vercel.app")


@auth_email_router.post(
    "/send",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_auth_email_token)],
    summary="Send a Better Auth transactional email via the backend mailer",
    tags=["auth-email"],
)
async def send_auth_email(request: AuthEmailRequest) -> None:
    if not _url_host_allowed(request.url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="url must be an https link on an allowed frontend host.",
        )

    subject = _SUBJECTS[request.type]
    action = _ACTIONS[request.type]
    body = (
        f"<p>Click the link below to {action} for the AutoGPT Platform:</p>"
        f'<p><a href="{request.url}">{request.url}</a></p>'
        "<p>If you didn't request this, you can safely ignore this email.</p>"
    )

    # _email_sender.send_transactional wraps a blocking Postmark HTTP call; run
    # it off the event loop and let its RuntimeError surface as a 5xx so a
    # misconfigured mailer fails loudly instead of dropping the auth email.
    await asyncio.to_thread(_email_sender.send_transactional, request.to, subject, body)
    logger.info("Sent %s auth email to %s", request.type, request.to)
