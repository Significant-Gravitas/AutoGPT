"""Render the fixed account emails through the shared service-email kit: the
Better Auth link emails and the account-deleted confirmation."""

import asyncio
from typing import Literal, assert_never

from prisma.enums import AccountDeletionSource
from pydantic import BaseModel

from backend.notifications.renderer import RenderedEmail, build_urls, otto_asset_url
from backend.notifications.template_env import create_body_environment
from backend.util.clients import get_notification_manager_client
from backend.util.settings import Settings

# The emails a frontend service token may trigger: each one carries a link.
AuthLinkEmailType = Literal[
    "reset_password", "verify_email", "change_email", "set_password"
]
AuthEmailType = Literal[AuthLinkEmailType, "account_deleted"]
settings = Settings()
_html_env = create_body_environment(autoescape=True)
_text_env = create_body_environment(autoescape=False)


class AuthEmailContent(BaseModel):
    subject: str
    eyebrow: str
    headline: str
    intro: str
    preheader: str
    # The button label. Without one the email has no button, no link and no
    # "If you didn't request this" line.
    action: str | None = None
    reason: str
    # Must match the links the frontend issues.
    expiry: str | None = None
    # What the recipient may not expect from following the link. Without a
    # link it ends the body paragraph instead.
    note: str | None = None
    # Close with the shared inbox (which also takes replies) instead of
    # Discord, for an email about the person's own account.
    close_with_inbox: bool = False
    # (lead, rest) progress steps, for the templates that show them.
    steps: list[tuple[str, str]] = []


def render_auth_email(
    email_type: AuthEmailType,
    action_url: str | None = None,
    *,
    reason: str | None = None,
) -> RenderedEmail:
    content = _CONTENT[email_type]
    if content.action and not action_url:
        raise ValueError(f"The {email_type} email needs an action URL")
    if reason:
        content = content.model_copy(update={"reason": reason})
    assets = settings.config.email_asset_base_url.rstrip("/")
    context = {
        "content": content,
        "action_url": action_url,
        "assets": assets,
        "otto_asset_url": otto_asset_url(assets),
        "urls": build_urls(""),
        "inbox": settings.config.billing_reply_to_email,
    }
    template = _TEMPLATES.get(email_type, "auth")
    return RenderedEmail(
        subject=content.subject,
        preheader=content.preheader,
        html=_html_env.get_template(f"{template}.html.j2").render(**context),
        text=_text_env.get_template(f"{template}.txt.j2").render(**context).strip(),
    )


async def send_account_deleted_email(to: str, source: AccountDeletionSource) -> None:
    """Confirm a closed account to its owner, with replies going to the shared
    inbox. Raises when delivery fails; the caller decides what that means."""
    email = render_auth_email("account_deleted", reason=_deletion_reason(source))
    await asyncio.to_thread(
        get_notification_manager_client().send_email_or_raise,
        to,
        email.subject,
        email.html,
        email.text,
        reply_to=settings.config.billing_reply_to_email,
    )


def _deletion_reason(source: AccountDeletionSource) -> str:
    """The email's reason line, by where the deletion was asked for. A new
    source fails the type check here until it has its own line."""
    match source:
        case AccountDeletionSource.SETTINGS | AccountDeletionSource.PAYWALL:
            return "You deleted your account in AutoGPT."
        case AccountDeletionSource.EMAIL:
            return "You asked us by email to delete your account."
        case _:
            assert_never(source)


# Often the first email a new user gets, so it has its own fuller layout. The
# rest use the compact auth layout.
_TEMPLATES: dict[AuthEmailType, str] = {"verify_email": "auth_verify"}


_CONTENT: dict[AuthEmailType, AuthEmailContent] = {
    "reset_password": AuthEmailContent(
        subject="Reset your AutoGPT Platform password",
        eyebrow="ACCOUNT · RESET PASSWORD",
        headline="Reset your password",
        intro="A password reset was requested for your AutoGPT account.",
        preheader="Use the button below to choose a new password.",
        action="Reset password",
        reason="This link was requested from the AutoGPT password-reset form.",
        expiry="1 hour",
    ),
    "verify_email": AuthEmailContent(
        subject="Verify your AutoGPT Platform email",
        eyebrow="ONE LAST STEP",
        headline="Confirm your email address",
        intro=(
            "You're one click away from AutoGPT. Confirm that this is your "
            "email address and your account is ready to use."
        ),
        preheader="One click to confirm your address and finish setting up AutoGPT.",
        action="Confirm email address",
        reason="This link was requested from the AutoGPT sign-up form.",
        expiry="24 hours",
        steps=[
            ("Create your account.", "Done."),
            ("Confirm your email.", "You're here."),
            (
                "Start working.",
                "Describe a task in plain language and Otto takes it from there.",
            ),
        ],
    ),
    "change_email": AuthEmailContent(
        subject="Confirm your new AutoGPT Platform email",
        eyebrow="ACCOUNT · CONFIRM EMAIL",
        headline="Confirm your new email address",
        intro="An email address change was requested for your AutoGPT account.",
        preheader="Use the button below to confirm your new email address.",
        action="Confirm email",
        reason="This link was requested from your AutoGPT account settings.",
    ),
    # Sent for an unverified account when a sign-up or resend can't safely send
    # a verification link, so the email has to say why it came.
    "set_password": AuthEmailContent(
        subject="Set your AutoGPT Platform password",
        eyebrow="ACCOUNT · SET PASSWORD",
        headline="Set your password",
        intro=(
            "Someone asked to sign up for, or verify, an AutoGPT Platform "
            "account with this email address."
        ),
        preheader="Use the button below to set a password and finish signing up.",
        action="Set password",
        reason="This link was requested from the AutoGPT sign-up form.",
        expiry="1 hour",
        note=(
            "If you already finished signing up with an earlier link, the "
            "password you chose then no longer works: set a new one with this "
            'link, or use "Forgot password" on the login page.'
        ),
    ),
    "account_deleted": AuthEmailContent(
        subject="Your AutoGPT account is deleted",
        eyebrow="ACCOUNT · DELETED",
        headline="Your account is deleted",
        intro="We've closed your AutoGPT account and disconnected your apps.",
        preheader="We finish removing your data within 30 days.",
        reason="You deleted your account in AutoGPT.",
        note="If you have a minute, reply and tell us what we could have done better.",
        close_with_inbox=True,
    ),
}
