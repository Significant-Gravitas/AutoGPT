"""Render the fixed account emails through the shared service-email kit: the
Better Auth link emails and the account-closed confirmation."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Literal, assert_never

from prisma.enums import AccountDeletionSource
from pydantic import BaseModel

from backend.notifications.renderer import RenderedEmail, build_urls, otto_asset_url
from backend.notifications.template_env import create_body_environment
from backend.util.clients import get_notification_manager_client
from backend.util.settings import Settings

# The emails a frontend service token may trigger: each one carries a link.
AuthLinkEmailType = Literal[
    "reset_password", "verify_email", "verify_new_email", "change_email", "set_password"
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
    action_url: str | None,
    recipient: str,
    new_email: str | None = None,
    *,
    reason: str | None = None,
    preheader: str | None = None,
) -> RenderedEmail:
    content = _CONTENT[email_type]
    if content.action and not action_url:
        raise ValueError(f"The {email_type} email needs an action URL")
    if reason:
        content = content.model_copy(update={"reason": reason})
    if preheader:
        content = content.model_copy(update={"preheader": preheader})
    assets = settings.config.email_asset_base_url.rstrip("/")
    context = {
        "content": content,
        "action_url": action_url,
        "recipient": recipient,
        # The address an email change moves the account to, when known.
        "new_email": new_email,
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


async def send_account_deleted_email(
    to: str,
    source: AccountDeletionSource,
    erase_after: datetime,
    *,
    erase_now: bool = False,
    now: datetime | None = None,
) -> None:
    """Confirm a closed account to its owner, with the date until which they
    can still keep it and replies going to the shared inbox. Sent once that
    date is past (the erasure sends the confirmation closing could not), it
    says the account is being erased instead. With `erase_now` (the owner
    asked to be erased at once, so there were never 14 days) it says it is
    being erased as they asked. Raises when delivery fails; the caller decides
    what that means."""
    if erase_now:
        preheader = ACCOUNT_ERASED_AS_ASKED_PREHEADER
    elif erase_after <= (now or datetime.now(UTC)):
        preheader = ACCOUNT_BEING_ERASED_PREHEADER
    else:
        preheader = account_closed_preheader(erase_after)
    email = render_auth_email(
        "account_deleted",
        None,
        to,
        reason=_deletion_reason(source),
        preheader=preheader,
    )
    await asyncio.to_thread(
        get_notification_manager_client().send_email_or_raise,
        to,
        email.subject,
        email.html,
        email.text,
        reply_to=settings.config.billing_reply_to_email,
    )


ACCOUNT_ERASED_AS_ASKED_PREHEADER = (
    "As you asked, everything in your account is being erased now."
)

ACCOUNT_BEING_ERASED_PREHEADER = (
    "The 14 days to keep it are over, so everything in your account is being "
    "erased now."
)


def account_closed_preheader(erase_after: datetime) -> str:
    """How long the owner can still change their mind, as a date with no time
    zone. Keeping the account is refused from the exact erasure moment, and
    the named day must be over by then wherever the reader is: its end is
    at most 36 hours after its UTC start in the westernmost zone (UTC-12), so
    the date is the UTC day of ``erase_after`` minus 36 hours."""
    last_day = (erase_after - timedelta(hours=36)).date()
    day = f"{last_day.day} {last_day.strftime('%b %Y')}"
    return (
        f"Changed your mind? Sign in by {day} and choose Keep my account. "
        "After that, everything in your account is erased."
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


# Often the first email a new user gets, and the two emails of an address
# change, so they have their own fuller layouts. The rest use the compact auth
# layout.
_TEMPLATES: dict[AuthEmailType, str] = {
    "verify_email": "auth_verify",
    "verify_new_email": "auth_verify_new",
    "change_email": "auth_change_email",
}


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
    # Sent to the new address when someone changes their account's email, so
    # it must not read like a sign-up.
    "verify_new_email": AuthEmailContent(
        subject="Verify your new AutoGPT Platform email",
        eyebrow="ACCOUNT · NEW EMAIL ADDRESS",
        headline="Verify your new email address",
        # For an unverified account the change has already been made when this
        # is sent, so the copy must not say it is still pending.
        intro=(
            "You asked to change the email address on your AutoGPT account "
            "to this one. Verify it to confirm the address is yours."
        ),
        preheader="One click to confirm this is your new AutoGPT email address.",
        action="Verify new email",
        reason="This link was requested from your AutoGPT account settings.",
        expiry="24 hours",
    ),
    # Sent to the account's current address, which has to approve a change
    # before the new address is sent its own link.
    "change_email": AuthEmailContent(
        subject="Approve the email change on your AutoGPT account",
        eyebrow="ACCOUNT · EMAIL CHANGE",
        headline="Approve your email change",
        intro=(
            "A request was made to move your AutoGPT account from this email "
            "address to a new one. If it was you, approve the change."
        ),
        preheader="Approve moving your AutoGPT account to a new email address.",
        action="Approve change",
        reason="This link was requested from your AutoGPT account settings.",
        expiry="24 hours",
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
    # The preheader names the date (account_closed_preheader); this one is
    # only for a render without it.
    "account_deleted": AuthEmailContent(
        subject="Your AutoGPT account is closed",
        eyebrow="ACCOUNT · CLOSED",
        headline="Your account is closed",
        intro=(
            "We've closed your AutoGPT account, signed you out, disconnected "
            "your apps and revoked your API keys. Your Marketplace listings and "
            "reviews are removed, and notification emails stay off until you "
            "switch them back on."
        ),
        preheader=(
            "Changed your mind? Sign in within 14 days and choose Keep my "
            "account. After that, everything in your account is erased."
        ),
        reason="You deleted your account in AutoGPT.",
        note="If you have a minute, reply and tell us what we could have done better.",
        close_with_inbox=True,
    ),
}
