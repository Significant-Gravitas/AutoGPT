"""Render the fixed Better Auth messages through the shared service-email kit."""

from typing import Literal

from pydantic import BaseModel

from backend.notifications.renderer import RenderedEmail, build_urls, otto_asset_url
from backend.notifications.template_env import create_body_environment
from backend.util.settings import Settings

AuthEmailType = Literal[
    "reset_password", "verify_email", "change_email", "set_password"
]
settings = Settings()
_html_env = create_body_environment(autoescape=True)
_text_env = create_body_environment(autoescape=False)


class AuthEmailContent(BaseModel):
    subject: str
    eyebrow: str
    headline: str
    intro: str
    preheader: str
    action: str
    reason: str
    # Must match the links the frontend issues.
    expiry: str | None = None
    # What the recipient may not expect from following the link.
    note: str | None = None
    # (lead, rest) progress steps, for the templates that show them.
    steps: list[tuple[str, str]] = []


def render_auth_email(email_type: AuthEmailType, action_url: str) -> RenderedEmail:
    content = _CONTENT[email_type]
    assets = settings.config.email_asset_base_url.rstrip("/")
    context = {
        "content": content,
        "action_url": action_url,
        "assets": assets,
        "otto_asset_url": otto_asset_url(assets),
        "urls": build_urls(""),
    }
    template = _TEMPLATES.get(email_type, "auth")
    return RenderedEmail(
        subject=content.subject,
        preheader=content.preheader,
        html=_html_env.get_template(f"{template}.html.j2").render(**context),
        text=_text_env.get_template(f"{template}.txt.j2").render(**context).strip(),
    )


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
}
