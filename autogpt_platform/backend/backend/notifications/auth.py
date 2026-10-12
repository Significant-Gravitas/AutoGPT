"""Render the fixed Better Auth messages through the shared service-email kit."""

from typing import Literal

from pydantic import BaseModel

from backend.notifications.renderer import RenderedEmail, build_urls
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


def render_auth_email(email_type: AuthEmailType, action_url: str) -> RenderedEmail:
    content = _CONTENT[email_type]
    context = {
        "content": content,
        "action_url": action_url,
        "assets": settings.config.email_asset_base_url.rstrip("/"),
        "urls": build_urls(""),
    }
    return RenderedEmail(
        subject=content.subject,
        preheader=content.preheader,
        html=_html_env.get_template("auth.html.j2").render(**context),
        text=_text_env.get_template("auth.txt.j2").render(**context).strip(),
    )


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
        eyebrow="ACCOUNT · VERIFY EMAIL",
        headline="Confirm your email address",
        intro="Confirm this address for your AutoGPT account.",
        preheader="Use the button below to verify your email.",
        action="Verify email",
        reason="This link was requested from the AutoGPT sign-up form.",
        expiry="24 hours",
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
