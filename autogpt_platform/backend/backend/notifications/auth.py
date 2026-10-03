"""Render the fixed Better Auth messages through the shared service-email kit."""

from typing import Literal

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from backend.notifications.renderer import TEMPLATE_DIR, RenderedEmail, build_urls
from backend.util.settings import Settings

AuthEmailType = Literal["reset_password", "verify_email", "change_email"]
settings = Settings()
_html_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)
_text_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


class AuthEmailContent(BaseModel):
    subject: str
    eyebrow: str
    headline: str
    intro: str
    preheader: str
    action: str
    reason: str


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
    ),
    "verify_email": AuthEmailContent(
        subject="Verify your AutoGPT Platform email",
        eyebrow="ACCOUNT · VERIFY EMAIL",
        headline="Confirm your email address",
        intro="Confirm this address for your AutoGPT account.",
        preheader="Use the button below to verify your email.",
        action="Verify email",
        reason="This link was requested from the AutoGPT sign-up form.",
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
}
