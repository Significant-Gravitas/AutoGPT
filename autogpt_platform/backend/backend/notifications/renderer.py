"""Renders a notification payload into a complete, sendable email.

These templates are complete documents, not content fragments, so they replace
the old base template *and* the per-type content templates. They are rendered
directly rather than through the old `bleach.clean()` content path: that
sanitizer's allow-list has no `<table>`, `<tr>` or `<td>` and would strip the
entire layout. That is safe here — the sanitizer existed to neutralise HTML
inside content templates, and every user-supplied value (agent names, reviewer
comments, refund reasons) is neutralised by Jinja's autoescape, which is the
correct defence.

The subject template renders first and produces two lines (subject, then
preheader); both are then passed into the body template.
"""

import pathlib

from jinja2 import Environment, FileSystemLoader
from prisma.enums import NotificationType
from pydantic import BaseModel

from backend.data.notifications import (
    BaseNotificationData,
    get_lifecycle_kind,
    get_template_family,
)
from backend.util.settings import Settings

settings = Settings()

TEMPLATE_DIR = pathlib.Path(__file__).parent / "templates"

# Autoescape is the defence for user-supplied values, so it is not optional.
_html_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)
# NB: no trim_blocks here — the subject template's line break between subject
# and preheader must survive block tags at end-of-line.
_subject_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
# The plain-text part is built from the same context, so a text-only client
# gets the same facts rather than a tag-stripped approximation of the HTML.
_text_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


class RenderedEmail(BaseModel):
    subject: str
    preheader: str
    html: str
    text: str


class EmailUrls(BaseModel):
    """Per-recipient destinations. Kept out of the queued payload so a message
    that sat in the queue over a deploy still links at today's platform."""

    dashboard: str
    settings: str
    unsubscribe: str
    attention: str
    billing: str
    prefs: str
    marketplace: str
    docs: str
    discord: str


def build_urls(unsubscribe_link: str) -> EmailUrls:
    base = settings.config.frontend_base_url or settings.config.platform_base_url
    return EmailUrls(
        dashboard=f"{base}/library",
        # The Briefing footer appends ?f=daily|weekly|monthly|alerts|off to
        # this, and the settings page applies it on load — that is what makes
        # the volume knob one click rather than a form.
        settings=f"{base}/settings/account",
        unsubscribe=unsubscribe_link,
        attention=f"{base}/library?filter=needs-attention",
        billing=f"{base}/settings/billing",
        prefs=f"{base}/settings/account",
        marketplace=f"{base}/marketplace",
        docs=settings.config.docs_base_url,
        discord=settings.config.discord_invite_url,
    )


def render(
    notification_type: NotificationType,
    data: BaseNotificationData,
    user_email: str,
    urls: EmailUrls,
) -> RenderedEmail:
    """Render one notification into subject, preheader, HTML and plain text."""
    family = get_template_family(notification_type)
    context = _build_context(notification_type, data, user_email, urls)

    subject, preheader = _render_subject(family, context)
    html = _html_env.get_template(f"{family}.html.j2").render(
        **context, subject=subject, preheader=preheader
    )
    text = _text_env.get_template(f"{family}.txt.j2").render(
        **context, subject=subject, preheader=preheader
    )
    return RenderedEmail(subject=subject, preheader=preheader, html=html, text=text)


def _build_context(
    notification_type: NotificationType,
    data: BaseNotificationData,
    user_email: str,
    urls: EmailUrls,
) -> dict:
    # Optional slots are dropped rather than passed as None, because the
    # templates reach for them with Jinja's `default` filter — which only fires
    # on undefined, not on a None that happens to be present. Only the top
    # level is pruned: `totals.usd_estimate` is deliberately tested with
    # `is not none` and has to survive as an explicit None.
    context = {k: v for k, v in data.model_dump().items() if v is not None}
    context["user_email"] = user_email
    context["urls"] = urls.model_dump()
    # Hero art is hosted, not inline: Outlook does not render inline SVG, and
    # Gmail does not display data-URI images.
    context["assets"] = settings.config.email_asset_base_url.rstrip("/")
    if kind := get_lifecycle_kind(notification_type):
        context["kind"] = kind
    return context


def _render_subject(family: str, context: dict) -> tuple[str, str]:
    """Subject templates render two lines: subject, then preheader.

    The preheader is the lede's second sentence, never "View this email in your
    browser". A template that emits only one line still returns cleanly.
    """
    rendered = _subject_env.get_template(f"{family}.subject.j2").render(**context)
    lines = [line.strip() for line in rendered.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{family}.subject.j2 rendered no subject line")
    return lines[0], (lines[1] if len(lines) > 1 else "")
