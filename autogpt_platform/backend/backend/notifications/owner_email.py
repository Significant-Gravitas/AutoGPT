"""Email the account owner a message one of their own runs composed.

The recipient is never a parameter: it is the account's verified address, read
here from the user id, so nothing upstream of this call can redirect it.
"""

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.data.redis_client import get_redis_async
from backend.notifications.email import EmailSender
from backend.notifications.renderer import RenderedEmail, build_urls
from backend.notifications.template_env import create_body_environment
from backend.util.clients import get_database_manager_async_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

_html_env = create_body_environment(autoescape=True)
_text_env = create_body_environment(autoescape=False)
# Counters outlive the UTC day they cover, so a send at 23:59 is still counted.
_COUNTER_TTL_SECONDS = 60 * 60 * 48


class OwnerEmail(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=50_000)
    link: str | None = None

    @field_validator("subject")
    @classmethod
    def _one_line(cls, subject: str) -> str:
        return " ".join(subject.split())

    @field_validator("link")
    @classmethod
    def _web_link(cls, link: str | None) -> str | None:
        if link and not link.startswith(("https://", "http://")):
            raise ValueError("The link must be a web address (https://...).")
        return link or None


def parse_owner_email(subject: str, body: str, link: str | None) -> OwnerEmail:
    """A ValidationError's type is not one the RPC client can rebuild, so it
    crosses as the ValueError the block reports."""
    try:
        return OwnerEmail(subject=subject, body=body, link=link)
    except ValidationError as e:
        raise ValueError(
            "; ".join(
                f"{'.'.join(map(str, err['loc']))}: "
                + err["msg"].removeprefix("Value error, ")
                for err in e.errors()
            )
        ) from e


async def send_owner_email(
    email_sender: EmailSender, user_id: str, message: OwnerEmail
) -> None:
    """Raises ValueError when the send is refused, so the caller can say why."""
    db = get_database_manager_async_client(should_retry=False)
    address = await db.get_user_email_by_id(user_id)
    if not address:
        raise ValueError("This account has no email address to send to.")
    if not await db.get_user_email_verification(user_id):
        raise ValueError(
            "This account's email address isn't verified yet, so nothing can be "
            "sent to it. Verify it from the sign-in email, then try again."
        )

    counter = f"owner_email:daily:{user_id}:{_utc_today()}"
    limit = settings.config.owner_email_daily_limit
    if not await _take_daily_send(counter, limit):
        raise ValueError(
            f"Already sent {limit} emails to this account today, the most "
            "allowed in a day. Try again tomorrow."
        )
    try:
        await email_sender.send_owner_email(address, render_owner_email(message))
    except Exception:
        await _give_back(counter)
        raise
    logger.info(f"Sent an owner email for user {user_id}")


def render_owner_email(message: OwnerEmail) -> RenderedEmail:
    context = {
        "subject": message.subject,
        "paragraphs": [p.strip() for p in message.body.split("\n\n") if p.strip()],
        "link": message.link,
        "assets": settings.config.email_asset_base_url.rstrip("/"),
        "urls": build_urls(""),
    }
    return RenderedEmail(
        subject=message.subject,
        preheader="",
        html=_html_env.get_template("owner.html.j2").render(**context),
        text=_text_env.get_template("owner.txt.j2").render(**context).strip(),
    )


def _utc_today() -> str:
    return datetime.now(tz=UTC).date().isoformat()


async def _take_daily_send(counter: str, limit: int) -> bool:
    """Fails closed, unlike the notification cap: this one exists to stop a
    looping run, and a run can retry where a lost payment notice cannot."""
    try:
        redis_client = await get_redis_async()
        sent = await redis_client.incr(counter)
        if sent == 1:
            await redis_client.expire(counter, _COUNTER_TTL_SECONDS)
    except Exception as e:
        raise RuntimeError(
            "Couldn't check today's email count, so nothing was sent. "
            "Try again in a minute."
        ) from e
    return sent <= limit


async def _give_back(counter: str) -> None:
    try:
        redis_client = await get_redis_async()
        await redis_client.decr(counter)
    except Exception:
        logger.warning(
            "Could not return a failed owner email to the daily count",
            exc_info=True,
        )
