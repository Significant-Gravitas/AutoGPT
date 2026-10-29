"""Email the account owner a message one of their own runs composed.

The recipient is never a parameter: it is the account's verified address, read
here from the user id, so nothing upstream of this call can redirect it.
"""

import asyncio
import logging
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, cast

from postmarker.exceptions import ClientError
from pydantic import BaseModel, Field, ValidationError, field_validator
from requests import RequestException

from backend.data.redis_client import get_redis_async
from backend.notifications.email import EmailSender
from backend.notifications.owner_email_body import (
    MAX_IMAGES,
    BodyFormat,
    InlineImage,
    render_body,
)
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


class OwnerRenderedEmail(RenderedEmail):
    attachments: list[dict[str, str]] = Field(default_factory=list)


class OwnerEmail(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=50_000)
    link: str | None = None
    body_format: BodyFormat = BodyFormat.MARKDOWN
    images: list[InlineImage] = Field(default_factory=list, max_length=MAX_IMAGES)

    # Before the length check, so a subject of only whitespace is refused.
    @field_validator("subject", mode="before")
    @classmethod
    def _one_line(cls, subject: object) -> object:
        return " ".join(subject.split()) if isinstance(subject, str) else subject

    @field_validator("body")
    @classmethod
    def _not_blank(cls, body: str) -> str:
        if not body.strip():
            raise ValueError("The body is empty.")
        return body

    @field_validator("link")
    @classmethod
    def _web_link(cls, link: str | None) -> str | None:
        if link and not link.startswith(("https://", "http://")):
            raise ValueError("The link must be a web address (https:// or http://).")
        return link or None


def parse_owner_email(
    subject: str,
    body: str,
    link: str | None,
    body_format: BodyFormat,
    images: list[InlineImage] | None = None,
) -> OwnerEmail:
    """A ValidationError's type is not one the RPC client can rebuild, so it
    crosses as the ValueError the block reports."""
    try:
        return OwnerEmail(
            subject=subject,
            body=body,
            link=link,
            body_format=body_format,
            images=images or [],
        )
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
    """Raises ValueError when the send is refused and TimeoutError when its
    outcome is unknown, so the caller can say which."""
    db = get_database_manager_async_client(should_retry=False)
    address = await db.get_user_email_by_id(user_id)
    if not address:
        raise ValueError("This account has no email address to send to.")
    if not await db.get_user_email_verification(user_id):
        raise ValueError(
            "This account's email address isn't verified yet, so nothing can be "
            "sent to it. Verify it from the sign-in email, then try again."
        )

    # Off the event loop: rendering a long body takes a CPU-bound tenth of a second.
    email = await asyncio.to_thread(render_owner_email, message)
    counter = f"owner_email:daily:{user_id}:{_utc_today()}"
    limit = settings.config.owner_email_daily_limit
    if not await _take_daily_send(counter, limit):
        raise ValueError(
            f"Already sent {limit} emails to this account today, the most "
            "allowed in a day. Try again after midnight UTC."
        )
    # Only a send that certainly went nowhere gives its slot back: one that
    # timed out may have arrived, and refunding it lets a looping run past the cap.
    try:
        await email_sender.send_owner_email(address, email, email.attachments)
    except ClientError as e:
        await _give_back(counter)
        raise ValueError(f"The email service refused this email: {e}") from e
    except RuntimeError:
        await _give_back(counter)
        raise
    except RequestException as e:
        raise TimeoutError(
            "The email service didn't confirm the send, so it may still arrive. "
            "Check your inbox before sending it again."
        ) from e
    logger.info(f"Sent an owner email for user {user_id}")


def render_owner_email(message: OwnerEmail) -> OwnerRenderedEmail:
    content_ids = {
        image.source: f"image-{n}" for n, image in enumerate(message.images, 1)
    }
    body = render_body(message.body, message.body_format, content_ids)
    context = {
        "subject": message.subject,
        "content": body.html,
        "content_text": body.text,
        "link": message.link,
        "assets": settings.config.email_asset_base_url.rstrip("/"),
        "urls": build_urls(""),
    }
    return OwnerRenderedEmail(
        subject=message.subject,
        preheader="",
        html=_html_env.get_template("owner.html.j2").render(**context),
        text=_text_env.get_template("owner.txt.j2").render(**context).strip(),
        attachments=[
            {
                "Name": f"{content_ids[image.source]}.{image.content_type[6:]}",
                "Content": image.data,
                "ContentType": image.content_type,
                "ContentID": f"cid:{content_ids[image.source]}",
            }
            for image in message.images
        ],
    )


def _utc_today() -> str:
    return datetime.now(tz=UTC).date().isoformat()


async def _take_daily_send(counter: str, limit: int) -> bool:
    """Fails closed, unlike the notification cap: this one exists to stop a
    looping run, and a run can retry where a lost payment notice cannot."""
    try:
        redis_client = await get_redis_async()
        async with redis_client.pipeline(transaction=True) as pipe:
            pipe.incr(counter)
            pipe.expire(counter, _COUNTER_TTL_SECONDS)
            sent, _ = await cast(Awaitable[list[Any]], pipe.execute())
    except Exception as e:
        raise RuntimeError(
            "Couldn't check today's email count, so nothing was sent. "
            "Try again in a minute."
        ) from e
    if sent <= limit:
        return True
    # A refused send gives its slot back too, or a later failed send's give-back
    # leaves the count above what was really sent.
    await _give_back(counter)
    return False


async def _give_back(counter: str) -> None:
    try:
        redis_client = await get_redis_async()
        await redis_client.decr(counter)
    except Exception:
        logger.warning(
            "Could not return a failed owner email to the daily count",
            exc_info=True,
        )
