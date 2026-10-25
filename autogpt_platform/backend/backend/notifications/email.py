"""Postmark delivery for the notification families.

Everything goes out on a transactional stream, separate from marketing mail
(which the platform does not send at all — the onboarding tour and the monthly
changelog are MailerLite's). Every message carries a plain-text MIME part built
from the same data model. Product notifications carry one-click
List-Unsubscribe headers; billing, auth and internal Ops mail do not.

Each message is tagged with what it is (the notification type, ``auth`` or
``owner_email``).
Postmark echoes the tag in its delivery, open, click and bounce webhooks,
which ``api/features/postmark`` turns into PostHog events. Every message also
says whether Postmark tracks its opens and clicks, so the stream's own default
never decides: notification mail follows ``postmark_track_opens_and_clicks``,
and auth mail is never tracked, since its links carry sign-in tokens, nor is
mail a user's run sends them, whose links are their own.
"""

import asyncio
import logging
from typing import Any

from postmarker.core import PostmarkClient
from postmarker.exceptions import ClientError
from postmarker.models.emails import EmailManager
from prisma.enums import NotificationType
from requests import RequestException

from backend.data.notifications import (
    BaseNotificationData,
    DeliveryStream,
    get_delivery_stream,
    supports_list_unsubscribe,
)
from backend.notifications.renderer import EmailUrls, RenderedEmail, build_urls, render
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Bounds how long one send can hold a thread-pool worker.
POSTMARK_TIMEOUT_SECONDS = 30
# A sign-up waits on this lookup, so it gives up long before a send would.
SUPPRESSION_LOOKUP_TIMEOUT_SECONDS = 2

# Postmark tag for Better Auth's emails, which are not notifications.
AUTH_EMAIL_TAG = "auth"
# Postmark tag for SendEmailToMeBlock's emails to the account owner.
OWNER_EMAIL_TAG = "owner_email"


class TypedPostmarkClient(PostmarkClient):
    """Workaround so the type checker sees `emails`; the Postmark library ships
    no annotations."""

    emails: EmailManager


class EmailSender:
    def __init__(self):
        if settings.secrets.postmark_server_api_token:
            self.postmark = TypedPostmarkClient(
                server_token=settings.secrets.postmark_server_api_token,
                # postmarker defaults to no timeout, which hands `requests` an
                # unbounded wait. Every send occupies a thread-pool worker, so a
                # hung Postmark call would otherwise exhaust the pool.
                timeout=POSTMARK_TIMEOUT_SECONDS,
            )
            self.postmark_lookups = TypedPostmarkClient(
                server_token=settings.secrets.postmark_server_api_token,
                timeout=SUPPRESSION_LOOKUP_TIMEOUT_SECONDS,
            )
        else:
            logger.warning(
                "Postmark server API token not found, email sending disabled"
            )
            self.postmark = None
            self.postmark_lookups = None

    async def send_notification(
        self,
        notification_type: NotificationType,
        user_email: str,
        data: BaseNotificationData,
        unsubscribe_link: str,
        volume_links: dict[str, str] | None = None,
        first_name: str | None = None,
    ) -> None:
        """Render and send one notification. Raises on delivery failure so the
        queue consumer's retry-with-backoff can recover."""
        urls = build_urls(unsubscribe_link, volume_links)
        email = render(notification_type, data, user_email, urls, first_name=first_name)
        await self._deliver(notification_type, user_email, email, urls)

    async def _deliver(
        self,
        notification_type: NotificationType,
        user_email: str,
        email: RenderedEmail,
        urls: EmailUrls,
    ) -> None:
        headers = (
            {
                "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
                "List-Unsubscribe": f"<{urls.unsubscribe}>",
            }
            if supports_list_unsubscribe(notification_type)
            else None
        )
        stream = get_delivery_stream(notification_type)
        await self._send(
            user_email=user_email,
            tag=notification_type.value.lower(),
            sender=_sender_for(stream),
            reply_to=_reply_to_for(stream),
            subject=email.subject,
            html_body=email.html,
            text_body=email.text,
            headers=headers,
        )

    def send_email_or_raise(
        self, user_email: str, subject: str, body: str, text_body: str | None = None
    ) -> None:
        """Send a one-off transactional email (e.g. Better Auth password-reset
        and verification links) with no notification templating or preference
        gating. Raises if the Postmark client is not configured so callers can
        surface the failure instead of silently dropping an auth email.
        Fake preview recipients are intentionally skipped."""
        if _suppress_preview_email(user_email):
            return
        if not self.postmark:
            raise RuntimeError("Postmark is not configured; cannot send email")
        self.postmark.emails.send(
            From=settings.config.postmark_sender_email,
            To=user_email,
            Subject=subject,
            HtmlBody=body,
            TextBody=text_body,
            MessageStream=settings.config.postmark_transactional_stream,
            Tag=AUTH_EMAIL_TAG,
            # A reset or verification link holds a token; link tracking would
            # route it through Postmark's redirect and into the click webhook.
            **_tracking(False),
        )

    def is_suppressed(self, user_email: str) -> bool:
        """Whether Postmark has marked this address inactive (a hard bounce,
        spam complaint or manual suppression), so a send to it would be refused.
        False when Postmark isn't configured or doesn't answer: the lookup must
        never block a sign-up."""
        if not self.postmark_lookups:
            return False
        stream = settings.config.postmark_transactional_stream
        try:
            found = self.postmark_lookups.call(
                "GET",
                f"message-streams/{stream}/suppressions/dump",
                EmailAddress=user_email,
            )
        except (ClientError, RequestException, ValueError) as e:
            logger.warning(f"Postmark suppression lookup failed: {type(e).__name__}")
            return False
        return any(
            entry.get("EmailAddress", "").lower() == user_email.lower()
            for entry in found.get("Suppressions", [])
        )

    async def send_owner_email(
        self,
        user_email: str,
        email: RenderedEmail,
        attachments: list[dict[str, str]],
    ) -> None:
        """Raises when Postmark is not configured: the run that asked for this
        reports a failure instead of a send that never happened."""
        if not self.postmark:
            raise RuntimeError("Email sending isn't set up on this server.")
        await self._send(
            user_email=user_email,
            tag=OWNER_EMAIL_TAG,
            sender=_sender_for(DeliveryStream.PRODUCT),
            reply_to=_reply_to_for(DeliveryStream.PRODUCT),
            subject=email.subject,
            html_body=email.html,
            text_body=email.text,
            headers=None,
            attachments=attachments,
            # Its links are the user's own, and can carry their tokens.
            tracked=False,
        )

    async def _send(
        self,
        user_email: str,
        tag: str,
        sender: str,
        subject: str,
        html_body: str,
        text_body: str,
        headers: dict[str, str] | None,
        reply_to: str | None = None,
        attachments: list[dict[str, str]] | None = None,
        tracked: bool = True,
    ) -> None:
        if _suppress_preview_email(user_email):
            return
        if not self.postmark:
            logger.warning("Email tried to send without Postmark configured")
            return
        logger.debug("Sending email to %s with subject %s", user_email, subject)
        # postmarker's send is a blocking HTTP call; keep it off the event loop
        # so a slow Postmark response can't stall the notification service.
        await asyncio.to_thread(
            self.postmark.emails.send,
            From=sender,
            ReplyTo=reply_to,
            To=user_email,
            Subject=subject,
            HtmlBody=html_body,
            TextBody=text_body,
            MessageStream=settings.config.postmark_transactional_stream,
            Headers=headers,
            Attachments=attachments,
            Tag=tag,
            **_tracking(tracked and settings.config.postmark_track_opens_and_clicks),
        )


def _suppress_preview_email(user_email: str) -> bool:
    """Preview accounts have fake addresses that would damage sender reputation."""
    if not user_email.strip().lower().endswith("@previews.agpt.co"):
        return False
    logger.info("Skipping transactional email to the preview recipient domain")
    return True


def _tracking(enabled: bool) -> dict[str, Any]:
    """Postmark's per-message tracking fields. Links are rewritten in the
    HTML part only: a rewritten link in the plain-text part is a long opaque
    redirect a reader can't check before following."""
    if enabled:
        return {"TrackOpens": True, "TrackLinks": "HtmlOnly"}
    return {"TrackOpens": False, "TrackLinks": "None"}


def _sender_for(stream: DeliveryStream) -> str:
    return {
        DeliveryStream.BILLING: settings.config.billing_sender_email,
        DeliveryStream.PRODUCT: settings.config.product_sender_email,
        DeliveryStream.OPS: settings.config.ops_sender_email,
    }[stream]


def _reply_to_for(stream: DeliveryStream) -> str | None:
    return {
        DeliveryStream.BILLING: settings.config.billing_reply_to_email,
        DeliveryStream.PRODUCT: settings.config.product_reply_to_email,
        DeliveryStream.OPS: None,
    }[stream]
