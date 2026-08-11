"""Postmark delivery for the notification families.

Everything goes out on a transactional stream, separate from marketing mail
(which the platform does not send at all — the onboarding tour and the monthly
changelog are MailerLite's). Every message carries a plain-text MIME part built
from the same data model, and one-click List-Unsubscribe headers on every
family except Ops, which is internal mail and deliberately not opt-in.
"""

import asyncio
import logging

from postmarker.core import PostmarkClient
from postmarker.models.emails import EmailManager
from prisma.enums import NotificationType

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


class TypedPostmarkClient(PostmarkClient):
    """Workaround so the type checker sees `emails`; the Postmark library ships
    no annotations."""

    emails: EmailManager


class EmailSender:
    def __init__(self):
        if settings.secrets.postmark_server_api_token:
            self.postmark = TypedPostmarkClient(
                server_token=settings.secrets.postmark_server_api_token
            )
        else:
            logger.warning(
                "Postmark server API token not found, email sending disabled"
            )
            self.postmark = None

    async def send_notification(
        self,
        notification_type: NotificationType,
        user_email: str,
        data: BaseNotificationData,
        unsubscribe_link: str,
    ) -> None:
        """Render and send one notification. Raises on delivery failure so the
        queue consumer's retry-with-backoff can recover."""
        urls = build_urls(unsubscribe_link)
        email = render(notification_type, data, user_email, urls)
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
        await self._send(
            user_email=user_email,
            sender=_sender_for(get_delivery_stream(notification_type)),
            subject=email.subject,
            html_body=email.html,
            text_body=email.text,
            headers=headers,
        )

    def send_email_or_raise(self, user_email: str, subject: str, body: str) -> None:
        """Send a one-off transactional email (e.g. Better Auth password-reset
        and verification links) with no notification templating or preference
        gating. Raises if the Postmark client is not configured so callers can
        surface the failure instead of silently dropping an auth email."""
        if not self.postmark:
            raise RuntimeError("Postmark is not configured; cannot send email")
        self.postmark.emails.send(
            From=settings.config.postmark_sender_email,
            To=user_email,
            Subject=subject,
            HtmlBody=body,
            MessageStream=settings.config.postmark_transactional_stream,
        )

    async def _send(
        self,
        user_email: str,
        sender: str,
        subject: str,
        html_body: str,
        text_body: str,
        headers: dict[str, str] | None,
    ) -> None:
        if not self.postmark:
            logger.warning("Email tried to send without Postmark configured")
            return
        logger.debug("Sending email to %s with subject %s", user_email, subject)
        # postmarker's send is a blocking HTTP call; keep it off the event loop
        # so a slow Postmark response can't stall the notification service.
        await asyncio.to_thread(
            self.postmark.emails.send,
            From=sender,
            To=user_email,
            Subject=subject,
            HtmlBody=html_body,
            TextBody=text_body,
            MessageStream=settings.config.postmark_transactional_stream,
            Headers=headers,
        )


def _sender_for(stream: DeliveryStream) -> str:
    return {
        DeliveryStream.BILLING: settings.config.billing_sender_email,
        DeliveryStream.PRODUCT: settings.config.product_sender_email,
        DeliveryStream.OPS: settings.config.ops_sender_email,
    }[stream]
