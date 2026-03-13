import logging
import pathlib
from typing import Any

from postmarker.core import PostmarkClient
from postmarker.models.emails import EmailManager
from prisma.enums import NotificationType
from pydantic import BaseModel

from backend.data.notifications import (
    AgentRunData,
    NotificationDataType_co,
    NotificationEventModel,
    NotificationTypeOverride,
)
from backend.util.settings import Settings
from backend.util.text import TextFormatter

logger = logging.getLogger(__name__)
settings = Settings()


# The following is a workaround to get the type checker to recognize the EmailManager type
# This is a temporary solution and should be removed once the Postmark library is updated
# to support type annotations.
class TypedPostmarkClient(PostmarkClient):
    emails: EmailManager


class Template(BaseModel):
    subject_template: str
    body_template: str
    base_template: str


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
        self.formatter = TextFormatter()

    MAX_EMAIL_CHARS = 5_000_000  # ~5MB buffer

    def _build_large_output_summary(
        self,
        data: (
            NotificationEventModel[NotificationDataType_co]
            | list[NotificationEventModel[NotificationDataType_co]]
        ),
        *,
        email_size: int,
        base_url: str,
    ) -> str:
        if isinstance(data, list):
            if not data:
                return (
                    "⚠️ A notification generated a very large output "
                    f"({email_size / 1_000_000:.2f} MB)."
                )
            event = data[0]
        else:
            event = data
        execution_url = (
            f"{base_url}/executions/{event.id}" if event.id is not None else None
        )

        if isinstance(event.data, AgentRunData):
            lines = [
                f"⚠️ Your agent '{event.data.agent_name}' generated a very large output ({email_size / 1_000_000:.2f} MB).",
                "",
                f"Execution time: {event.data.execution_time}",
                f"Credits used: {event.data.credits_used}",
            ]
            if execution_url is not None:
                lines.append(f"View full results: {execution_url}")
            return "\n".join(lines)

        lines = [
            f"⚠️ A notification generated a very large output ({email_size / 1_000_000:.2f} MB).",
        ]
        if execution_url is not None:
            lines.extend(["", f"View full results: {execution_url}"])
        return "\n".join(lines)

    def send_template(
        self,
        *,
        user_email: str,
        subject: str,
        template_name: str,
        data: dict[str, Any] | None = None,
        user_unsubscribe_link: str | None = None,
    ) -> None:
        if not self.postmark:
            logger.warning("Postmark client not initialized, email not sent")
            return

        base_url = (
            settings.config.frontend_base_url or settings.config.platform_base_url
        )
        unsubscribe_link = user_unsubscribe_link or f"{base_url}/profile/settings"

        _, full_message = self.formatter.format_email(
            subject_template="{{ subject }}",
            base_template=self._read_template("templates/base.html.jinja2"),
            content_template=self._read_template(f"templates/{template_name}"),
            data={"subject": subject, **(data or {})},
            unsubscribe_link=unsubscribe_link,
        )

        self._send_email(
            user_email=user_email,
            subject=subject,
            body=full_message,
            user_unsubscribe_link=user_unsubscribe_link,
        )

    def send_templated(
        self,
        notification: NotificationType,
        user_email: str,
        data: (
            NotificationEventModel[NotificationDataType_co]
            | list[NotificationEventModel[NotificationDataType_co]]
        ),
        user_unsub_link: str | None = None,
    ):
        """Send an email to a user using a template pulled from the notification type, or fallback"""
        if not self.postmark:
            logger.warning("Postmark client not initialized, email not sent")
            return

        template = self._get_template(notification)

        base_url = (
            settings.config.frontend_base_url or settings.config.platform_base_url
        )

        # Normalize data
        template_data = {"notifications": data} if isinstance(data, list) else data

        try:
            subject, full_message = self.formatter.format_email(
                base_template=template.base_template,
                subject_template=template.subject_template,
                content_template=template.body_template,
                data=template_data,
                unsubscribe_link=f"{base_url}/profile/settings",
            )
        except Exception as e:
            logger.error(f"Error formatting full message: {e}")
            raise e

        # Check email size & send summary if too large
        email_size = len(full_message)
        if email_size > self.MAX_EMAIL_CHARS:
            logger.warning(
                f"Email size ({email_size} chars) exceeds safe limit. "
                "Sending summary email instead."
            )

            summary_message = self._build_large_output_summary(
                data,
                email_size=email_size,
                base_url=base_url,
            )

            self._send_email(
                user_email=user_email,
                subject=f"{subject} (Output Too Large)",
                body=summary_message,
                user_unsubscribe_link=user_unsub_link,
            )
            return  # Skip sending full email

        logger.debug(f"Sending email with size: {email_size} characters")
        self._send_email(
            user_email=user_email,
            subject=subject,
            body=full_message,
            user_unsubscribe_link=user_unsub_link,
        )

    def _get_template(self, notification: NotificationType):
        # convert the notification type to a notification type override
        notification_type_override = NotificationTypeOverride(notification)
        # find the template in templates/name.html (the .template returns with the .html)
        template_path = f"templates/{notification_type_override.template}.jinja2"
        logger.debug(
            f"Template full path: {pathlib.Path(__file__).parent / template_path}"
        )
        base_template = self._read_template("templates/base.html.jinja2")
        template = self._read_template(template_path)
        return Template(
            subject_template=notification_type_override.subject,
            body_template=template,
            base_template=base_template,
        )

    def _read_template(self, template_path: str) -> str:
        with open(pathlib.Path(__file__).parent / template_path, "r") as file:
            return file.read()

    def _send_email(
        self,
        user_email: str,
        subject: str,
        body: str,
        user_unsubscribe_link: str | None = None,
    ):
        if not self.postmark:
            logger.warning("Email tried to send without postmark configured")
            return
        logger.debug(f"Sending email to {user_email} with subject {subject}")
        self.postmark.emails.send(
            From=settings.config.postmark_sender_email,
            To=user_email,
            Subject=subject,
            HtmlBody=body,
            Headers=(
                {
                    "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
                    "List-Unsubscribe": f"<{user_unsubscribe_link}>",
                }
                if user_unsubscribe_link
                else None
            ),
        )

    def send_html(
        self,
        user_email: str,
        subject: str,
        body: str,
        user_unsubscribe_link: str | None = None,
    ) -> None:
        self._send_email(
            user_email=user_email,
            subject=subject,
            body=body,
            user_unsubscribe_link=user_unsubscribe_link,
        )
