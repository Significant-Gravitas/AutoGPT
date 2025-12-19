"""
Direct email sending for authentication flows.

This module bypasses the notification queue system to ensure auth emails
(password reset, email verification) are sent immediately in all environments.
"""

import logging
import pathlib
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from postmarker.core import PostmarkClient

from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Template directory
TEMPLATE_DIR = pathlib.Path(__file__).parent / "templates"


class AuthEmailSender:
    """Handles direct email sending for authentication flows."""

    def __init__(self):
        if settings.secrets.postmark_server_api_token:
            self.postmark = PostmarkClient(
                server_token=settings.secrets.postmark_server_api_token
            )
        else:
            logger.warning(
                "Postmark server API token not found, auth email sending disabled"
            )
            self.postmark = None

        # Set up Jinja2 environment for templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=True,
        )

    def _get_frontend_url(self) -> str:
        """Get the frontend base URL for email links."""
        return (
            settings.config.frontend_base_url
            or settings.config.platform_base_url
            or "http://localhost:3000"
        )

    def _render_template(
        self, template_name: str, subject: str, **context
    ) -> tuple[str, str]:
        """Render an email template with the base template wrapper."""
        # Render the content template
        content_template = self.jinja_env.get_template(template_name)
        content = content_template.render(**context)

        # Render with base template
        base_template = self.jinja_env.get_template("base.html.jinja2")
        html_body = base_template.render(
            data={"title": subject, "message": content, "unsubscribe_link": None}
        )

        return subject, html_body

    def _send_email(self, to_email: str, subject: str, html_body: str) -> bool:
        """Send an email directly via Postmark."""
        if not self.postmark:
            logger.warning(
                f"Postmark not configured. Would send email to {to_email}: {subject}"
            )
            return False

        try:
            self.postmark.emails.send(
                From=settings.config.postmark_sender_email,
                To=to_email,
                Subject=subject,
                HtmlBody=html_body,
            )
            logger.info(f"Auth email sent to {to_email}: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send auth email to {to_email}: {e}")
            return False

    def send_password_reset_email(
        self, to_email: str, reset_token: str, user_name: Optional[str] = None
    ) -> bool:
        """
        Send a password reset email.

        Args:
            to_email: Recipient email address
            reset_token: The raw password reset token
            user_name: Optional user name for personalization

        Returns:
            True if email was sent successfully, False otherwise
        """
        frontend_url = self._get_frontend_url()
        reset_link = f"{frontend_url}/reset-password?token={reset_token}"

        subject, html_body = self._render_template(
            "password_reset.html.jinja2",
            subject="Reset Your AutoGPT Password",
            reset_link=reset_link,
            user_name=user_name,
            frontend_url=frontend_url,
        )

        return self._send_email(to_email, subject, html_body)

    def send_email_verification(
        self, to_email: str, verification_token: str, user_name: Optional[str] = None
    ) -> bool:
        """
        Send an email verification email.

        Args:
            to_email: Recipient email address
            verification_token: The raw verification token
            user_name: Optional user name for personalization

        Returns:
            True if email was sent successfully, False otherwise
        """
        frontend_url = self._get_frontend_url()
        verification_link = f"{frontend_url}/verify-email?token={verification_token}"

        subject, html_body = self._render_template(
            "email_verification.html.jinja2",
            subject="Verify Your AutoGPT Email",
            verification_link=verification_link,
            user_name=user_name,
            frontend_url=frontend_url,
        )

        return self._send_email(to_email, subject, html_body)


# Singleton instance
_auth_email_sender: Optional[AuthEmailSender] = None


def get_auth_email_sender() -> AuthEmailSender:
    """Get or create the auth email sender singleton."""
    global _auth_email_sender
    if _auth_email_sender is None:
        _auth_email_sender = AuthEmailSender()
    return _auth_email_sender
