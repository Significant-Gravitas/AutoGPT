"""
Email service for authentication flows.

Uses Postmark to send transactional emails for:
- Email verification
- Password reset
- Account security notifications
"""

import logging
import pathlib
from typing import Optional

from jinja2 import Template
from postmarker.core import PostmarkClient

from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Template directory
TEMPLATE_DIR = pathlib.Path(__file__).parent / "templates"


class AuthEmailService:
    """Email service for authentication-related emails."""

    def __init__(self):
        if settings.secrets.postmark_server_api_token:
            self.postmark = PostmarkClient(
                server_token=settings.secrets.postmark_server_api_token
            )
            self.enabled = True
        else:
            logger.warning(
                "Postmark server API token not found, auth emails disabled"
            )
            self.postmark = None
            self.enabled = False

        self.sender_email = settings.config.postmark_sender_email
        self.frontend_url = (
            settings.config.frontend_base_url or settings.config.platform_base_url
        )

    def _send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
    ) -> bool:
        """
        Send an email via Postmark.

        Returns True if sent successfully, False otherwise.
        """
        if not self.enabled or not self.postmark:
            logger.warning(f"Email not sent (disabled): {subject} to {to_email}")
            return False

        try:
            self.postmark.emails.send(
                From=self.sender_email,
                To=to_email,
                Subject=subject,
                HtmlBody=html_body,
            )
            logger.info(f"Auth email sent: {subject} to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send auth email: {e}")
            return False

    def send_verification_email(self, email: str, token: str) -> bool:
        """
        Send email verification link.

        Args:
            email: Recipient email address
            token: Verification token

        Returns:
            True if sent successfully
        """
        verify_url = f"{self.frontend_url}/auth/verify-email?token={token}"

        subject = "Verify your email address"
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .button {{ display: inline-block; padding: 12px 24px; background-color: #5046e5; color: white; text-decoration: none; border-radius: 6px; font-weight: 500; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Verify your email address</h2>
                <p>Thanks for signing up! Please verify your email address by clicking the button below:</p>
                <p style="margin: 30px 0;">
                    <a href="{verify_url}" class="button">Verify Email</a>
                </p>
                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; color: #666;">{verify_url}</p>
                <p>This link will expire in 24 hours.</p>
                <div class="footer">
                    <p>If you didn't create an account, you can safely ignore this email.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return self._send_email(email, subject, html_body)

    def send_password_reset_email(self, email: str, token: str) -> bool:
        """
        Send password reset link.

        Args:
            email: Recipient email address
            token: Password reset token

        Returns:
            True if sent successfully
        """
        reset_url = f"{self.frontend_url}/reset-password?token={token}"

        subject = "Reset your password"
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .button {{ display: inline-block; padding: 12px 24px; background-color: #5046e5; color: white; text-decoration: none; border-radius: 6px; font-weight: 500; }}
                .warning {{ background-color: #fef3c7; border: 1px solid #f59e0b; padding: 12px; border-radius: 6px; margin: 20px 0; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Reset your password</h2>
                <p>We received a request to reset your password. Click the button below to choose a new password:</p>
                <p style="margin: 30px 0;">
                    <a href="{reset_url}" class="button">Reset Password</a>
                </p>
                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; color: #666;">{reset_url}</p>
                <div class="warning">
                    <strong>This link will expire in 15 minutes.</strong>
                </div>
                <div class="footer">
                    <p>If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return self._send_email(email, subject, html_body)

    def send_password_changed_notification(self, email: str) -> bool:
        """
        Send notification that password was changed.

        Args:
            email: Recipient email address

        Returns:
            True if sent successfully
        """
        subject = "Your password was changed"
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .warning {{ background-color: #fee2e2; border: 1px solid #ef4444; padding: 12px; border-radius: 6px; margin: 20px 0; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Password Changed</h2>
                <p>Your password was successfully changed.</p>
                <div class="warning">
                    <strong>If you didn't make this change</strong>, please contact support immediately and reset your password.
                </div>
                <div class="footer">
                    <p>This is an automated security notification.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return self._send_email(email, subject, html_body)

    def send_migrated_user_password_reset(self, email: str, token: str) -> bool:
        """
        Send password reset email for users migrated from Supabase.

        Args:
            email: Recipient email address
            token: Password reset token

        Returns:
            True if sent successfully
        """
        reset_url = f"{self.frontend_url}/reset-password?token={token}"

        subject = "Action Required: Set your password"
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .button {{ display: inline-block; padding: 12px 24px; background-color: #5046e5; color: white; text-decoration: none; border-radius: 6px; font-weight: 500; }}
                .info {{ background-color: #dbeafe; border: 1px solid #3b82f6; padding: 12px; border-radius: 6px; margin: 20px 0; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Set Your Password</h2>
                <div class="info">
                    <strong>We've upgraded our authentication system!</strong>
                    <p style="margin: 8px 0 0 0;">For enhanced security, please set a new password to continue using your account.</p>
                </div>
                <p>Click the button below to set your password:</p>
                <p style="margin: 30px 0;">
                    <a href="{reset_url}" class="button">Set Password</a>
                </p>
                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; color: #666;">{reset_url}</p>
                <p>This link will expire in 24 hours.</p>
                <div class="footer">
                    <p>If you signed up with Google, no action is needed - simply continue signing in with Google.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return self._send_email(email, subject, html_body)


# Singleton instance
_email_service: Optional[AuthEmailService] = None


def get_auth_email_service() -> AuthEmailService:
    """Get the singleton auth email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = AuthEmailService()
    return _email_service
