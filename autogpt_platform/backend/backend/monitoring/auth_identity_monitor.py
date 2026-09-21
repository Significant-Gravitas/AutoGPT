"""Auth identity <-> platform User invariant monitor.

Every auth identity (Better Auth user) must have a platform ``User`` row with
the same id. Three things maintain that: the auth hook writes the row at
sign-up, the client's ``POST /auth/user`` writes it after sign-in, and every
authenticated backend request self-heals it. This monitor is the backstop for
all three: it heals any identity that slipped through anyway and pages, so a
regression in the invariant is seen the first time it happens rather than
when a user reports a broken account.
"""

import logging

from backend.util.clients import (
    get_database_manager_client,
    get_notification_manager_client,
)
from backend.util.metrics import sentry_capture_error
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()


class OrphanedAuthIdentityException(Exception):
    """Raised (to Sentry) when auth identities without a platform User exist."""


class AuthIdentityMonitor:
    """Heal orphaned auth identities and alert when there were any."""

    def __init__(self):
        self.config = config
        self.notification_client = get_notification_manager_client()

    def check_orphaned_auth_identities(self) -> str:
        report = get_database_manager_client().heal_orphaned_auth_identities(
            grace_secs=self.config.auth_identity_orphan_grace_secs,
            limit=self.config.auth_identity_orphan_check_limit,
        )

        if report.is_clean:
            return "No orphaned auth identities detected."

        # Ids only: the alert goes to Discord and Sentry, neither of which
        # should carry the emails behind these accounts.
        message_parts = [
            "Auth identities without a platform User row were found. "
            "Sign-up should have created these rows; investigate why it did not.",
        ]
        if report.healed:
            message_parts.append(
                f"* Healed {len(report.healed)} (User + Profile + personal org "
                f"provisioned): {', '.join(report.healed)}"
            )
        if report.collided:
            message_parts.append(
                f"* {len(report.collided)} could NOT be healed: their email is "
                "already owned by a different platform User, so these accounts "
                "need manual reconciliation: "
                + ", ".join(
                    f"{identity.id} (email owned by {identity.email_owner_id})"
                    for identity in report.collided
                )
            )
        if report.failed:
            message_parts.append(
                f"* {len(report.failed)} failed to provision (see logs): "
                f"{', '.join(report.failed)}"
            )

        msg = "\n".join(message_parts)
        sentry_capture_error(OrphanedAuthIdentityException(msg))
        self.notification_client.discord_system_alert(msg)
        return msg


def report_orphaned_auth_identities() -> str:
    """Heal orphaned auth identities and send alerts if there were any."""
    return AuthIdentityMonitor().check_orphaned_auth_identities()
