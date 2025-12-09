"""
OAuth Audit Logging.

Logs all OAuth-related operations for security auditing and compliance.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from backend.data.db import prisma

logger = logging.getLogger(__name__)


class OAuthEventType(str, Enum):
    """Types of OAuth events to audit."""

    # Client events
    CLIENT_REGISTERED = "client.registered"
    CLIENT_UPDATED = "client.updated"
    CLIENT_DELETED = "client.deleted"
    CLIENT_SECRET_ROTATED = "client.secret_rotated"
    CLIENT_SUSPENDED = "client.suspended"
    CLIENT_ACTIVATED = "client.activated"

    # Authorization events
    AUTHORIZATION_REQUESTED = "authorization.requested"
    AUTHORIZATION_GRANTED = "authorization.granted"
    AUTHORIZATION_DENIED = "authorization.denied"
    AUTHORIZATION_REVOKED = "authorization.revoked"

    # Token events
    TOKEN_ISSUED = "token.issued"
    TOKEN_REFRESHED = "token.refreshed"
    TOKEN_REVOKED = "token.revoked"
    TOKEN_EXPIRED = "token.expired"

    # Grant events
    GRANT_CREATED = "grant.created"
    GRANT_UPDATED = "grant.updated"
    GRANT_REVOKED = "grant.revoked"
    GRANT_USED = "grant.used"

    # Credential events
    CREDENTIAL_CONNECTED = "credential.connected"
    CREDENTIAL_DELETED = "credential.deleted"

    # Execution events
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_CANCELLED = "execution.cancelled"


async def log_oauth_event(
    event_type: OAuthEventType,
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
    grant_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> str:
    """
    Log an OAuth audit event.

    Args:
        event_type: Type of event
        user_id: User ID involved (if any)
        client_id: OAuth client ID involved (if any)
        grant_id: Grant ID involved (if any)
        ip_address: Client IP address
        user_agent: Client user agent
        details: Additional event details

    Returns:
        ID of the created audit log entry
    """
    try:
        from prisma import Json

        audit_entry = await prisma.oauthauditlog.create(
            data={
                "eventType": event_type.value,
                "userId": user_id,
                "clientId": client_id,
                "grantId": grant_id,
                "ipAddress": ip_address,
                "userAgent": user_agent,
                "details": Json(details or {}),
            }
        )

        logger.debug(
            f"OAuth audit: {event_type.value} - "
            f"user={user_id}, client={client_id}, grant={grant_id}"
        )

        return audit_entry.id

    except Exception as e:
        # Log but don't fail the operation if audit logging fails
        logger.error(f"Failed to create OAuth audit log: {e}")
        return ""


async def get_audit_logs(
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
    event_type: Optional[OAuthEventType] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> list:
    """
    Query OAuth audit logs.

    Args:
        user_id: Filter by user ID
        client_id: Filter by client ID
        event_type: Filter by event type
        start_date: Filter by start date
        end_date: Filter by end date
        limit: Maximum number of results
        offset: Offset for pagination

    Returns:
        List of audit log entries
    """
    where: dict[str, Any] = {}

    if user_id:
        where["userId"] = user_id
    if client_id:
        where["clientId"] = client_id
    if event_type:
        where["eventType"] = event_type.value
    if start_date:
        where["createdAt"] = {"gte": start_date}
    if end_date:
        if "createdAt" in where:
            where["createdAt"]["lte"] = end_date
        else:
            where["createdAt"] = {"lte": end_date}

    return await prisma.oauthauditlog.find_many(
        where=where if where else None,  # type: ignore[arg-type]
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
    )


async def cleanup_old_audit_logs(days_to_keep: int = 90) -> int:
    """
    Delete audit logs older than the specified number of days.

    Args:
        days_to_keep: Number of days of logs to retain

    Returns:
        Number of logs deleted
    """
    from datetime import timedelta

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

    result = await prisma.oauthauditlog.delete_many(
        where={"createdAt": {"lt": cutoff_date}}
    )

    logger.info(f"Cleaned up {result} OAuth audit logs older than {days_to_keep} days")
    return result
