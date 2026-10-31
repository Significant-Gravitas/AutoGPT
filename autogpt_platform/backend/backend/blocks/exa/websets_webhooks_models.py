"""
Models shared by the Websets webhook and event blocks.
"""

from typing import Any

from pydantic import BaseModel

from backend.sdk import SchemaField

from ._exa_rest import WEBSETS_API_BASE
from ._platform_key import TeamScopedExaMixin
from ._webhook import LEGACY_EVENT_TYPES, ExaEventType

WEBHOOKS_URL = f"{WEBSETS_API_BASE}/webhooks"
EVENTS_URL = f"{WEBSETS_API_BASE}/events"


class WebsetWebhookModel(BaseModel):
    """A Websets webhook subscription."""

    id: str = SchemaField(description="The webhook ID")
    status: str = SchemaField(description="active, or inactive once deleted")
    events: list[str] = SchemaField(default_factory=list)
    url: str = SchemaField(default="", description="Where events are delivered")
    metadata: dict[str, str] = SchemaField(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "WebsetWebhookModel":
        return cls(
            id=data.get("id", ""),
            status=data.get("status", ""),
            events=data.get("events") or [],
            url=data.get("url", ""),
            metadata=data.get("metadata") or {},
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", ""),
        )


class WebhookAttemptModel(BaseModel):
    """One attempt by Exa to deliver an event to a webhook."""

    id: str
    event_id: str = ""
    event_type: str = ""
    webhook_id: str = ""
    url: str = ""
    successful: bool = False
    response_status_code: int | None = None
    response_body: str | None = None
    response_headers: dict[str, str] = SchemaField(default_factory=dict)
    attempt: int | None = None
    attempted_at: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "WebhookAttemptModel":
        status_code = data.get("responseStatusCode")
        attempt = data.get("attempt")
        return cls(
            id=data.get("id", ""),
            event_id=data.get("eventId", ""),
            event_type=data.get("eventType", ""),
            webhook_id=data.get("webhookId", ""),
            url=data.get("url", ""),
            successful=bool(data.get("successful", False)),
            response_status_code=int(status_code) if status_code is not None else None,
            response_body=data.get("responseBody"),
            response_headers=data.get("responseHeaders") or {},
            attempt=int(attempt) if attempt is not None else None,
            attempted_at=data.get("attemptedAt", ""),
        )


class WebsetEventModel(BaseModel):
    """An event recorded by the Websets API (kept for 60 days)."""

    id: str = SchemaField(description="The event ID")
    type: str = SchemaField(description="The event type, e.g. webset.item.created")
    created_at: str = SchemaField(default="", description="When it happened")
    data: dict[str, Any] = SchemaField(
        default_factory=dict,
        description="The full resource the event is about (webset, item, "
        "search, import or monitor)",
    )

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "WebsetEventModel":
        return cls(
            id=data.get("id", ""),
            type=data.get("type", ""),
            created_at=data.get("createdAt", ""),
            data=data.get("data") or {},
        )


def event_type_values(events: list[ExaEventType]) -> list[str]:
    """Event names to send to Exa, rejecting legacy ones Exa does not accept."""
    legacy = [event.value for event in events if event in LEGACY_EVENT_TYPES]
    if legacy:
        raise ValueError(
            f"{', '.join(legacy)} is a legacy event Exa no longer sends; "
            "use import.completed instead"
        )
    return [event.value for event in events]


class WebsetsTeamApiMixin(TeamScopedExaMixin):
    """Exa calls for Exa Websets webhooks and events, which act on the whole Exa team."""

    TEAM_SCOPED_FEATURE = "Exa Websets webhooks and events"
