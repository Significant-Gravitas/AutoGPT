"""Shared enums and response models for the All Quiet blocks.

Field names mirror the All Quiet Public API v1 payloads so responses can be
validated straight off the wire.
"""

from enum import Enum
from typing import Optional

from backend.sdk import BaseModel, Field


class AllQuietRegion(str, Enum):
    """All Quiet runs separate US and EU deployments; an API key is valid on one."""

    US = "us"
    EU = "eu"


class IncidentStatus(str, Enum):
    """The only two statuses the API accepts (`Values must be one of Open,Resolved`)."""

    OPEN = "Open"
    RESOLVED = "Resolved"


class IncidentSeverity(str, Enum):
    """Severity ladder used for routing and escalation."""

    CRITICAL = "Critical"
    WARNING = "Warning"
    MINOR = "Minor"


class IncidentIntent(str, Enum):
    """Intents accepted by the incident patch endpoint's `appendIntent` operation.

    Which intents an incident accepts depends on its current status: an Open
    incident allows e.g. `Investigated`/`Resolved`/`Escalated`, while a Resolved
    one allows `Unresolved`. `GetIncident` surfaces the current set on
    `allowed_intents`.
    """

    INVESTIGATED = "Investigated"
    RESOLVED = "Resolved"
    UNRESOLVED = "Unresolved"
    ESCALATED = "Escalated"
    COMMENTED = "Commented"
    SNOOZED = "Snoozed"
    ARCHIVED = "Archived"


class IncidentSortBy(str, Enum):
    """Sort fields the incident search accepts.

    (`Value must be one of Urgency,LatestInteraction,LastUpdatedAt,Created,Title`)
    """

    CREATED = "Created"
    LAST_UPDATED_AT = "LastUpdatedAt"
    LATEST_INTERACTION = "LatestInteraction"
    URGENCY = "Urgency"
    TITLE = "Title"


class AllQuietEntity(BaseModel):
    """A named reference to a team, service or integration."""

    id: str = ""
    display_name: str = Field(default="", alias="displayName")

    model_config = {"populate_by_name": True}


class AllQuietUser(BaseModel):
    id: str = ""
    display_name: str = Field(default="", alias="displayName")
    email: str = ""
    avatar_url: str = Field(default="", alias="avatarUrl")

    model_config = {"populate_by_name": True}


class IncidentAttribute(BaseModel):
    """A key/value pair carried on an incident, e.g. `host: web-01`."""

    name: str = ""
    value: str = ""
    is_image: bool = Field(default=False, alias="isImage")
    is_grouping_key: bool = Field(default=False, alias="isGroupingKey")
    hide_in_previews: bool = Field(default=False, alias="hideInPreviews")

    model_config = {"populate_by_name": True}


class Incident(BaseModel):
    """An All Quiet incident.

    Status and severity live on the incident's event history rather than the
    incident itself, so they are flattened here from the most recent event.
    """

    id: str = ""
    title: str = ""
    status: Optional[IncidentStatus] = None
    severity: Optional[IncidentSeverity] = None
    created_at: str = Field(default="", alias="createdAt")
    last_updated_at: str = Field(default="", alias="lastUpdatedAt")
    integration: Optional[AllQuietEntity] = None
    teams: list[AllQuietEntity] = Field(default_factory=list)
    services: list[AllQuietEntity] = Field(default_factory=list)
    on_call_users: list[AllQuietUser] = Field(default_factory=list, alias="onCallUsers")
    attributes: list[IncidentAttribute] = Field(default_factory=list)
    allowed_intents: list[str] = Field(default_factory=list, alias="allowedIntents")
    unattended: bool = False
    is_archived: bool = Field(default=False, alias="isArchived")

    model_config = {"populate_by_name": True}


class OnCallAvailability(BaseModel):
    """One escalation tier a user covers for a team."""

    tier: int = 0
    is_online: bool = Field(default=False, alias="isOnline")
    fill_up: bool = Field(default=False, alias="fillUp")

    model_config = {"populate_by_name": True}


class OnCallShift(BaseModel):
    """A user's on-call coverage for one team at the requested point in time."""

    user: Optional[AllQuietUser] = None
    team: Optional[AllQuietEntity] = None
    availabilities: list[OnCallAvailability] = Field(default_factory=list)


class Team(BaseModel):
    id: str = ""
    display_name: str = Field(default="", alias="displayName")
    time_zone_id: str = Field(default="", alias="timeZoneId")
    labels: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}
