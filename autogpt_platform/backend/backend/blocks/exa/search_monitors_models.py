"""
Models shared by the Exa Search Monitor blocks (``/monitors`` on the main API).

Search Monitors run an Exa search on a schedule and post new results to a
webhook. They are separate from Websets monitors (``/websets/v0/monitors``),
which add items to a webset and are managed by ``websets_monitor.py``.
"""

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel

from backend.sdk import SchemaField

from ._exa_rest import EXA_API_BASE
from ._platform_key import TeamScopedExaMixin
from .helpers import ContentSettings, process_contents_settings

MONITORS_URL = f"{EXA_API_BASE}/monitors"

# Exa documents a single unit and a one-hour minimum ("1h", "6h", "1d", "7d").
_PERIOD_PATTERN = re.compile(r"^([1-9][0-9]*)([hd])$")
# Keys of `SearchMonitorContents` in Exa's spec.
_MONITOR_CONTENT_KEYS = frozenset(
    {
        "text",
        "highlights",
        "summary",
        "extras",
        "livecrawl",
        "livecrawlTimeout",
        "maxAgeHours",
        "filterEmptyResults",
        "subpages",
        "subpageTarget",
        "context",
    }
)


class SearchMonitorStatus(str, Enum):
    """Monitor states Exa reports."""

    ACTIVE = "active"  # runs on schedule and can be triggered manually
    PAUSED = "paused"  # only runs when triggered manually
    DISABLED = "disabled"  # turned off by Exa after repeated auth failures


class SearchMonitorSettableStatus(str, Enum):
    """States a monitor can be put into by an update."""

    ACTIVE = "active"
    PAUSED = "paused"


class SearchMonitorWebhookEvent(str, Enum):
    """Events a Search Monitor can post to its webhook."""

    MONITOR_CREATED = "monitor.created"
    MONITOR_UPDATED = "monitor.updated"
    MONITOR_DELETED = "monitor.deleted"
    RUN_CREATED = "monitor.run.created"
    RUN_COMPLETED = "monitor.run.completed"


class SearchMonitorRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchMonitorModel(BaseModel):
    """A Search Monitor as returned by Exa."""

    id: str = SchemaField(description="The monitor ID")
    name: str | None = SchemaField(default=None, description="Display name")
    status: str = SchemaField(description="active, paused or disabled")
    query: str = SchemaField(default="", description="The search query it runs")
    num_results: int | None = SchemaField(
        default=None, description="Results requested per run"
    )
    include_domains: list[str] = SchemaField(default_factory=list)
    exclude_domains: list[str] = SchemaField(default_factory=list)
    contents: dict[str, Any] = SchemaField(
        default_factory=dict, description="Content options applied to each result"
    )
    period: str | None = SchemaField(
        default=None, description="Run interval, e.g. 1d; empty if manual-only"
    )
    output_schema: dict[str, Any] | None = SchemaField(
        default=None, description="Format of each run's synthesized output"
    )
    metadata: dict[str, str] = SchemaField(default_factory=dict)
    webhook_url: str = SchemaField(default="", description="Where runs are posted")
    webhook_events: list[str] = SchemaField(
        default_factory=list, description="Subscribed events; empty means all"
    )
    next_run_at: str | None = SchemaField(
        default=None, description="Next scheduled run (ISO 8601)"
    )
    created_at: str = SchemaField(default="")
    updated_at: str = SchemaField(default="")

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "SearchMonitorModel":
        search = data.get("search") or {}
        trigger = data.get("trigger") or {}
        webhook = data.get("webhook") or {}
        return cls(
            id=data.get("id", ""),
            name=data.get("name"),
            status=data.get("status", ""),
            query=search.get("query", ""),
            num_results=search.get("numResults"),
            include_domains=search.get("includeDomains") or [],
            exclude_domains=search.get("excludeDomains") or [],
            contents=search.get("contents") or {},
            period=trigger.get("period"),
            output_schema=data.get("outputSchema"),
            metadata=data.get("metadata") or {},
            webhook_url=webhook.get("url", ""),
            webhook_events=webhook.get("events") or [],
            next_run_at=data.get("nextRunAt"),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", ""),
        )


class MonitorCitation(BaseModel):
    url: str = SchemaField(description="Source URL")
    title: str = SchemaField(default="", description="Source title")


class MonitorGrounding(BaseModel):
    """Citations backing one field of a run's synthesized output."""

    field: str = SchemaField(description="The output field these citations support")
    citations: list[MonitorCitation] = SchemaField(default_factory=list)
    confidence: str = SchemaField(default="", description="low, medium or high")


class SearchMonitorRunModel(BaseModel):
    """One run of a Search Monitor."""

    id: str = SchemaField(description="The run ID")
    monitor_id: str = SchemaField(default="", description="The monitor it belongs to")
    status: str = SchemaField(
        description="pending, running, completed, failed or cancelled"
    )
    results: list[dict[str, Any]] = SchemaField(
        default_factory=list, description="New search results found by the run"
    )
    content: Any = SchemaField(
        default=None,
        description="Synthesized findings: text, or JSON matching the output schema",
    )
    grounding: list[MonitorGrounding] = SchemaField(default_factory=list)
    fail_reason: str | None = SchemaField(
        default=None, description="Why the run failed, if it did"
    )
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    cancelled_at: str | None = None
    duration_ms: int | None = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "SearchMonitorRunModel":
        output = data.get("output") or {}
        return cls(
            id=data.get("id", ""),
            monitor_id=data.get("monitorId", ""),
            status=data.get("status", ""),
            results=output.get("results") or [],
            content=output.get("content"),
            grounding=[
                MonitorGrounding.model_validate(item)
                for item in output.get("grounding") or []
            ],
            fail_reason=data.get("failReason"),
            started_at=data.get("startedAt"),
            completed_at=data.get("completedAt"),
            failed_at=data.get("failedAt"),
            cancelled_at=data.get("cancelledAt"),
            duration_ms=data.get("durationMs"),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", ""),
        )


def build_trigger(period: str) -> dict[str, str]:
    """Validate a run interval and build Exa's ``trigger`` object."""
    cleaned = period.strip().lower()
    if not _PERIOD_PATTERN.match(cleaned):
        raise ValueError(
            f"Invalid run interval '{period}'. Use a whole number followed by "
            "h (hours) or d (days), e.g. 1h, 6h, 1d or 7d."
        )
    return {"type": "interval", "period": cleaned}


def build_contents(contents: ContentSettings | None) -> dict[str, Any]:
    """Convert content settings into the subset Search Monitors accept.

    The settings are shared with /search, which takes a few options monitors
    do not (e.g. ``snapshotAsOf`` or a summary JSON schema); those are dropped.
    """
    payload = {
        key: value
        for key, value in process_contents_settings(contents).items()
        if key in _MONITOR_CONTENT_KEYS
    }
    summary = payload.get("summary")
    if isinstance(summary, dict):
        payload["summary"] = {k: v for k, v in summary.items() if k == "query"}
    return payload


def build_search(
    *,
    query: str | None,
    num_results: int | None,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    contents: ContentSettings | None,
) -> dict[str, Any]:
    """Build Exa's ``search`` object, leaving out anything not set."""
    search: dict[str, Any] = {}
    if query:
        search["query"] = query
    if num_results is not None:
        search["numResults"] = num_results
    if include_domains:
        search["includeDomains"] = include_domains
    if exclude_domains:
        search["excludeDomains"] = exclude_domains
    contents_payload = build_contents(contents)
    if contents_payload:
        search["contents"] = contents_payload
    return search


def build_webhook(
    url: str | None, events: list[SearchMonitorWebhookEvent] | None
) -> dict[str, Any]:
    """Build Exa's ``webhook`` object; no events means Exa sends all of them."""
    webhook: dict[str, Any] = {}
    if url:
        webhook["url"] = url.strip()
    if events:
        webhook["events"] = [event.value for event in events]
    return webhook


class SearchMonitorApiMixin(TeamScopedExaMixin):
    """Exa calls for Exa Search Monitors, which act on the whole Exa team."""

    TEAM_SCOPED_FEATURE = "Exa Search Monitors"
