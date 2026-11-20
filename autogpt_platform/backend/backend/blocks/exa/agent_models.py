"""
Models for the Exa Agent API (``/agent/runs``).

Holds the enums the Agent blocks offer, the typed output models they yield,
and the parsers that turn Exa's camelCase JSON into those models. Shared by the
Agent blocks and by the legacy Research blocks, which now run on Exa Agent.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.sdk import SchemaField

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

EFFORT_DESCRIPTION = (
    "How hard the agent works, which sets cost and speed. Fixed efforts have a "
    "flat price: minimal ($0.012, cheapest, very narrow lookups), low ($0.025, "
    "fast, simple lookups), medium ($0.10, balanced; the usual starting point), "
    "high ($0.50, slower, more thorough and better cited), xhigh ($1.00, the "
    "slowest fixed effort, for complex schemas and hard verification). Metered "
    "efforts bill actual usage up to a cap: auto (Exa picks the effort; up to "
    "$5 by default; good when the scope is unknown, such as list building) and "
    "ultra (the most exhaustive; up to $20 by default; usually ~30 minutes and "
    "up to 3 hours)."
)


class AgentEffort(str, Enum):
    """Cost and reasoning effort for an Exa Agent run."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    AUTO = "auto"
    ULTRA = "ultra"


METERED_EFFORTS = frozenset({AgentEffort.AUTO, AgentEffort.ULTRA})


class AgentDataSource(str, Enum):
    """Self-serve Exa Connect data providers listed in Exa's docs."""

    FIBER = "fiber"
    SIMILARWEB = "similarweb"
    BASELAYER = "baselayer"
    POLYMARKET = "polymarket"
    MACROBOND = "macrobond"
    AFFILIATE = "affiliate"
    PARTICLE = "particle"
    FINANCIAL_DATASETS = "financial_datasets"
    JINKO = "jinko"


DATA_SOURCES_DESCRIPTION = (
    "Exa Connect partner databases the agent may query alongside web search "
    "(at most 5, billed per call on top of the run): fiber (Fiber.ai, B2B "
    "companies and people), similarweb (web traffic and rankings), baselayer "
    "(US business verification / KYB), polymarket (prediction market odds), "
    "macrobond (macroeconomic time series), affiliate (Affiliate.com product "
    "catalogs), particle (podcast transcripts), financial_datasets (US stock "
    "prices, fundamentals and filings), jinko (flight and hotel search). "
    "Mention the source in your query or output schema (e.g. 'monthly visits "
    "from Similarweb') so the agent uses it."
)


class AgentCitation(BaseModel):
    """A source supporting one output field."""

    url: str = SchemaField(description="Source URL")
    title: str | None = SchemaField(default=None, description="Source title")


class AgentGrounding(BaseModel):
    """Citations for one field of the run's output."""

    field: str = SchemaField(description="Output field the citations support")
    citations: list[AgentCitation] = SchemaField(
        default_factory=list, description="Sources for this field"
    )
    confidence: str | None = SchemaField(
        default=None,
        description="Model-reported reliability for this field: low, medium or high",
    )


class AgentUsage(BaseModel):
    """Billed usage for a run. Quantities can be fractional when a budget capped the run."""

    agent_compute_units: float = SchemaField(
        default=0.0, description="Agent Compute Units consumed ($0.10 each)"
    )
    searches: float = SchemaField(
        default=0.0, description="Search tool calls made ($0.005 each)"
    )
    emails: float = SchemaField(
        default=0.0, description="Email contacts enriched ($0.02 each)"
    )
    phone_numbers: float = SchemaField(
        default=0.0, description="Phone numbers enriched ($0.07 each)"
    )
    data_sources: dict[str, int] = SchemaField(
        default_factory=dict,
        description="Exa Connect tool calls per provider, e.g. {'fiber': 3}",
    )


class AgentCost(BaseModel):
    """Cost breakdown for a run in US dollars."""

    total: float = SchemaField(default=0.0, description="Total cost in USD")
    agent_compute: float = SchemaField(default=0.0, description="Agent compute cost")
    search: float = SchemaField(default=0.0, description="Search tool call cost")
    emails: float = SchemaField(default=0.0, description="Email enrichment cost")
    phone_numbers: float = SchemaField(
        default=0.0, description="Phone number enrichment cost"
    )
    data_sources: dict[str, float] = SchemaField(
        default_factory=dict, description="Exa Connect cost per provider"
    )


class AgentRun(BaseModel):
    """An Exa Agent run as returned by the Agent blocks."""

    run_id: str = SchemaField(description="Run ID (starts with agent_run_)")
    status: str = SchemaField(
        description="queued, running, completed, failed or cancelled"
    )
    stop_reason: str | None = SchemaField(
        default=None,
        description="Why the run stopped, e.g. schema_satisfied, budget_reached, "
        "time_limit_reached, stopped, error, cancelled. Empty while running.",
    )
    query: str = SchemaField(default="", description="The query the run was given")
    effort: str | None = SchemaField(default=None, description="Effort used")
    output_schema: dict[str, Any] | None = SchemaField(
        default=None, description="The output schema the run was given, if any"
    )
    text: str = SchemaField(default="", description="Natural-language answer")
    structured: Any | None = SchemaField(
        default=None, description="JSON matching the output schema, if one was given"
    )
    grounding: list[AgentGrounding] = SchemaField(
        default_factory=list, description="Field-level citations"
    )
    usage: AgentUsage = SchemaField(
        default_factory=AgentUsage, description="Billed usage"
    )
    cost: AgentCost = SchemaField(
        default_factory=AgentCost, description="Cost breakdown in USD"
    )
    created_at: str = SchemaField(default="", description="Creation time (ISO 8601)")
    completed_at: str | None = SchemaField(
        default=None, description="Completion time (ISO 8601), if finished"
    )
    error: str | None = SchemaField(
        default=None, description="Why the run failed, when it failed"
    )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class AgentRunEvent(BaseModel):
    """One stored lifecycle event of a run."""

    id: str = SchemaField(description="Event ID within the run")
    event: str = SchemaField(
        description="Event name, e.g. agent_run.created, agent_run.completed"
    )
    data: Any = SchemaField(default=None, description="Event payload")
    created_at: str = SchemaField(default="", description="Event time (ISO 8601)")


class AgentPage(BaseModel):
    """Pagination state shared by the list endpoints."""

    has_more: bool = False
    next_cursor: str | None = None


def parse_run(data: dict) -> AgentRun:
    """Convert an Exa ``AgentRun`` JSON object into an :class:`AgentRun`."""
    request = data.get("request") or {}
    output = data.get("output") or {}
    return AgentRun(
        run_id=data.get("id", ""),
        status=data.get("status", ""),
        stop_reason=data.get("stopReason"),
        query=request.get("query", ""),
        effort=request.get("effort"),
        output_schema=request.get("outputSchema"),
        text=output.get("text") or "",
        structured=output.get("structured"),
        grounding=[_parse_grounding(g) for g in output.get("grounding") or []],
        usage=_parse_usage(data.get("usage") or {}),
        cost=_parse_cost(data.get("costDollars") or {}),
        created_at=data.get("createdAt", ""),
        completed_at=data.get("completedAt"),
        error=_parse_error(data.get("error")),
    )


def parse_event(data: dict) -> AgentRunEvent:
    """Convert an Exa ``AgentRunEvent`` JSON object into an :class:`AgentRunEvent`."""
    return AgentRunEvent(
        id=str(data.get("id", "")),
        event=data.get("event", ""),
        data=data.get("data"),
        created_at=data.get("createdAt", ""),
    )


def parse_page(data: dict) -> AgentPage:
    """Read ``hasMore`` / ``nextCursor`` from a list response."""
    return AgentPage(
        has_more=bool(data.get("hasMore", False)),
        next_cursor=data.get("nextCursor"),
    )


def failure_message_from_events(events: list[AgentRunEvent]) -> str | None:
    """Return the error message of the last ``agent_run.failed`` event, if any."""
    failed = [e for e in events if e.event == "agent_run.failed"]
    if not failed:
        return None
    try:
        payload = _FailedEventData.model_validate(failed[-1].data)
    except ValidationError:
        return None
    return payload.error.text() if payload.error else None


def _parse_grounding(data: dict) -> AgentGrounding:
    return AgentGrounding(
        field=data.get("field", ""),
        citations=[
            AgentCitation(url=c.get("url", ""), title=c.get("title"))
            for c in data.get("citations") or []
        ],
        confidence=data.get("confidence"),
    )


def _parse_usage(data: dict) -> AgentUsage:
    return AgentUsage(
        agent_compute_units=data.get("agentComputeUnits", 0.0),
        searches=data.get("searches", 0.0),
        emails=data.get("emails", 0.0),
        phone_numbers=data.get("phoneNumbers", 0.0),
        data_sources=data.get("dataSources") or {},
    )


def _parse_cost(data: dict) -> AgentCost:
    return AgentCost(
        total=data.get("total", 0.0),
        agent_compute=data.get("agentCompute", 0.0),
        search=data.get("search", 0.0),
        emails=data.get("emails", 0.0),
        phone_numbers=data.get("phoneNumbers", 0.0),
        data_sources=data.get("dataSources") or {},
    )


def _parse_error(raw: Any) -> str | None:
    """Exa reports errors as ``{code, message}``; tolerate other shapes."""
    if not raw:
        return None
    try:
        return _ApiError.model_validate(raw).text()
    except ValidationError:
        return str(raw)


class _ApiError(BaseModel):
    code: str | None = None
    message: str | None = None

    def text(self) -> str:
        if self.code and self.message:
            return f"{self.code}: {self.message}"
        return self.message or self.code or "Unknown error"


class _FailedEventData(BaseModel):
    error: _ApiError | None = None
