"""
Legacy Research API shapes, rebuilt from Exa Agent runs.

Exa retired ``/research/v1`` (it now answers 410 RESEARCH_RETIRED). The four
Research blocks keep their inputs and outputs for saved workflows and run on
``/agent/runs``; this module maps between the two vocabularies.

Gaps: Exa Agent reports no page count and no reasoning-token count, so
``num_pages`` / ``reasoning_tokens`` (and the Get block's ``cost_pages`` /
``cost_reasoning_tokens``) are always 0.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from .agent_models import AgentEffort, AgentRun


class ResearchModel(str, Enum):
    """Legacy research models; each now selects an Exa Agent effort."""

    FAST = "exa-research-fast"
    STANDARD = "exa-research"
    PRO = "exa-research-pro"


MODEL_TO_EFFORT: dict[ResearchModel, AgentEffort] = {
    ResearchModel.FAST: AgentEffort.LOW,
    ResearchModel.STANDARD: AgentEffort.MEDIUM,
    ResearchModel.PRO: AgentEffort.HIGH,
}

_EFFORT_TO_MODEL = {
    effort.value: model.value for model, effort in MODEL_TO_EFFORT.items()
}

# Exa Agent status -> legacy research status.
_STATUS_TO_LEGACY = {
    "queued": "pending",
    "running": "running",
    "completed": "completed",
    "cancelled": "canceled",
    "failed": "failed",
}

_DICT_ADAPTER = TypeAdapter(dict[str, Any])


class ResearchCostModel(BaseModel):
    """Cost breakdown for a research request."""

    total: float
    num_searches: int
    num_pages: int
    reasoning_tokens: int


class ResearchOutputModel(BaseModel):
    """Research output with content and optional structured data."""

    content: str
    parsed: dict[str, Any] | None = None


class ResearchTaskModel(BaseModel):
    """Stable output model for research tasks."""

    research_id: str
    created_at: int
    model: str
    instructions: str
    status: str
    output_schema: dict[str, Any] | None = None
    output: ResearchOutputModel | None = None
    cost_dollars: ResearchCostModel | None = None
    finished_at: int | None = None
    error: str | None = None

    @classmethod
    def from_agent_run(cls, run: AgentRun) -> "ResearchTaskModel":
        """Describe an Exa Agent run in the legacy research shape."""
        return cls(
            research_id=run.run_id,
            created_at=iso_to_ms(run.created_at) or 0,
            model=legacy_model(run.effort),
            instructions=run.query,
            status=legacy_status(run.status),
            output_schema=run.output_schema,
            output=legacy_output(run),
            cost_dollars=legacy_cost(run),
            finished_at=iso_to_ms(run.completed_at),
            error=run.error,
        )


def legacy_status(status: str) -> str:
    """Map an Exa Agent status onto pending/running/completed/canceled/failed."""
    return _STATUS_TO_LEGACY.get(status, status)


def legacy_model(effort: str | None) -> str:
    """Name the research model for an effort; other efforts are returned as-is."""
    if not effort:
        return ResearchModel.STANDARD.value
    return _EFFORT_TO_MODEL.get(effort, effort)


def legacy_output(run: AgentRun) -> ResearchOutputModel | None:
    """Research output, present once the run has completed."""
    if run.status != "completed":
        return None
    return ResearchOutputModel(content=run.text, parsed=as_dict(run.structured))


def legacy_cost(run: AgentRun) -> ResearchCostModel | None:
    """Research cost, present once the run has finished and settled."""
    if not run.is_terminal:
        return None
    return ResearchCostModel(
        total=run.cost.total,
        num_searches=round(run.usage.searches),
        num_pages=0,
        reasoning_tokens=0,
    )


def iso_to_ms(value: str | None) -> int | None:
    """ISO 8601 timestamp -> Unix milliseconds, as the Research API returned."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def as_dict(value: Any) -> dict[str, Any] | None:
    """Return ``value`` when it is a JSON object; legacy outputs only held objects."""
    if value is None:
        return None
    try:
        return _DICT_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def legacy_finished_outputs(run: AgentRun) -> list[tuple[str, Any]]:
    """The output_content / output_parsed / cost_total outputs of a finished run."""
    outputs: list[tuple[str, Any]] = []
    output = legacy_output(run)
    if output:
        outputs += [
            ("output_content", output.content),
            ("output_parsed", output.parsed),
        ]
    cost = legacy_cost(run)
    if cost:
        outputs.append(("cost_total", cost.total))
    return outputs
