"""
HTTP client for the Exa Agent API (``https://api.exa.ai/agent/runs``).

One place for every Agent request, the create-run payload, its validation and
the polling loop, so the Agent blocks and the legacy Research blocks share them.
"""

from asyncio import sleep
from time import monotonic
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from backend.sdk import APIKeyCredentials

from ._exa_rest import EXA_API_BASE, ExaApiError, QueryParams, add_param, exa_request
from .agent_models import (
    METERED_EFFORTS,
    AgentDataSource,
    AgentEffort,
    AgentPage,
    AgentRun,
    AgentRunEvent,
    failure_message_from_events,
    parse_event,
    parse_page,
    parse_run,
)

AGENT_RUNS_URL = f"{EXA_API_BASE}/agent/runs"
MAX_DATA_SOURCES = 5
EVENT_PAGE_LIMIT = 100


class AgentRunRequest(BaseModel):
    """Everything a caller can set on ``POST /agent/runs``."""

    query: str
    effort: AgentEffort = AgentEffort.AUTO
    system_prompt: str | None = None
    input_data: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    output_schema: dict[str, Any] | None = None
    previous_run_id: str | None = None
    metadata: dict[str, str] = {}
    data_sources: list[AgentDataSource] = []
    custom_data_sources: list[str] = []
    max_cost_dollars: float | None = None
    max_duration_seconds: int | None = None
    user_identifier: str | None = None


class AgentWaitResult(BaseModel):
    """Outcome of polling a run: its last state and whether time ran out."""

    run: AgentRun
    timed_out: bool
    elapsed_seconds: float


def build_create_payload(request: AgentRunRequest) -> dict[str, Any]:
    """Validate ``request`` and build the camelCase JSON body Exa expects."""
    validate_run_request(request)
    payload: dict[str, Any] = {
        "query": request.query,
        "effort": request.effort.value,
    }
    optional = {
        "systemPrompt": request.system_prompt,
        "input": _input_block(request),
        "outputSchema": request.output_schema,
        "previousRunId": request.previous_run_id,
        "metadata": request.metadata,
        "dataSources": _data_sources(request),
        "budget": _budget(request),
        "userIdentifier": request.user_identifier,
    }
    payload.update({key: value for key, value in optional.items() if value})
    return payload


def validate_run_request(request: AgentRunRequest) -> None:
    """Reject effort/budget/data-source combinations Exa would refuse."""
    if not request.query.strip():
        raise ValueError("Query must not be empty.")
    if request.max_cost_dollars is not None and request.effort not in METERED_EFFORTS:
        raise ValueError(
            "Max cost only applies to the 'auto' and 'ultra' efforts; "
            f"'{request.effort.value}' has a fixed price. Clear the max cost "
            "or choose auto/ultra."
        )
    if request.max_duration_seconds is not None and request.effort != AgentEffort.ULTRA:
        raise ValueError(
            "Max duration only applies to the 'ultra' effort. Clear the max "
            "duration or choose ultra."
        )
    if len(_data_sources(request)) > MAX_DATA_SOURCES:
        raise ValueError(f"Exa allows at most {MAX_DATA_SOURCES} data sources per run.")


async def create_agent_run(
    credentials: APIKeyCredentials, payload: dict[str, Any]
) -> dict:
    return await exa_request(credentials, "POST", AGENT_RUNS_URL, body=payload)


async def get_agent_run(credentials: APIKeyCredentials, run_id: str) -> dict:
    return await exa_request(credentials, "GET", _run_url(run_id))


async def list_agent_runs(
    credentials: APIKeyCredentials, limit: int, cursor: str | None
) -> dict:
    return await exa_request(
        credentials, "GET", AGENT_RUNS_URL, params=_page_params(limit, cursor)
    )


async def cancel_agent_run(credentials: APIKeyCredentials, run_id: str) -> dict:
    return await exa_request(credentials, "POST", f"{_run_url(run_id)}/cancel")


async def stop_agent_run(credentials: APIKeyCredentials, run_id: str) -> dict:
    return await exa_request(credentials, "POST", f"{_run_url(run_id)}/stop")


async def delete_agent_run(credentials: APIKeyCredentials, run_id: str) -> dict:
    return await exa_request(credentials, "DELETE", _run_url(run_id))


async def list_agent_run_events(
    credentials: APIKeyCredentials, run_id: str, limit: int, cursor: str | None
) -> dict:
    return await exa_request(
        credentials,
        "GET",
        f"{_run_url(run_id)}/events",
        params=_page_params(limit, cursor),
    )


async def list_all_agent_run_events(
    credentials: APIKeyCredentials, run_id: str, max_pages: int = 10
) -> list[AgentRunEvent]:
    """Follow the events cursor (up to ``max_pages`` pages of 100)."""
    events: list[AgentRunEvent] = []
    cursor: str | None = None
    for _ in range(max_pages):
        data = await list_agent_run_events(
            credentials, run_id, EVENT_PAGE_LIMIT, cursor
        )
        events.extend(parse_event(e) for e in data.get("data") or [])
        page: AgentPage = parse_page(data)
        if not page.has_more or not page.next_cursor:
            break
        cursor = page.next_cursor
    return events


async def wait_for_agent_run(
    credentials: APIKeyCredentials,
    run_id: str,
    timeout_seconds: float,
    interval_seconds: float = 5.0,
    max_interval_seconds: float = 30.0,
) -> AgentWaitResult:
    """Poll a run until it finishes or ``timeout_seconds`` pass.

    The interval grows by half each poll, up to ``max_interval_seconds``, so
    short runs return quickly and hours-long ``ultra`` runs aren't hammered.
    """
    start = monotonic()
    interval = interval_seconds
    while True:
        run = parse_run(await get_agent_run(credentials, run_id))
        elapsed = monotonic() - start
        if run.is_terminal:
            return AgentWaitResult(run=run, timed_out=False, elapsed_seconds=elapsed)
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            return AgentWaitResult(run=run, timed_out=True, elapsed_seconds=elapsed)
        await sleep(min(interval, remaining))
        interval = min(interval * 1.5, max_interval_seconds)


async def describe_failure(credentials: APIKeyCredentials, run: AgentRun) -> str:
    """Best explanation of why ``run`` failed, reading its events if needed.

    The run object has no error field; Exa puts ``{code, message}`` on the
    ``agent_run.failed`` event instead.
    """
    if run.error:
        return f"Exa Agent run {run.run_id} failed: {run.error}"
    try:
        events = await list_all_agent_run_events(credentials, run.run_id, max_pages=3)
    except ExaApiError:
        # Zero Data Retention teams have no stored events to read.
        events = []
    message = failure_message_from_events(events)
    detail = message or f"stop reason '{run.stop_reason or 'unknown'}'"
    return f"Exa Agent run {run.run_id} failed: {detail}"


def _input_block(request: AgentRunRequest) -> dict[str, Any]:
    block: dict[str, Any] = {}
    if request.input_data:
        block["data"] = request.input_data
    if request.exclusions:
        block["exclusion"] = request.exclusions
    return block


def _data_sources(request: AgentRunRequest) -> list[dict[str, str]]:
    providers = [s.value for s in request.data_sources] + [
        p.strip() for p in request.custom_data_sources if p.strip()
    ]
    unique = list(dict.fromkeys(providers))
    return [{"provider": provider} for provider in unique]


def _budget(request: AgentRunRequest) -> dict[str, Any]:
    budget: dict[str, Any] = {}
    if request.max_cost_dollars is not None:
        budget["maxCostDollars"] = request.max_cost_dollars
    if request.max_duration_seconds is not None:
        budget["maxDurationSeconds"] = request.max_duration_seconds
    return budget


def _run_url(run_id: str) -> str:
    run_id = run_id.strip()
    if not run_id:
        raise ValueError("Run ID must not be empty.")
    return f"{AGENT_RUNS_URL}/{quote(run_id, safe='')}"


def _page_params(limit: int, cursor: str | None) -> QueryParams:
    params: QueryParams = []
    add_param(params, "limit", limit)
    add_param(params, "cursor", cursor)
    return params
