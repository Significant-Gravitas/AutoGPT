from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field

from backend.sdk import APIKeyCredentials, BaseModel, Requests

BROWSER_USE_API_URL = "https://api.browser-use.com"
RUNS_URL = f"{BROWSER_USE_API_URL}/api/v4/runs"
SESSIONS_URL = f"{BROWSER_USE_API_URL}/api/v4/sessions"


class BrowserUseRunStatus(str, Enum):
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES = frozenset(
    {
        BrowserUseRunStatus.COMPLETED,
        BrowserUseRunStatus.FAILED,
        BrowserUseRunStatus.CANCELLED,
    }
)


class BrowserUseRun(BaseModel):
    """The fields of a v4 RunSummary (GET /api/v4/runs/{id}) the block reads."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    status: BrowserUseRunStatus
    session_id: str = Field(validation_alias="sessionId")
    result: str | None = None
    output: Any = None
    error: str | None = None
    # The API returns a decimal string such as "0.0123".
    total_cost_usd: float = Field(default=0.0, validation_alias="totalCostUsd")


class BrowserUseSessionCost(BaseModel):
    """Browser time and proxy traffic from GET /api/v4/sessions/{id}/cost.

    A run's totalCostUsd covers only its LLM and web-search charges; the
    session's browser meters time and proxy bandwidth separately.
    """

    model_config = ConfigDict(populate_by_name=True)

    browser_cost_usd: float = Field(default=0.0, validation_alias="browserCostUsd")
    proxy_cost_usd: float = Field(default=0.0, validation_alias="proxyCostUsd")


class BrowserUseClient:
    """Thin async REST client for the Browser Use Cloud v4 runs API.

    Auth is the X-Browser-Use-API-Key header (no Bearer prefix). Non-2xx
    responses raise through ``Requests`` with the response body attached.
    """

    def __init__(self, credentials: APIKeyCredentials):
        self.requests = Requests(
            trusted_origins=[BROWSER_USE_API_URL],
            extra_headers={
                "X-Browser-Use-API-Key": credentials.api_key.get_secret_value()
            },
        )

    async def create_run(self, payload: dict[str, Any]) -> str:
        """POST /api/v4/runs and return the new run's id."""
        response = await self.requests.post(RUNS_URL, json=payload)
        return response.json()["id"]

    async def get_status(self, run_id: str) -> BrowserUseRunStatus:
        """GET /api/v4/runs/{id}/status, the cheap poll target."""
        response = await self.requests.get(f"{RUNS_URL}/{run_id}/status")
        return BrowserUseRunStatus(response.json()["status"])

    async def get_run(self, run_id: str) -> BrowserUseRun:
        """GET /api/v4/runs/{id}, read once the run is terminal."""
        response = await self.requests.get(f"{RUNS_URL}/{run_id}")
        return BrowserUseRun.model_validate(response.json())

    async def get_session_cost(self, session_id: str) -> BrowserUseSessionCost:
        """GET /api/v4/sessions/{id}/cost: what the session's browser used."""
        response = await self.requests.get(f"{SESSIONS_URL}/{session_id}/cost")
        return BrowserUseSessionCost.model_validate(response.json())

    async def cancel_run(self, run_id: str) -> None:
        """POST /api/v4/runs/{id}/cancel; idempotent for terminal runs."""
        await self.requests.post(f"{RUNS_URL}/{run_id}/cancel")


def build_run_payload(
    task: str,
    model: str,
    output_schema: dict[str, Any] | None,
    proxy_country_code: str,
    profile_id: str,
    max_cost_usd: float | None,
) -> dict[str, Any]:
    """Map block inputs to a RunCreateRequest, omitting unset optionals."""
    browser_settings: dict[str, Any] = {
        "proxyCountryCode": proxy_country_code.strip().lower() or None
    }
    if profile_id.strip():
        browser_settings["profileId"] = profile_id.strip()
    payload: dict[str, Any] = {"task": task, "browserSettings": browser_settings}
    if model.strip():
        payload["model"] = model.strip()
    if output_schema:
        payload["outputSchema"] = output_schema
    if max_cost_usd:
        payload["maxCostUsd"] = max_cost_usd
    return payload
