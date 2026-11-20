"""Canned Treg answers for the block self-tests and unit tests."""

from typing import Any

from ._api import TregResponse

TEST_CALL_ID = "call_01JTREGTEST0000000000000"
TEST_COST_USD = 0.0245
TEST_PROVIDERS = ["hunter"]


def mock_routed(
    output: dict[str, Any],
    *,
    served_by: str | None = "hunter.email.finder",
    raw: Any = None,
    cost_usd: float = TEST_COST_USD,
    tried: list[str] | None = None,
    advice: str | None = None,
    verification: dict[str, Any] | None = None,
    outcome: str = "hit",
) -> TregResponse:
    """A 200 answer from a routed tool, shaped like treg.to's."""
    providers = TEST_PROVIDERS if tried is None else tried
    meta: dict[str, Any] = {
        "served_by": served_by,
        "outcome": outcome,
        "tried": [
            {"endpoint_id": f"{p}.lookup", "provider": p, "outcome": outcome}
            for p in providers
        ],
        "charged_micro": round(cost_usd * 1_000_000),
    }
    if advice:
        meta["advice"] = advice
    if verification is not None:
        meta["verification"] = verification
    return TregResponse(
        status=200,
        body={
            "output": output,
            "raw": {"data": output} if raw is None and served_by else raw,
            "_treg": meta,
        },
        cost_usd=cost_usd,
        call_id=TEST_CALL_ID,
        served_by=served_by,
        async_task=None,
        cached=False,
        upstream_failed=False,
    )


def mock_miss(fields: list[str], cost_usd: float = 0.0) -> TregResponse:
    """A routed miss: every provider answered, none had it."""
    return mock_routed(
        {field: None for field in fields},
        served_by=None,
        raw=None,
        cost_usd=cost_usd,
        outcome="miss",
    )


def mock_response(
    body: Any,
    *,
    status: int = 200,
    cost_usd: float = 0.0,
    call_id: str | None = TEST_CALL_ID,
    served_by: str | None = None,
    async_task: dict[str, Any] | None = None,
) -> TregResponse:
    return TregResponse(
        status=status,
        body=body,
        cost_usd=cost_usd,
        call_id=call_id,
        served_by=served_by,
        async_task=async_task,
        cached=False,
        upstream_failed=False,
    )


def routed_details(
    found: bool = True,
    served_by: str = "hunter.email.finder",
    cost_usd: float = TEST_COST_USD,
    advice: str | None = None,
) -> list[tuple[str, Any]]:
    """The shared outputs a routed block emits after its own fields."""
    details: list[tuple[str, Any]] = [("found", found)]
    if found:
        details.append(("served_by", served_by))
    details += [
        ("providers_tried", TEST_PROVIDERS),
        ("cost_usd", cost_usd),
        ("call_id", TEST_CALL_ID),
    ]
    if advice:
        details.append(("advice", advice))
    if found:
        details.append(("raw", dict))
    return details
