"""What the sync writes, PostHog serves exactly as LaunchDarkly does.

Each vendor's real evaluator answers through the backend's dual comparison:
LaunchDarkly from a synthetic export shaped like Dev's, PostHog from the
definitions the sync maps that export to, evaluated in-process with no remote
fallback.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from ldclient import LDClient
from ldclient.config import Config as LDConfig
from ldclient.integrations import Files
from posthog import Posthog

import backend.data.db_accessors as db_accessors
import backend.util.feature_flag as ff
import backend.util.feature_flag.posthog as ph
from backend.data.user import AuthUserFlagFields
from backend.util.settings import FeatureFlagBackend
from scripts.feature_flag_sync import map_flag, map_segment, resolve_cohort_refs

ENV = "test"
TARGETED = str(uuid.uuid4())
PEOPLE = {
    "employee": AuthUserFlagFields(
        role="admin",
        email="ada@agpt.co",
        created_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
    ),
    "outsider": AuthUserFlagFields(
        role="authenticated",
        email="eve@elsewhere.test",
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    ),
    "targeted": AuthUserFlagFields(role="authenticated", email="bob@elsewhere.test"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("who", [*PEOPLE, "anonymous"])
async def test_every_ported_flag_serves_launchdarklys_answer(
    who, tmp_path, monkeypatch, caplog
):
    ld = _launchdarkly(tmp_path)
    posthog_client = _posthog()
    monkeypatch.setattr(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.DUAL
    )
    monkeypatch.setattr(ff, "get_client", lambda: ld)
    monkeypatch.setattr(ph, "get_flag_client", lambda: posthog_client)
    monkeypatch.setattr(db_accessors, "user_db", _AuthTable)
    ff._unanswered_flags_reported.clear()
    posthog_reads = _record_posthog_reads(monkeypatch)
    user_id = "system" if who == "anonymous" else USER_IDS[who]
    # A flag targeting a person cannot be decided in-process without one, so
    # PostHog would answer it remotely; the backend reads only these anonymously.
    keys = ANONYMOUS_READS if who == "anonymous" else [f["key"] for f in EXPORT]

    served: dict[str, Any] = {}
    with caplog.at_level(logging.WARNING, logger=ff.mismatch_logger.name):
        try:
            for key in keys:
                served[key] = await ff.get_feature_flag_value(key, user_id, None)
            while ff._shadow_evaluations:
                await asyncio.gather(*list(ff._shadow_evaluations))
        finally:
            ld.close()
            posthog_client.shutdown()

    assert served == {key: ANSWERS[key].get(who, ANSWERS[key]["*"]) for key in keys}
    assert posthog_reads == {key: (value, True) for key, value in served.items()}
    assert not [
        r.getMessage() for r in caplog.records if r.name == ff.mismatch_logger.name
    ]


def _segment(key: str, rules: list[list[dict[str, Any]]], **members: Any):
    return {
        "key": key,
        "name": key.title(),
        "rules": [{"clauses": clauses} for clauses in rules],
        **members,
    }


def _clause(attribute: str, op: str, *values: Any) -> dict[str, Any]:
    # Without a context kind LaunchDarkly reads "/custom/role" as a literal name.
    return {
        "attribute": attribute,
        "op": op,
        "values": list(values),
        "negate": False,
        "contextKind": "user",
    }


def _in_segment(*keys: str) -> dict[str, Any]:
    return _clause("segmentMatch", "segmentMatch", *keys)


def _flag(
    key: str,
    variations: list[Any],
    *,
    on: bool = True,
    off: int = 1,
    targets: list[dict[str, Any]] | None = None,
    rules: list[tuple[list[dict[str, Any]], int]] | None = None,
    fallthrough: int = 1,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """A flag as LaunchDarkly's REST API exports it."""
    return {
        "key": key,
        "name": key,
        "kind": "boolean" if variations == [True, False] else "multivariate",
        "variations": [
            {"value": v, **({"name": names[i]} if names else {})}
            for i, v in enumerate(variations)
        ],
        "environments": {
            ENV: {
                "on": on,
                "offVariation": off,
                "targets": targets or [],
                "contextTargets": [],
                "rules": [
                    {"clauses": clauses, "variation": variation}
                    for clauses, variation in rules or []
                ],
                "fallthrough": {"variation": fallthrough},
                "prerequisites": [],
            }
        },
    }


USER_IDS = {"employee": str(uuid.uuid4()), "outsider": str(uuid.uuid4())}
USER_IDS["targeted"] = TARGETED

SEGMENTS = [
    _segment(
        "employee", [[_clause("email", "endsWith", "@agpt.co")]], included=[TARGETED]
    ),
    _segment("admin", [[_clause("/custom/role", "in", "admin")]]),
]

LIMITS = {"daily": 625000, "weekly": 3125000}
MULTIPLIERS = {"MAX": 42.66, "PRO": 5}
EXPORT = [
    _flag("AutoMod", [True, False], on=False),
    _flag("artifacts-page", [True, False], fallthrough=0),
    _flag("copilot-voice-mode", [True, False], rules=[([_in_segment("employee")], 0)]),
    _flag(
        "chat-mode-option",
        [True, False],
        rules=[
            ([_clause("email", "in", "eve@elsewhere.test")], 0),
            ([_in_segment("admin")], 0),
        ],
    ),
    _flag(
        "ai-agent-execution-summary",
        [True, False],
        targets=[{"values": [TARGETED], "variation": 0}],
        rules=[([_clause("email_domain", "in", "agpt.co")], 0)],
    ),
    _flag("copilot-cost-limits", [LIMITS, {}], fallthrough=0),
    _flag(
        "copilot-tier-multipliers",
        [MULTIPLIERS, {}],
        rules=[([_in_segment("employee")], 1)],
        fallthrough=0,
    ),
    _flag("copilot-model-routing", [{"fast": {"standard": "a/b"}}, {}], on=False),
    _flag(
        "copilot-tier-stripe-prices",
        [{"PRO": "price_dev"}, {"PRO": "price_prod"}],
        names=["dev", "prod"],
        fallthrough=0,
    ),
    _flag("stripe-product-id-topup", ["", "prod_topup"], on=False),
]
ANONYMOUS_READS = [
    "AutoMod",
    "artifacts-page",
    "copilot-cost-limits",
    "copilot-tier-multipliers",
    "copilot-model-routing",
    "copilot-tier-stripe-prices",
    "stripe-product-id-topup",
]
# What LaunchDarkly serves each reader, so a fixture that served everyone the
# same could not pass for a port that ignores targeting.
ANSWERS: dict[str, dict[str, Any]] = {
    "AutoMod": {"*": False},
    "artifacts-page": {"*": True},
    "copilot-voice-mode": {"*": False, "employee": True, "targeted": True},
    "chat-mode-option": {"*": False, "employee": True, "outsider": True},
    "ai-agent-execution-summary": {"*": False, "employee": True, "targeted": True},
    "copilot-cost-limits": {"*": LIMITS},
    "copilot-tier-multipliers": {"*": MULTIPLIERS, "employee": {}, "targeted": {}},
    "copilot-model-routing": {"*": {}},
    "copilot-tier-stripe-prices": {"*": {"PRO": "price_dev"}},
    "stripe-product-id-topup": {"*": "prod_topup"},
}


def _launchdarkly(tmp_path) -> LDClient:
    """LaunchDarkly's own evaluator over the export, in its SDK data format."""
    path = tmp_path / "launchdarkly.json"
    path.write_text(
        json.dumps(
            {
                "flags": {f["key"]: _sdk_flag(f) for f in EXPORT},
                "segments": {s["key"]: _sdk_segment(s) for s in SEGMENTS},
            }
        )
    )
    client = LDClient(
        LDConfig(
            "sdk-test",
            update_processor_class=Files.new_data_source(paths=[str(path)]),
            send_events=False,
        )
    )
    assert client.is_initialized()
    return client


def _sdk_flag(flag: dict[str, Any]) -> dict[str, Any]:
    cfg = flag["environments"][ENV]
    return {
        **{k: cfg[k] for k in ("on", "offVariation", "targets", "contextTargets")},
        "key": flag["key"],
        "version": 1,
        "salt": flag["key"],
        "variations": [v["value"] for v in flag["variations"]],
        "rules": [{"id": f"rule-{i}", **r} for i, r in enumerate(cfg["rules"])],
        "fallthrough": cfg["fallthrough"],
        "prerequisites": [],
    }


def _sdk_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": segment["key"],
        "version": 1,
        "salt": segment["key"],
        "included": segment.get("included", []),
        "excluded": [],
        "rules": [{"id": f"rule-{i}", **r} for i, r in enumerate(segment["rules"])],
    }


def _posthog() -> Posthog:
    """PostHog's in-process evaluator over what the sync would write."""
    cohorts = {s["key"]: map_segment(s, ENV) for s in SEGMENTS}
    payloads = [c.payload for c in cohorts.values() if c.payload is not None]
    ids = {p["name"]: i for i, p in enumerate(payloads, start=1)}
    flags = []
    for i, exported in enumerate(EXPORT, start=1):
        mapped = map_flag(exported, ENV, cohorts)
        assert mapped.payload is not None, mapped.decisions
        filters = resolve_cohort_refs(mapped.payload["filters"], ids)
        flags.append({**mapped.payload, "id": i, "filters": filters})

    client = Posthog("phc-test", secret_key="phx-test", enable_local_evaluation=False)
    client.feature_flags = flags
    client.cohorts = {str(ids[p["name"]]): p["filters"]["properties"] for p in payloads}
    client._get_flags_decision = _no_remote_flags
    return client


def _no_remote_flags(*args, **kwargs):
    raise ConnectionError("only an in-process answer counts here")


def _record_posthog_reads(monkeypatch) -> dict[str, tuple[Any, bool]]:
    reads: dict[str, tuple[Any, bool]] = {}
    evaluate = ph.evaluate_flag

    async def recording(flag_key: str, *args: Any, **kwargs: Any):
        reads[flag_key] = await evaluate(flag_key, *args, **kwargs)
        return reads[flag_key]

    monkeypatch.setattr(ph, "evaluate_flag", recording)
    return reads


class _AuthTable:
    """The auth rows the backend builds both vendors' user context from."""

    async def get_auth_user_flag_fields(self, user_id: str):
        return {USER_IDS[who]: fields for who, fields in PEOPLE.items()}[user_id]
