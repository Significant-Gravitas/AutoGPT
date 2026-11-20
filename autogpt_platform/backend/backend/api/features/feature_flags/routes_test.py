import asyncio
import json
import os
import uuid
from unittest.mock import AsyncMock

import fastapi
import httpx
import pytest
from autogpt_libs.auth import get_optional_user_id
from posthog import Posthog
from posthog.types import FeatureFlag

from backend.util.feature_flag import (
    Flag,
    _UserContext,
    frontend,
    get_feature_flag_value,
    is_feature_enabled,
)
from backend.util.feature_flag import posthog as ph
from backend.util.feature_flag.frontend import (
    FrontendFlag,
    _anonymous_flags,
    _defined_flags,
)

from .routes import router

SEARCH_TERMS = ["Research", "Marketing"]
VISITOR_TERMS = ["Sales"]
VISITOR = "0b6c1f3e-visitor-0001"
USER_ID = "3f2b8c1e-7a4d-4e9b-9c2a-5d6e7f8a9b0c"
PINNING = FrontendFlag.CHAT_PINNING.value
GENERIC_TRIGGER_AGENTS = FrontendFlag.GENERIC_TRIGGER_AGENTS.value
PROGRESS_BAR = FrontendFlag.TASK_PROGRESS_BAR.value
STREAM_SHADOW = FrontendFlag.COPILOT_STREAM_SHADOW.value

app = fastapi.FastAPI()
app.include_router(router, prefix="/api")


@pytest.fixture(autouse=True)
def posthog_flags(mocker, monkeypatch: pytest.MonkeyPatch):
    """A real PostHog client serving fixed targeting from local definitions,
    with no network and no env overrides."""
    for name in list(os.environ):
        if name.startswith(
            ("FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_", "FORCE_ALL_FLAGS")
        ):
            monkeypatch.delenv(name)
    mocker.patch.object(ph.settings.secrets, "posthog_api_key", "phc-test")
    client = _posthog_with_fixed_targeting()
    mocker.patch.object(client, "_get_flags_decision", side_effect=ConnectionError)
    mocker.patch.object(client, "capture")
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    _anonymous_flags.cache_clear()
    _defined_flags.cache_clear()
    yield client
    _anonymous_flags.cache_clear()
    _defined_flags.cache_clear()
    app.dependency_overrides.clear()
    client.shutdown()


@pytest.mark.parametrize("caller", ["u-staff", "u-other", None])
async def test_each_flag_answers_what_the_backends_own_read_answers(caller):
    body = await _get_flags(caller)

    as_backend = caller or "anonymous"
    assert body["configured"] is True
    assert body["flags"][
        FrontendFlag.COPILOT_VOICE_MODE.value
    ] is await is_feature_enabled(Flag.COPILOT_VOICE_MODE, as_backend)
    assert body["flags"][
        FrontendFlag.MARKETPLACE_SEARCH_TERMS.value
    ] == await get_feature_flag_value(
        FrontendFlag.MARKETPLACE_SEARCH_TERMS.value, as_backend
    )


@pytest.mark.parametrize("caller", [USER_ID, None])
async def test_retired_rollout_flags_are_not_served_or_restored_by_overrides(
    caller, monkeypatch, mocker
):
    mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        AsyncMock(
            return_value=(
                _UserContext(key=caller or "anonymous", anonymous=caller is None),
                True,
            )
        ),
    )
    retired = {
        "artifacts-page",
        "hire-experts",
        "skills-hub",
        "onboarding-brain-dump",
        "onboarding-expert-team",
        "graphiti-memory",
        "graphiti-communities-enabled",
        "expert-spend-approval",
        "ai-agent-execution-summary",
    }
    for key in retired:
        monkeypatch.setenv(f"FORCE_FLAG_{key.replace('-', '_').upper()}", "false")

    flags = (await _get_flags(caller))["flags"]

    assert retired.isdisjoint(flags)
    assert set(flags) == {flag.value for flag in FrontendFlag}


async def test_a_flag_the_vendor_has_no_definition_of_is_answered_null():
    flags = (await _get_flags("u-staff"))["flags"]

    # Null serves the page's default now; left out, the page would wait for good.
    assert (
        flags.get(PROGRESS_BAR, "left out"),
        flags.get(STREAM_SHADOW, "left out"),
        flags[GENERIC_TRIGGER_AGENTS],
    ) == (None, None, True)


async def test_a_defined_flag_the_vendor_could_not_answer_is_left_out(mocker):
    # A failed user lookup makes every answer a guess about someone else.
    mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        AsyncMock(return_value=(_UserContext(key=USER_ID), False)),
    )

    flags = (await _get_flags(USER_ID))["flags"]

    # A default here would read as an answer and 404 a gated route for a user who has it.
    assert (GENERIC_TRIGGER_AGENTS in flags, flags.get(PROGRESS_BAR, "left out")) == (
        False,
        None,
    )


@pytest.mark.parametrize("ready_first", [False, True])
async def test_a_signed_out_answer_is_kept_long_only_when_complete(
    mocker, monkeypatch, posthog_flags, ready_first
):
    """A vendor that is down or starting is asked again within seconds, while a
    complete answer keeps serving every signed-out visitor from one read."""
    monkeypatch.setattr(frontend, "PARTIAL_TTL_SECONDS", 0.01)
    not_ready = _posthog_still_loading(mocker)
    get = mocker.patch.object(
        ph,
        "get_flag_client",
        return_value=posthog_flags if ready_first else not_ready,
    )
    try:
        first = (await _get_flags(None))["flags"]
        get.return_value = not_ready if ready_first else posthog_flags
        await asyncio.sleep(0.05)
        second = (await _get_flags(None))["flags"]
    finally:
        not_ready.shutdown()

    # Either way the second read sees the first, complete answer.
    assert (GENERIC_TRIGGER_AGENTS in first, second.get(GENERIC_TRIGGER_AGENTS)) == (
        ready_first,
        True,
    )


async def test_a_vendor_still_loading_leaves_every_flag_out_and_says_it_will_answer(
    mocker,
):
    # Definitions not loaded and /flags unreachable, as while PostHog starts.
    client = _posthog_still_loading(mocker)
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    try:
        answer = await _get_flags("u-staff")
    finally:
        client.shutdown()

    # PostHog keeps retrying, so the page keeps asking rather than settling on defaults.
    assert answer == {"flags": {}, "configured": True}


async def test_a_signed_in_read_asks_posthog_nothing_about_flags_it_lacks(mocker):
    """Each would be a remote /flags call on the executor every flag read shares."""
    client = _posthog_splitting_pinning_and_gating_hire_on_email()
    remote = mocker.patch.object(
        client, "_get_flags_decision", side_effect=ConnectionError
    )
    mocker.patch.object(client, "capture")
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    try:
        flags = (await _get_flags("u-staff"))["flags"]
    finally:
        client.shutdown()

    # generic-trigger-agents is defined but needs /flags; nothing else may ask.
    assert (remote.call_count, flags.get(PROGRESS_BAR, "left out")) == (1, None)


@pytest.mark.parametrize("loaded", [True, False])
async def test_on_posthog_only_loaded_definitions_can_rule_a_flag_undefined(
    mocker, loaded
):
    client = _posthog_splitting_pinning_and_gating_hire_on_email()
    if not loaded:
        client.feature_flags = None
    mocker.patch.object(client, "load_feature_flags")
    mocker.patch.object(client, "_get_flags_decision", side_effect=ConnectionError)
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    try:
        flags = (await _get_flags("u-staff"))["flags"]
    finally:
        client.shutdown()

    # generic-trigger-agents is defined but needs /flags, which is down: left out either way.
    expected = (False, None) if loaded else (False, "left out")
    assert (
        GENERIC_TRIGGER_AGENTS in flags,
        flags.get(PROGRESS_BAR, "left out"),
    ) == expected


async def test_posthog_with_no_key_leaves_every_flag_to_the_frontend(mocker):
    mocker.patch.object(ph.settings.secrets, "posthog_api_key", "")
    # What the real client getter answers without a key.
    mocker.patch.object(ph, "get_flag_client", return_value=None)

    assert await _get_flags(USER_ID) == {"flags": {}, "configured": False}


@pytest.mark.parametrize(
    "user_id,visitor_id,terms",
    [("u-other", None, SEARCH_TERMS), (None, VISITOR, VISITOR_TERMS)],
)
async def test_an_env_override_forces_a_switch_but_never_a_payload(
    monkeypatch, user_id, visitor_id, terms
):
    monkeypatch.setenv("FORCE_FLAG_COPILOT_VOICE_MODE", "true")
    monkeypatch.setenv("FORCE_FLAG_MARKETPLACE_SEARCH_TERMS", "true")
    # Forced although the vendor has no definition of it.
    monkeypatch.setenv("FORCE_FLAG_TASK_PROGRESS_BAR", "true")

    flags = (await _get_flags(user_id, visitor_id))["flags"]

    assert flags[FrontendFlag.COPILOT_VOICE_MODE.value] is True
    assert flags[FrontendFlag.MARKETPLACE_SEARCH_TERMS.value] == terms
    assert flags[PROGRESS_BAR] is True


async def test_a_consenting_visitor_is_bucketed_by_their_own_id():
    own = (await _get_flags(None, VISITOR))["flags"]
    shared = (await _get_flags(None))["flags"]

    assert own[PINNING] is True
    assert shared[PINNING] is False


async def test_a_signed_in_caller_is_never_bucketed_by_a_visitor_id():
    flags = (await _get_flags("u-other", VISITOR))["flags"]

    assert flags[PINNING] is False


@pytest.mark.parametrize("malformed", ["v" * 65, "visitor:bad$id"])
async def test_a_malformed_visitor_id_counts_as_none(malformed):
    flags = (await _get_flags(None, malformed))["flags"]

    assert flags[PINNING] is False


async def test_a_users_id_sent_as_a_visitor_id_is_never_looked_up_as_that_user(
    mocker,
):
    # The lookup would hand back the user, whom the flag targets.
    user = _UserContext(key=USER_ID, anonymous=False)
    lookup = mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        AsyncMock(return_value=(user, True)),
    )

    flags = (await _get_flags(None, USER_ID))["flags"]

    assert flags[PINNING] is False
    assert USER_ID not in [call.args[0] for call in lookup.await_args_list]


async def test_a_users_id_sent_as_a_visitor_id_never_matches_that_users_targets(
    mocker,
):
    mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        AsyncMock(return_value=(_UserContext(key=USER_ID, anonymous=False), True)),
    )

    as_user = (await _get_flags(USER_ID))["flags"]
    as_visitor = (await _get_flags(None, USER_ID))["flags"]

    assert (as_user[PINNING], as_visitor[PINNING]) == (True, False)


async def test_fresh_visitor_ids_cost_posthog_nothing(mocker):
    """No billed event, no remote /flags call and no turn on the executor every
    flag read shares: the id is the caller's to pick, so a new one must be free."""
    client = _posthog_splitting_pinning_and_gating_hire_on_email()
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    remote = mocker.patch.object(
        client,
        "_get_flags_decision",
        return_value={
            "flags": {
                GENERIC_TRIGGER_AGENTS: FeatureFlag.from_value_and_payload(
                    GENERIC_TRIGGER_AGENTS, True, None
                )
            }
        },
    )
    events = mocker.patch.object(client, "capture")
    executor = mocker.patch.object(
        ph._read_executor, "submit", wraps=ph._read_executor.submit
    )
    try:
        await _get_flags(None)
        for spy in (remote, events, executor):
            spy.reset_mock()
        visitors = [
            (await _get_flags(None, str(uuid.uuid4())))["flags"] for _ in range(40)
        ]
    finally:
        client.shutdown()

    assert (remote.call_count, events.call_count, executor.call_count) == (0, 0, 0)
    # Each is still bucketed by their own id where the definitions decide it, and
    # takes the shared signed-out answer where only PostHog's servers could.
    assert {v.get(PINNING) for v in visitors} == {True, False}
    assert all(v.get(GENERIC_TRIGGER_AGENTS) is True for v in visitors)


def _posthog_with_fixed_targeting() -> Posthog:
    """Local definitions every read decides in-process, without /flags."""
    client = Posthog("phc-test", secret_key="phx-test", enable_local_evaluation=False)
    client.feature_flags = [
        _targeted(1, FrontendFlag.COPILOT_VOICE_MODE.value, ["u-staff"]),
        {
            "id": 2,
            "key": GENERIC_TRIGGER_AGENTS,
            "active": True,
            "filters": {"groups": [{"properties": [], "rollout_percentage": 100}]},
        },
        {
            "id": 3,
            "key": FrontendFlag.MARKETPLACE_SEARCH_TERMS.value,
            "active": True,
            "filters": {
                "groups": [
                    {
                        "properties": [_distinct_id_in([f"visitor:{VISITOR}"])],
                        "rollout_percentage": 100,
                        "variant": "visitor",
                    },
                    {"properties": [], "rollout_percentage": 100, "variant": "all"},
                ],
                "multivariate": {
                    "variants": [
                        {"key": "all", "rollout_percentage": 100},
                        {"key": "visitor", "rollout_percentage": 0},
                    ]
                },
                "payloads": {
                    "all": json.dumps(SEARCH_TERMS),
                    "visitor": json.dumps(VISITOR_TERMS),
                },
            },
        },
        _targeted(
            4,
            PINNING,
            [
                f"visitor:{VISITOR}",
                f"visitor:{'v' * 65}",
                "visitor:visitor:bad$id",
                USER_ID,
            ],
        ),
    ]
    return client


def _targeted(flag_id: int, key: str, distinct_ids: list[str]) -> dict:
    """A boolean flag on for exactly these distinct ids, off for everyone else."""
    return {
        "id": flag_id,
        "key": key,
        "active": True,
        "filters": {
            "groups": [
                {
                    "properties": [_distinct_id_in(distinct_ids)],
                    "rollout_percentage": 100,
                }
            ]
        },
    }


def _distinct_id_in(distinct_ids: list[str]) -> dict:
    return {
        "key": "distinct_id",
        "type": "person",
        "operator": "exact",
        "value": distinct_ids,
    }


def _posthog_still_loading(mocker) -> Posthog:
    """No definitions yet and /flags unreachable: every read goes unanswered."""
    client = Posthog("phc-test", secret_key="phx-test", enable_local_evaluation=False)
    mocker.patch.object(client, "load_feature_flags")
    mocker.patch.object(client, "_get_flags_decision", side_effect=ConnectionError)
    mocker.patch.object(client, "capture")
    return client


def _posthog_splitting_pinning_and_gating_hire_on_email() -> Posthog:
    """Local definitions: pinning is a 50% split, generic-trigger-agents needs an email."""
    client = Posthog("phc-test", secret_key="phx-test", enable_local_evaluation=False)
    client.feature_flags = [
        {
            "id": 1,
            "key": PINNING,
            "active": True,
            "filters": {"groups": [{"properties": [], "rollout_percentage": 50}]},
        },
        {
            "id": 2,
            "key": GENERIC_TRIGGER_AGENTS,
            "active": True,
            "filters": {
                "groups": [
                    {
                        "properties": [
                            {
                                "key": "email",
                                "type": "person",
                                "operator": "regex",
                                "value": r"@agpt\.co$",
                            }
                        ],
                        "rollout_percentage": 100,
                    }
                ]
            },
        },
    ]
    return client


async def _get_flags(user_id: str | None, visitor_id: str | None = None) -> dict:
    app.dependency_overrides[get_optional_user_id] = lambda: user_id
    params = {"visitor_id": visitor_id} if visitor_id else {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/feature-flags", params=params)
    assert response.status_code == 200
    return response.json()
