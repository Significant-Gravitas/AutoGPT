import asyncio
import os
import uuid
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import fastapi
import httpx
import pytest
from autogpt_libs.auth import get_optional_user_id
from ldclient import Context, LDClient
from ldclient.config import Config as LDConfig
from ldclient.integrations.test_data import TestData
from ldclient.interfaces import DataSourceState, EventProcessor, UpdateProcessor
from posthog import Posthog
from posthog.types import FeatureFlag

import backend.util.feature_flag as ff
from backend.util.feature_flag import (
    Flag,
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
from backend.util.settings import FeatureFlagBackend

from .routes import router

SEARCH_TERMS = ["Research", "Marketing"]
VISITOR_TERMS = ["Sales"]
VISITOR = "0b6c1f3e-visitor-0001"
USER_ID = "3f2b8c1e-7a4d-4e9b-9c2a-5d6e7f8a9b0c"
PINNING = FrontendFlag.CHAT_PINNING.value
HIRE_EXPERTS = FrontendFlag.HIRE_EXPERTS.value
CHAT_SEARCH = FrontendFlag.CHAT_SEARCH.value
STREAM_SHADOW = FrontendFlag.COPILOT_STREAM_SHADOW.value

app = fastapi.FastAPI()
app.include_router(router, prefix="/api")


@pytest.fixture(autouse=True)
def launchdarkly(mocker, monkeypatch: pytest.MonkeyPatch):
    """A real LaunchDarkly client serving fixed targeting, and no env overrides.

    Yields every analytics event the client would have sent LaunchDarkly.
    """
    for name in list(os.environ):
        if name.startswith(
            ("FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_", "FORCE_ALL_FLAGS")
        ):
            monkeypatch.delenv(name)
    td = TestData.data_source()
    td.update(
        td.flag(FrontendFlag.COPILOT_VOICE_MODE.value)
        .boolean_flag()
        .variation_for_user("u-staff", True)
        .fallthrough_variation(False)
    )
    td.update(
        td.flag(FrontendFlag.HIRE_EXPERTS.value).boolean_flag().variation_for_all(True)
    )
    td.update(
        td.flag(FrontendFlag.MARKETPLACE_SEARCH_TERMS.value)
        .variations(SEARCH_TERMS, VISITOR_TERMS)
        .variation_for_user(f"visitor:{VISITOR}", 1)
        .fallthrough_variation(0)
    )
    td.update(
        td.flag(PINNING)
        .boolean_flag()
        .variation_for_user(f"visitor:{VISITOR}", True)
        .variation_for_user(f"visitor:{'v' * 65}", True)
        .variation_for_user("visitor:visitor:bad$id", True)
        .variation_for_user(USER_ID, True)
        .if_match("email_domain", "agpt.co")
        .then_return(True)
        .fallthrough_variation(False)
    )
    _RecordedEvents.sent.clear()
    client = LDClient(
        LDConfig(
            "sdk-test", update_processor_class=td, event_processor_class=_RecordedEvents
        )
    )
    mocker.patch.object(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.LAUNCHDARKLY
    )
    mocker.patch.object(ff.settings.secrets, "launch_darkly_sdk_key", "sdk-test")
    monkeypatch.setattr(ff, "_init_attempted", True)
    mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)
    _anonymous_flags.cache_clear()
    _defined_flags.cache_clear()
    yield _RecordedEvents.sent
    _anonymous_flags.cache_clear()
    _defined_flags.cache_clear()
    app.dependency_overrides.clear()
    client.close()


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


async def test_a_flag_the_vendor_has_no_definition_of_is_answered_null():
    flags = (await _get_flags("u-staff"))["flags"]

    # Null serves the page's default now; left out, the page would wait for good.
    assert (
        flags.get(CHAT_SEARCH, "left out"),
        flags.get(STREAM_SHADOW, "left out"),
        flags[HIRE_EXPERTS],
    ) == (None, None, True)


async def test_a_defined_flag_the_vendor_could_not_answer_is_left_out(mocker):
    # A failed user lookup makes every answer a guess about someone else.
    mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        AsyncMock(
            return_value=(Context.builder(USER_ID).anonymous(True).build(), False)
        ),
    )

    flags = (await _get_flags(USER_ID))["flags"]

    # A default here would read as an answer and 404 a gated route for a user who has it.
    assert (HIRE_EXPERTS in flags, flags.get(CHAT_SEARCH, "left out")) == (False, None)


@pytest.mark.parametrize("ready_first", [False, True])
async def test_a_signed_out_answer_is_kept_long_only_when_complete(
    mocker, monkeypatch, ready_first
):
    """A vendor that is down or starting is asked again within seconds, while a
    complete answer keeps serving every signed-out visitor from one read."""
    monkeypatch.setattr(frontend, "PARTIAL_TTL_SECONDS", 0.01)
    ready = ff.ldclient.get()
    not_ready = LDClient(LDConfig("sdk-test", offline=True))
    get = mocker.patch(
        "backend.util.feature_flag.ldclient.get",
        return_value=ready if ready_first else not_ready,
    )
    try:
        first = (await _get_flags(None))["flags"]
        get.return_value = not_ready if ready_first else ready
        await asyncio.sleep(0.05)
        second = (await _get_flags(None))["flags"]
    finally:
        not_ready.close()

    # Either way the second read sees the first, complete answer.
    assert (HIRE_EXPERTS in first, second.get(HIRE_EXPERTS)) == (ready_first, True)


@pytest.mark.parametrize(
    "data_source,body",
    [
        # A rejected key ends LaunchDarkly's attempts; the route gates then serve
        # defaults, and the page must too rather than poll forever.
        ("_RejectedKey", {"flags": {}, "configured": False}),
        # Still connecting past the start-up wait: the page keeps asking.
        ("_StillConnecting", {"flags": {}, "configured": True}),
    ],
)
async def test_a_vendor_that_cannot_answer_says_whether_it_ever_will(
    mocker, data_source, body
):
    client = LDClient(
        LDConfig(
            "sdk-test", update_processor_class=globals()[data_source], send_events=False
        ),
        start_wait=0,
    )
    mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)
    try:
        answer = await _get_flags("u-staff")
    finally:
        client.close()

    assert answer == body


async def test_a_launchdarkly_client_not_ready_yet_leaves_every_flag_out(mocker):
    # Offline, the client answers every read as CLIENT_NOT_READY, as while it starts.
    client = LDClient(LDConfig("sdk-test", offline=True))
    mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)
    try:
        flags = (await _get_flags("u-staff"))["flags"]
    finally:
        client.close()

    assert flags == {}


async def test_a_signed_in_read_asks_posthog_nothing_about_flags_it_lacks(mocker):
    """Each would be a remote /flags call on the executor every flag read shares."""
    mocker.patch.object(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.POSTHOG
    )
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

    # hire-experts is defined but needs /flags; nothing else may ask.
    assert (remote.call_count, flags.get(CHAT_SEARCH, "left out")) == (1, None)


@pytest.mark.parametrize("loaded", [True, False])
async def test_on_posthog_only_loaded_definitions_can_rule_a_flag_undefined(
    mocker, loaded
):
    mocker.patch.object(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.POSTHOG
    )
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

    # hire-experts is defined but needs /flags, which is down: left out either way.
    expected = (False, None) if loaded else (False, "left out")
    assert (HIRE_EXPERTS in flags, flags.get(CHAT_SEARCH, "left out")) == expected


async def test_with_no_vendor_configured_every_flag_is_the_frontends_default(mocker):
    mocker.patch.object(ff.settings.secrets, "launch_darkly_sdk_key", "")

    body = await _get_flags("u-staff")

    assert body["configured"] is False


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
    monkeypatch.setenv("FORCE_FLAG_CHAT_SEARCH", "true")

    flags = (await _get_flags(user_id, visitor_id))["flags"]

    assert flags[FrontendFlag.COPILOT_VOICE_MODE.value] is True
    assert flags[FrontendFlag.MARKETPLACE_SEARCH_TERMS.value] == terms
    assert flags[CHAT_SEARCH] is True


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
    # The lookup would hand back an agpt.co user, whom the flag's rule turns on.
    employee = Context.builder("u-staff").set("email_domain", "agpt.co").build()
    lookup = mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        AsyncMock(return_value=(employee, True)),
    )

    flags = (await _get_flags(None, USER_ID))["flags"]

    assert flags[PINNING] is False
    assert USER_ID not in [call.args[0] for call in lookup.await_args_list]


async def test_a_users_id_sent_as_a_visitor_id_never_matches_that_users_targets(
    mocker,
):
    mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        AsyncMock(return_value=(Context.builder(USER_ID).build(), True)),
    )

    as_user = (await _get_flags(USER_ID))["flags"]
    as_visitor = (await _get_flags(None, USER_ID))["flags"]

    assert (as_user[PINNING], as_visitor[PINNING]) == (True, False)


async def test_fresh_visitor_ids_send_launchdarkly_nothing(launchdarkly):
    await _get_flags(None)
    launchdarkly.clear()

    for _ in range(20):
        await _get_flags(None, str(uuid.uuid4()))

    assert launchdarkly == []


async def test_fresh_visitor_ids_cost_posthog_nothing(mocker):
    """No billed event, no remote /flags call and no turn on the executor every
    flag read shares: the id is the caller's to pick, so a new one must be free."""
    mocker.patch.object(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.POSTHOG
    )
    client = _posthog_splitting_pinning_and_gating_hire_on_email()
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    remote = mocker.patch.object(
        client,
        "_get_flags_decision",
        return_value={
            "flags": {
                HIRE_EXPERTS: FeatureFlag.from_value_and_payload(
                    HIRE_EXPERTS, True, None
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
    assert all(v.get(HIRE_EXPERTS) is True for v in visitors)


class _StillConnecting(UpdateProcessor):
    """A data source that has not received flags yet."""

    def __init__(self, config: LDConfig, store: Any, ready: Any):
        self.config = config

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def is_alive(self) -> bool:
        return True

    def initialized(self) -> bool:
        return False


class _RejectedKey(_StillConnecting):
    """Gives up for good, as LaunchDarkly's own data sources do on a 401."""

    def start(self) -> None:
        sink = self.config.data_source_update_sink
        assert sink is not None
        sink.update_status(DataSourceState.OFF, None)


class _RecordedEvents(EventProcessor):
    sent: ClassVar[list[Any]] = []

    def __init__(self, config: LDConfig):
        pass

    def send_event(self, event: Any) -> None:
        self.sent.append(event)

    def flush(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _posthog_splitting_pinning_and_gating_hire_on_email() -> Posthog:
    """Local definitions: pinning is a 50% split, hire-experts needs an email."""
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
            "key": HIRE_EXPERTS,
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
