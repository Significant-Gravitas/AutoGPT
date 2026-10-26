"""Flag reads through PostHog, the one flag vendor, end to end."""

import json
import logging
import os
import uuid

import pytest
import sentry_sdk
import sentry_sdk.feature_flags
import urllib3.connectionpool
from fastapi import HTTPException
from posthog import Posthog
from posthog.types import FeatureFlagResult
from sentry_sdk.tracing import Span

import backend.util.feature_flag as ff
import backend.util.feature_flag.posthog as ph
from backend.util.feature_flag import (
    Flag,
    evaluate_feature_flag,
    feature_flag,
    frontend,
    is_feature_enabled,
)
from backend.util.settings import AppEnvironment, Settings

SERVER_READ_EXPERIMENT = "server-read-experiment"


@pytest.fixture(autouse=True)
def no_env_override(monkeypatch: pytest.MonkeyPatch):
    """`.env` may force flags; pin every flag under test to PostHog."""
    for name in list(os.environ):
        if name.startswith(_FORCE_PREFIXES):
            monkeypatch.delenv(name)


_FORCE_PREFIXES = (
    "FORCE_FLAG_",
    "NEXT_PUBLIC_FORCE_FLAG_",
    "FORCE_ALL_FLAGS",
    "NEXT_PUBLIC_FORCE_ALL_FLAGS",
)


@pytest.fixture
def user_context(mocker):
    """A resolved context, so `authoritative` turns purely on evaluation."""
    context = ff._UserContext(key="u-1", anonymous=False)
    return mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        return_value=(context, True),
    )


def stub_posthog(mocker, *, value, payload=None):
    """Stand in for PostHog's answer for one flag."""
    client = mocker.Mock()
    client.get_feature_flag_result.return_value = (
        FeatureFlagResult.from_value_and_payload("flag", value, payload)
    )
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    return client


class TestObsoleteBackendSettings:
    """Deploy config may still set the LaunchDarkly-era variables for a while."""

    def test_a_leftover_backend_or_sdk_key_does_not_stop_boot(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("FEATURE_FLAG_BACKEND", "launchdarkly")
        monkeypatch.setenv("LAUNCH_DARKLY_SDK_KEY", "sdk-leftover")

        settings = Settings()

        # Nothing reads them any more: they are not settings, just ignored values.
        assert "feature_flag_backend" not in type(settings.config).model_fields
        assert "launch_darkly_sdk_key" not in type(settings.secrets).model_fields


class TestPostHogWithoutAKey:
    """The default backend as a self-hosted install and CI run it."""

    @pytest.fixture(autouse=True)
    def no_key(self, mocker, monkeypatch: pytest.MonkeyPatch):
        mocker.patch.object(ph.settings.secrets, "posthog_api_key", "")
        mocker.patch.object(ff.settings.secrets, "posthog_secret_key", "")
        monkeypatch.setattr(ph, "_client", None)
        monkeypatch.setattr(ph, "_init_attempted", False)
        monkeypatch.setattr(ph, "_shut_down", False)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("attributes", [None, {"country": "US"}])
    async def test_a_read_serves_its_default_without_a_user_lookup(
        self, mocker, attributes
    ):
        user_id = str(uuid.uuid4())
        lookup = mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(ff._UserContext(key=user_id, anonymous=False), True),
        )

        value = await ff.get_feature_flag_value(
            Flag.GENERIC_TRIGGER_AGENTS.value, user_id, True, attributes=attributes
        )

        assert value is True
        lookup.assert_not_called()

    def test_startup_warns_once_and_logs_no_error(self, caplog):
        with caplog.at_level(logging.WARNING):
            ff.initialize_feature_flags()

        assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
            (
                logging.WARNING,
                "PostHog API key not configured; flag reads will not resolve",
            )
        ]


@pytest.mark.usefixtures("posthog_events_dropped")
class TestPostHogBackend:
    @pytest.mark.asyncio
    async def test_an_enabled_flag_is_authoritative(self, mocker, user_context):
        stub_posthog(mocker, value=True)

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            True,
            True,
        )

    @pytest.mark.asyncio
    async def test_a_conclusive_off_is_authoritative(self, mocker, user_context):
        """PostHog reports a conclusive off, so it is an answer, not a fallback."""
        stub_posthog(mocker, value=False)

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            False,
            True,
        )

    @pytest.mark.asyncio
    async def test_an_unresolved_flag_is_not(self, mocker, user_context):
        stub_posthog(mocker, value=None)

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            False,
            False,
        )

    @pytest.mark.asyncio
    async def test_an_unconfigured_client_is_not(self, mocker, user_context):
        mocker.patch.object(ph, "get_flag_client", return_value=None)

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            False,
            False,
        )

    @pytest.mark.asyncio
    async def test_an_evaluation_that_raises_is_not(self, mocker, user_context):
        client = stub_posthog(mocker, value=True)
        client.get_feature_flag_result.side_effect = Exception("evaluation exploded")

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            False,
            False,
        )

    @pytest.mark.asyncio
    async def test_a_degraded_user_context_is_not(self, mocker):
        stub_posthog(mocker, value=True)
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(ff._anonymous_context("u-1"), False),
        )

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            True,
            False,
        )

    @pytest.mark.asyncio
    async def test_a_payload_flag_returns_its_payload(self, mocker, user_context):
        """The JSON-valued flags carry their value as a payload."""
        stub_posthog(mocker, value=True, payload={"daily": 5, "weekly": 20})

        value = await ff.get_feature_flag_value("copilot-cost-limits", "system")

        assert value == {"daily": 5, "weekly": 20}

    @pytest.mark.asyncio
    async def test_a_non_boolean_value_is_not_authoritative(self, mocker, user_context):
        stub_posthog(mocker, value=True, payload={"some": "object"})

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            False,
            False,
        )

    @pytest.mark.asyncio
    async def test_targeting_attributes_are_passed_as_person_properties(self, mocker):
        """Every attribute a rule targets on has to reach PostHog, or the flag
        silently evaluates against a user without them — and nothing more."""
        client = stub_posthog(mocker, value=True)
        context = ff._UserContext(
            key="u-1",
            anonymous=False,
            attributes={
                "role": "admin",
                "email": "x@agpt.co",
                "email_domain": "agpt.co",
                "created_at": "2026-05-07T12:00:00+00:00",
                "unrelated": "never sent",
            },
        )
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(context, True),
        )

        await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1")

        _, kwargs = client.get_feature_flag_result.call_args
        assert kwargs["person_properties"] == {
            "role": "admin",
            "email": "x@agpt.co",
            "email_domain": "agpt.co",
            "created_at": "2026-05-07T12:00:00+00:00",
            "is_anonymous": False,
            "environment": ff.settings.config.app_env.value,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "email,expected", [("x@agpt.co", (True, True)), ("x@gmail.com", (False, True))]
    )
    async def test_an_email_cohort_evaluates_in_process(self, mocker, email, expected):
        """The ``ld:employee`` cohort matches on ``email``; without it a
        read cannot be decided locally and becomes a remote /flags call."""
        posthog_client = _posthog_with_an_employee_cohort()
        mocker.patch.object(ph, "get_flag_client", return_value=posthog_client)
        context = ff._UserContext(
            key="u-1",
            anonymous=False,
            attributes={"email": email, "email_domain": email.split("@")[1]},
        )
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(context, True),
        )
        try:
            result = await evaluate_feature_flag(Flag.COPILOT_VOICE_MODE, "u-1")
        finally:
            posthog_client.shutdown()

        assert result == expected

    @pytest.mark.asyncio
    async def test_an_anonymous_context_carries_only_is_anonymous_and_the_environment(
        self, mocker
    ):
        """The `"system"`-keyed config flags have no user to describe and are not
        the shared signed-out key, but can still differ between dev and prod."""
        client = stub_posthog(mocker, value=True, payload={"TIER": 1.5})

        await ff.get_feature_flag_value("copilot-tier-multipliers", "system")

        args, kwargs = client.get_feature_flag_result.call_args
        assert args[1] == "system"
        assert kwargs["person_properties"] == {
            "is_anonymous": False,
            "environment": ff.settings.config.app_env.value,
        }

    @pytest.mark.asyncio
    async def test_a_split_on_is_anonymous_leaves_out_only_the_shared_signed_out_key(
        self, mocker
    ):
        """A 20% rollout "where is_anonymous is not true" buckets consented
        visitors by their own id and never the visitors nobody can measure."""
        posthog_client = _posthog_with_a_split_on_is_anonymous()
        mocker.patch.object(ph, "get_flag_client", return_value=posthog_client)
        frontend._anonymous_flags.cache_clear()
        frontend._defined_flags.cache_clear()
        try:
            anonymous = await frontend.evaluate_frontend_flags(None)
            visitors = [
                await frontend.evaluate_frontend_flags(None, f"{i:08d}-visitor")
                for i in range(200)
            ]
        finally:
            frontend._anonymous_flags.cache_clear()
            frontend._defined_flags.cache_clear()
            posthog_client.shutdown()

        split = frontend.FrontendFlag.TASK_PROGRESS_BAR
        share = sum(v.flags.get(split) is True for v in visitors) / len(visitors)
        # One assertion, so a failure shows both halves.
        assert (anonymous.flags.get(split), 0.1 < share < 0.3) == (False, True), share

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flag_key,events",
        [(SERVER_READ_EXPERIMENT, 1), (Flag.GENERIC_TRIGGER_AGENTS.value, 0)],
    )
    async def test_only_an_experiment_read_sends_posthog_its_exposure_event(
        self, mocker, monkeypatch, flag_key, events
    ):
        """`$feature_flag_called` is billed per event, and only an experiment
        counts its exposures with it."""
        mocker.patch.object(ff, "EXPERIMENT_FLAGS", frozenset({SERVER_READ_EXPERIMENT}))
        monkeypatch.setattr(
            urllib3.connectionpool.HTTPConnectionPool, "urlopen", _refuse_network
        )
        posthog_client = _posthog_rolled_out(flag_key)
        capture = mocker.patch.object(posthog_client, "capture")
        mocker.patch.object(ph, "get_flag_client", return_value=posthog_client)
        try:
            value = await ff.get_feature_flag_value(flag_key, "u-1")
        finally:
            posthog_client.shutdown()

        sent = [call.args[0] for call in capture.call_args_list]
        assert (value, sent) == (True, ["$feature_flag_called"] * events)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "app_env,expected",
        [
            (AppEnvironment.DEVELOPMENT, (True, True)),
            (AppEnvironment.PRODUCTION, (False, True)),
        ],
    )
    async def test_a_dev_only_rollout_evaluates_in_process(
        self, mocker, user_context, app_env, expected
    ):
        """A release condition on ``environment`` decides in process, per app env."""
        mocker.patch.object(ff.settings.config, "app_env", app_env)
        posthog_client = _posthog_rolled_out_to_dev_only()
        mocker.patch.object(ph, "get_flag_client", return_value=posthog_client)
        try:
            result = await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1")
        finally:
            posthog_client.shutdown()

        assert result == expected

    def test_lifecycle_starts_and_stops_posthog(self, mocker):
        start = mocker.patch.object(ph, "initialize_posthog_flags")
        stop = mocker.patch.object(ph, "shutdown_posthog_flags")

        ff.initialize_feature_flags()
        start.assert_called_once()
        stop.assert_not_called()

        ff.shutdown_feature_flags()
        stop.assert_called_once()

    def test_readiness_follows_the_posthog_client(self, mocker):
        mocker.patch.object(ph, "get_flag_client", return_value=None)
        assert ff._flag_backend_initialized() is False

        mocker.patch.object(ph, "get_flag_client", return_value=mocker.Mock())
        assert ff._flag_backend_initialized() is True


class TestForcedFlags:
    @pytest.mark.asyncio
    async def test_an_env_override_answers_without_posthog(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        posthog = mocker.patch.object(ph, "evaluate_flag")
        monkeypatch.setenv("FORCE_FLAG_GENERIC_TRIGGER_AGENTS", "true")

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            True,
            True,
        )
        posthog.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_all_opens_the_route_gates(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        """The local override sits above PostHog, on every gate.

        Both gates consult it ahead of the "PostHog cannot answer" bail-out, so
        an unconfigured PostHog must not put a developer's switch back behind one.
        """
        mocker.patch.object(ff.settings.config, "app_env", AppEnvironment.LOCAL)
        mocker.patch.object(ff, "_force_all_logged", True)
        mocker.patch.object(ph, "is_configured", return_value=False)
        mocker.patch.object(ph, "get_flag_client", return_value=None)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            True,
            True,
        )
        await ff.create_feature_flag_dependency(Flag.GENERIC_TRIGGER_AGENTS)("u-1")
        assert await _gated_route()(user_id="u-1") == "served"

    @pytest.mark.asyncio
    async def test_posthog_answers_the_gates_again_without_the_switch(
        self, mocker, monkeypatch: pytest.MonkeyPatch, user_context
    ):
        mocker.patch.object(ff.settings.config, "app_env", AppEnvironment.LOCAL)
        monkeypatch.delenv("FORCE_ALL_FLAGS", raising=False)
        monkeypatch.delenv("NEXT_PUBLIC_FORCE_ALL_FLAGS", raising=False)
        mocker.patch.object(ph, "is_configured", return_value=True)
        stub_posthog(mocker, value=True)

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            True,
            True,
        )
        await ff.create_feature_flag_dependency(Flag.GENERIC_TRIGGER_AGENTS)("u-1")
        assert await _gated_route()(user_id="u-1") == "served"

        stub_posthog(mocker, value=False)
        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            False,
            True,
        )
        with pytest.raises(HTTPException) as off:
            await ff.create_feature_flag_dependency(Flag.GENERIC_TRIGGER_AGENTS)("u-1")
        assert off.value.status_code == 404


class TestSentryFlagContext:
    """The served value lands on the scope Sentry attaches to errors."""

    @pytest.fixture
    def sentry_flags(self):
        with sentry_sdk.isolation_scope() as scope:
            # A forked scope inherits whatever earlier tests recorded.
            scope.flags.clear()
            yield scope.flags

    @pytest.mark.asyncio
    async def test_a_boolean_flag_is_recorded_as_served(
        self, mocker, user_context, sentry_flags
    ):
        stub_posthog(mocker, value=True)

        assert await is_feature_enabled(Flag.GENERIC_TRIGGER_AGENTS, "u-1") is True
        assert sentry_flags.get() == [
            {"flag": Flag.GENERIC_TRIGGER_AGENTS.value, "result": True}
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured", [True, False])
    async def test_a_non_boolean_flag_records_nothing(
        self, mocker, user_context, sentry_flags, configured
    ):
        if configured:
            stub_posthog(mocker, value=True, payload={"daily": 5})
        else:
            mocker.patch.object(ph, "get_flag_client", return_value=None)

        await ff.get_feature_flag_value("copilot-cost-limits", "u-1", {"daily": 1})
        assert sentry_flags.get() == []

    @pytest.mark.asyncio
    async def test_a_fallback_is_marked_beside_its_value(
        self, mocker, user_context, sentry_flags
    ):
        mocker.patch.object(ph, "get_flag_client", return_value=None)

        await is_feature_enabled(Flag.GENERIC_TRIGGER_AGENTS, "u-1")

        assert sentry_flags.get() == [
            {"flag": Flag.GENERIC_TRIGGER_AGENTS.value, "result": False},
            {"flag": f"{Flag.GENERIC_TRIGGER_AGENTS.value}.fallback", "result": True},
        ]

    @pytest.mark.asyncio
    async def test_a_non_boolean_answer_to_a_boolean_read_records_the_default(
        self, mocker, user_context, sentry_flags
    ):
        stub_posthog(mocker, value=True, payload={"some": "object"})

        assert await evaluate_feature_flag(
            Flag.GENERIC_TRIGGER_AGENTS, "u-1", True
        ) == (
            True,
            False,
        )
        assert sentry_flags.get() == [
            {"flag": Flag.GENERIC_TRIGGER_AGENTS.value, "result": True},
            {"flag": f"{Flag.GENERIC_TRIGGER_AGENTS.value}.fallback", "result": True},
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "gate, configured, initialized",
        [
            ("decorator", True, False),
            ("decorator", True, True),
            ("dependency", False, True),
            ("dependency", True, False),
        ],
    )
    async def test_a_gate_serving_its_default_records_it(
        self,
        mocker,
        user_context,
        sentry_flags,
        gate,
        configured,
        initialized,
    ):
        mocker.patch.object(ph, "is_configured", return_value=configured)
        if initialized:
            # Answers with a non-boolean, which a gate also reads as its default.
            stub_posthog(mocker, value=True, payload={"some": "object"})
        else:
            mocker.patch.object(ph, "get_flag_client", return_value=None)
        # A stale earlier answer must not be what an error from this route carries.
        sentry_sdk.feature_flags.add_feature_flag(
            Flag.GENERIC_TRIGGER_AGENTS.value, True
        )

        with pytest.raises(HTTPException):
            if gate == "decorator":
                await _gated_route()(user_id="u-1")
            else:
                await ff.create_feature_flag_dependency(Flag.GENERIC_TRIGGER_AGENTS)(
                    "u-1"
                )

        assert {f["flag"]: f["result"] for f in sentry_flags.get()} == {
            Flag.GENERIC_TRIGGER_AGENTS.value: False,
            f"{Flag.GENERIC_TRIGGER_AGENTS.value}.fallback": True,
        }

    @pytest.mark.asyncio
    async def test_a_fallback_marker_stays_off_the_span(
        self, mocker, user_context, sentry_flags
    ):
        mocker.patch.object(ph, "get_flag_client", return_value=None)
        span = Span()
        scope = sentry_sdk.get_current_scope()
        previous, scope.span = scope.span, span
        try:
            await is_feature_enabled(Flag.GENERIC_TRIGGER_AGENTS, "u-1")
        finally:
            scope.span = previous

        assert span._flags == {
            f"flag.evaluation.{Flag.GENERIC_TRIGGER_AGENTS.value}": False
        }

    @pytest.mark.asyncio
    async def test_a_real_answer_clears_an_earlier_fallback(
        self, mocker, user_context, sentry_flags
    ):
        sentry_sdk.feature_flags.add_feature_flag(
            f"{Flag.GENERIC_TRIGGER_AGENTS.value}.fallback", True
        )
        stub_posthog(mocker, value=True)

        await is_feature_enabled(Flag.GENERIC_TRIGGER_AGENTS, "u-1")

        assert {f["flag"]: f["result"] for f in sentry_flags.get()} == {
            f"{Flag.GENERIC_TRIGGER_AGENTS.value}.fallback": False,
            Flag.GENERIC_TRIGGER_AGENTS.value: True,
        }

    @pytest.mark.asyncio
    async def test_a_recording_failure_does_not_break_the_read(
        self, mocker, user_context
    ):
        mocker.patch.object(
            sentry_sdk.feature_flags,
            "add_feature_flag",
            side_effect=RuntimeError("sentry down"),
        )
        stub_posthog(mocker, value=True)

        assert await evaluate_feature_flag(Flag.GENERIC_TRIGGER_AGENTS, "u-1") == (
            True,
            True,
        )


@pytest.mark.usefixtures("posthog_events_dropped")
class TestTheCountryRule:
    """The trial's country targeting, through PostHog's real evaluator."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "country,offered", [("US", True), ("BR", True), ("IN", False), (None, False)]
    )
    async def test_the_country_decides_the_offer(self, mocker, country, offered):
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(ff._UserContext(key="u-1", anonymous=False), True),
        )
        posthog_client = _posthog_serving_all_but_india()
        mocker.patch.object(ph, "get_flag_client", return_value=posthog_client)
        try:
            value = await ff.get_feature_flag_value(
                _TRIAL_FLAG,
                "u-1",
                None,
                attributes={"country": country} if country else None,
            )
        finally:
            posthog_client.shutdown()
        assert (value == {"version": "v1"}) is offered

    @pytest.mark.asyncio
    async def test_a_degraded_lookup_sends_no_country(self, mocker):
        """An anonymous stand-in for a user sends no attributes, the country
        included, so a country rule never offers the trial on a failed lookup."""
        client = stub_posthog(mocker, value=True)
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(ff._anonymous_context("u-1"), False),
        )

        await ff.get_feature_flag_value(
            _TRIAL_FLAG, "u-1", attributes={"country": "US"}
        )

        _, kwargs = client.get_feature_flag_result.call_args
        assert "country" not in kwargs["person_properties"]


_TRIAL_FLAG = "card-required-trial-offer"


def _posthog_serving_all_but_india() -> Posthog:
    """Every country but India gets the offer, evaluated locally."""
    client = Posthog("phc-test", secret_key="phx-test", enable_local_evaluation=False)
    client.feature_flags = [
        {
            "id": 1,
            "key": _TRIAL_FLAG,
            "active": True,
            "filters": {
                "groups": [
                    {
                        "properties": [
                            {
                                "key": "country",
                                "type": "person",
                                "operator": "is_set",
                                "value": "is_set",
                            },
                            {
                                "key": "country",
                                "type": "person",
                                "operator": "is_not",
                                "value": ["IN"],
                            },
                        ],
                        "rollout_percentage": 100,
                    }
                ],
                "payloads": {"true": json.dumps({"version": "v1"})},
            },
        }
    ]
    # A person without a country is inconclusive locally; keep the fallback offline.
    client._get_flags_decision = _no_remote_flags
    return client


def _posthog_with_an_employee_cohort() -> Posthog:
    """An email-targeted cohort, evaluated locally."""
    client = Posthog("phc-test", secret_key="phx-test", enable_local_evaluation=False)
    client.feature_flags = [
        {
            "id": 1,
            "key": Flag.COPILOT_VOICE_MODE.value,
            "active": True,
            "filters": {
                "groups": [
                    {
                        "properties": [{"key": "id", "type": "cohort", "value": 7}],
                        "rollout_percentage": 100,
                    }
                ]
            },
        }
    ]
    client.cohorts = {
        "7": {
            "type": "OR",
            "values": [
                {
                    "type": "AND",
                    "values": [
                        {
                            "key": "email",
                            "type": "person",
                            "operator": "regex",
                            "value": r"@agpt\.co$",
                        }
                    ],
                }
            ],
        }
    }
    client._get_flags_decision = _no_remote_flags
    return client


def _posthog_with_a_split_on_is_anonymous() -> Posthog:
    """A 20% rollout whose condition leaves out `is_anonymous: true`."""
    client = Posthog("phc-test", secret_key="phx-test", enable_local_evaluation=False)
    client.feature_flags = [
        {
            "id": 2,
            "key": "task-progress-bar",
            "active": True,
            "filters": {
                "groups": [
                    {
                        "properties": [
                            {
                                "key": "is_anonymous",
                                "type": "person",
                                "operator": "is_not",
                                "value": ["true"],
                            }
                        ],
                        "rollout_percentage": 20,
                    }
                ]
            },
        }
    ]
    client._get_flags_decision = _no_remote_flags
    return client


def _posthog_rolled_out(flag_key: str) -> Posthog:
    client = Posthog("phc-test", secret_key="phx-test", enable_local_evaluation=False)
    client.feature_flags = [
        {
            "id": 1,
            "key": flag_key,
            "active": True,
            "filters": {"groups": [{"properties": [], "rollout_percentage": 100}]},
        }
    ]
    return client


def _refuse_network(pool, method, url, *args, **kwargs):
    raise urllib3.exceptions.ProtocolError(f"no network in tests: {method} {url}")


def _posthog_rolled_out_to_dev_only() -> Posthog:
    """A flag whose one release condition is ``environment = dev``."""
    client = Posthog("phc-test", secret_key="phx-test", enable_local_evaluation=False)
    client.feature_flags = [
        {
            "id": 1,
            "key": Flag.GENERIC_TRIGGER_AGENTS.value,
            "active": True,
            "filters": {
                "groups": [
                    {
                        "properties": [
                            {
                                "key": "environment",
                                "type": "person",
                                "operator": "exact",
                                "value": ["dev"],
                            }
                        ],
                        "rollout_percentage": 100,
                    }
                ]
            },
        }
    ]
    client._get_flags_decision = _no_remote_flags
    return client


def _no_remote_flags(*args, **kwargs):
    raise ConnectionError("remote /flags is unavailable in tests")


def _gated_route():
    """A route behind the decorator, which holds a raw flag key rather than a Flag."""

    @feature_flag(Flag.GENERIC_TRIGGER_AGENTS.value)
    async def route(user_id: str) -> str:
        return "served"

    return route
