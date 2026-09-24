"""Backend selection: launchdarkly (default), posthog, and the dual diff run."""

import asyncio
import json
import logging
import os
import uuid

import pytest
import sentry_sdk
import sentry_sdk.feature_flags
from fastapi import HTTPException
from ldclient import Context, LDClient
from ldclient.config import Config as LDConfig
from ldclient.integrations.test_data import TestData
from posthog import Posthog
from sentry_sdk.tracing import Span

import backend.util.feature_flag as ff
import backend.util.feature_flag.posthog as ph
from backend.util.feature_flag import (
    Flag,
    evaluate_feature_flag,
    feature_flag,
    is_feature_enabled,
)
from backend.util.settings import AppEnvironment, Config, FeatureFlagBackend


@pytest.fixture(autouse=True)
async def no_leaked_shadow_evaluations():
    """A dual probe left pending would surface inside the NEXT test's caplog."""
    yield
    await drain_shadow_evaluations()


@pytest.fixture(autouse=True)
def no_unanswered_flags_reported():
    ff._unanswered_flags_reported.clear()
    yield
    ff._unanswered_flags_reported.clear()


@pytest.fixture(autouse=True)
def no_env_override(monkeypatch: pytest.MonkeyPatch):
    """`.env` may force flags; pin every flag under test to the vendors."""
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
def ld_client(mocker):
    client = mocker.Mock(spec=LDClient)
    mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)
    client.is_initialized.return_value = True
    return client


@pytest.fixture
def user_context(mocker):
    """A resolved context, so `authoritative` turns purely on evaluation."""
    context = Context.builder("u-1").kind("user").anonymous(False).build()
    return mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        return_value=(context, True),
    )


def _mismatch_record(message: str) -> dict:
    """The log formatter wraps the message in ANSI colour codes."""
    return json.loads(message[message.index("{") : message.rindex("}") + 1])


async def drain_shadow_evaluations() -> None:
    """Dual reports off the caller's path, so a test has to wait for it."""
    while ff._shadow_evaluations:
        await asyncio.gather(*list(ff._shadow_evaluations))


def use_backend(mocker, backend: FeatureFlagBackend):
    mocker.patch.object(ff.settings.config, "feature_flag_backend", backend)


def stub_posthog(mocker, *, value, payload=None):
    """Stand in for a PostHog evaluation snapshot for one flag."""
    snapshot = mocker.Mock()
    snapshot.get_flag.return_value = value
    snapshot.get_flag_payload.return_value = payload
    client = mocker.Mock()
    client.evaluate_flags.return_value = snapshot
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    return client


class TestDefaultBackendIsUnchanged:
    """Phase 1 is a no-op until the setting is flipped."""

    def test_the_default_is_launchdarkly(self):
        assert (
            ff.settings.config.feature_flag_backend is FeatureFlagBackend.LAUNCHDARKLY
        )

    @pytest.mark.asyncio
    async def test_the_default_never_touches_posthog(
        self, mocker, ld_client, user_context
    ):
        posthog = mocker.patch.object(ph, "evaluate_flag")
        ld_client.variation.return_value = True

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, True)
        ld_client.variation.assert_called_once()
        posthog.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_default_reads_launchdarkly_for_raw_values(
        self, ld_client, user_context
    ):
        ld_client.variation.return_value = {"daily": 5}

        assert await ff.get_feature_flag_value("copilot-cost-limits", "system") == {
            "daily": 5
        }

    def test_lifecycle_starts_only_launchdarkly(self, mocker):
        launchdarkly = mocker.patch.object(ff, "initialize_launchdarkly")
        posthog = mocker.patch.object(ph, "initialize_posthog_flags")

        ff.initialize_feature_flags()

        launchdarkly.assert_called_once()
        posthog.assert_not_called()

    def test_readiness_asks_only_launchdarkly(self, mocker, ld_client):
        posthog = mocker.patch.object(ph, "is_configured")

        ff._flag_backend_initialized()

        ld_client.is_initialized.assert_called_once()
        posthog.assert_not_called()


class TestAnUnknownBackendValue:
    """Settings is built at import of every module that reads a flag, so a
    rejected value is a boot crash rather than a misconfigured flag read."""

    def test_a_typo_falls_back_to_launchdarkly(self, monkeypatch):
        monkeypatch.setenv("FEATURE_FLAG_BACKEND", "posthogg")
        assert Config().feature_flag_backend is FeatureFlagBackend.LAUNCHDARKLY

    def test_the_value_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("FEATURE_FLAG_BACKEND", "PostHog")
        assert Config().feature_flag_backend is FeatureFlagBackend.POSTHOG


class TestPostHogBackend:
    @pytest.mark.asyncio
    async def test_an_enabled_flag_is_authoritative(self, mocker, user_context):
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        stub_posthog(mocker, value=True)

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, True)

    @pytest.mark.asyncio
    async def test_a_conclusive_off_is_authoritative(self, mocker, user_context):
        """The distinction LaunchDarkly had to infer: PostHog reports it."""
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        stub_posthog(mocker, value=False)

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, True)

    @pytest.mark.asyncio
    async def test_an_unresolved_flag_is_not(self, mocker, user_context):
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        stub_posthog(mocker, value=None)

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_an_unconfigured_client_is_not(self, mocker, user_context):
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        mocker.patch.object(ph, "get_flag_client", return_value=None)

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_an_evaluation_that_raises_is_not(self, mocker, user_context):
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        client = stub_posthog(mocker, value=True)
        client.evaluate_flags.side_effect = Exception("evaluation exploded")

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_a_degraded_user_context_is_not(self, mocker):
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        stub_posthog(mocker, value=True)
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(Context.create("u-1"), False),
        )

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, False)

    @pytest.mark.asyncio
    async def test_a_payload_flag_returns_its_payload(self, mocker, user_context):
        """The JSON-valued flags: a payload stands in for LD's variation value."""
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        stub_posthog(mocker, value=True, payload={"daily": 5, "weekly": 20})

        value = await ff.get_feature_flag_value("copilot-cost-limits", "system")

        assert value == {"daily": 5, "weekly": 20}

    @pytest.mark.asyncio
    async def test_a_non_boolean_value_is_not_authoritative(self, mocker, user_context):
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        stub_posthog(mocker, value=True, payload={"some": "object"})

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_targeting_attributes_are_passed_as_person_properties(self, mocker):
        """Every attribute an LD rule targets on has to reach PostHog, or the
        flag silently evaluates against a user without them — and nothing
        more: the raw email is not one of them."""
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        client = stub_posthog(mocker, value=True)
        context = (
            Context.builder("u-1")
            .kind("user")
            .anonymous(False)
            .set("role", "admin")
            .set("custom", {"role": "admin"})
            .set("email", "x@agpt.co")
            .set("email_domain", "agpt.co")
            .set("created_at", "2026-05-07T12:00:00+00:00")
            .build()
        )
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(context, True),
        )

        await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1")

        _, kwargs = client.evaluate_flags.call_args
        assert kwargs["person_properties"] == {
            "role": "admin",
            "email_domain": "agpt.co",
            "created_at": "2026-05-07T12:00:00+00:00",
        }

    @pytest.mark.asyncio
    async def test_an_anonymous_context_carries_no_person_properties(self, mocker):
        """The `"system"`-keyed config flags have no user to describe."""
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        client = stub_posthog(mocker, value=True, payload={"TIER": 1.5})

        await ff.get_feature_flag_value("copilot-tier-multipliers", "system")

        args, kwargs = client.evaluate_flags.call_args
        assert args[0] == "system"
        assert kwargs["person_properties"] is None

    def test_lifecycle_starts_only_posthog(self, mocker):
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        launchdarkly = mocker.patch.object(ff, "initialize_launchdarkly")
        posthog = mocker.patch.object(ph, "initialize_posthog_flags")

        ff.initialize_feature_flags()

        posthog.assert_called_once()
        launchdarkly.assert_not_called()

    def test_readiness_follows_the_posthog_client(self, mocker):
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        mocker.patch.object(ph, "get_flag_client", return_value=None)
        assert ff._flag_backend_initialized() is False

        mocker.patch.object(ph, "get_flag_client", return_value=mocker.Mock())
        assert ff._flag_backend_initialized() is True


class TestDualBackend:
    @pytest.mark.asyncio
    async def test_launchdarkly_answers_even_when_posthog_disagrees(
        self, mocker, ld_client, user_context
    ):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        stub_posthog(mocker, value=False)

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, True)

    @pytest.mark.asyncio
    async def test_a_disagreement_is_logged_as_a_structured_record(
        self, mocker, ld_client, user_context, caplog
    ):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        stub_posthog(mocker, value=False)
        user_id = str(uuid.uuid4())

        with caplog.at_level(
            logging.WARNING, logger="backend.util.feature_flag.mismatch"
        ):
            await evaluate_feature_flag(Flag.HIRE_EXPERTS, user_id)
            await drain_shadow_evaluations()

        [message] = [
            record.getMessage()
            for record in caplog.records
            if record.name == "backend.util.feature_flag.mismatch"
        ]
        record = _mismatch_record(message)
        assert record["flag"] == Flag.HIRE_EXPERTS.value
        assert record["launchdarkly"] == {"value": True, "evaluated": True}
        assert record["posthog"] == {"value": False, "evaluated": True}
        assert user_id not in message
        assert record["user"] == ff._user_digest(user_id)

    @pytest.mark.asyncio
    async def test_agreement_logs_nothing(
        self, mocker, ld_client, user_context, caplog
    ):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        stub_posthog(mocker, value=True)

        with caplog.at_level(
            logging.WARNING, logger="backend.util.feature_flag.mismatch"
        ):
            await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1")
            await drain_shadow_evaluations()

        assert not [
            r for r in caplog.records if r.name == "backend.util.feature_flag.mismatch"
        ]

    @pytest.mark.asyncio
    async def test_an_authoritativeness_difference_is_a_mismatch(
        self, mocker, ld_client, user_context, caplog
    ):
        """Both say "off", but only one of them knows it — that is the
        difference the diff week exists to find."""
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = False
        stub_posthog(mocker, value=None)

        with caplog.at_level(
            logging.WARNING, logger="backend.util.feature_flag.mismatch"
        ):
            result = await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1")
            await drain_shadow_evaluations()

        assert result == (False, True)
        [message] = [
            r.getMessage()
            for r in caplog.records
            if r.name == "backend.util.feature_flag.mismatch"
        ]
        assert _mismatch_record(message)["posthog"] == {
            "value": False,
            "evaluated": False,
        }

    @pytest.mark.asyncio
    async def test_an_unanswered_flag_is_reported_once_per_flag(
        self, mocker, ld_client, user_context, caplog
    ):
        """Before phase 2 creates the flags PostHog answers nothing, so every
        read would otherwise log the same record."""
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        stub_posthog(mocker, value=None)

        with caplog.at_level(
            logging.WARNING, logger="backend.util.feature_flag.mismatch"
        ):
            for _ in range(3):
                await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1")
                await evaluate_feature_flag(Flag.CHAT_MODE_OPTION, "u-1")
            await drain_shadow_evaluations()

        flags = [
            _mismatch_record(r.getMessage())["flag"]
            for r in caplog.records
            if r.name == "backend.util.feature_flag.mismatch"
        ]
        assert sorted(flags) == sorted(
            [Flag.HIRE_EXPERTS.value, Flag.CHAT_MODE_OPTION.value]
        )

    @pytest.mark.asyncio
    async def test_an_answered_disagreement_is_reported_every_time(
        self, mocker, ld_client, user_context, caplog
    ):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        stub_posthog(mocker, value=False)

        with caplog.at_level(
            logging.WARNING, logger="backend.util.feature_flag.mismatch"
        ):
            for _ in range(3):
                await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1")
            await drain_shadow_evaluations()

        assert (
            len(
                [
                    r
                    for r in caplog.records
                    if r.name == "backend.util.feature_flag.mismatch"
                ]
            )
            == 3
        )

    @pytest.mark.asyncio
    async def test_a_posthog_failure_cannot_break_the_read(
        self, mocker, ld_client, user_context
    ):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        mocker.patch.object(ph, "evaluate_flag", side_effect=Exception("boom"))

        assert await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1") is True
        await drain_shadow_evaluations()

    @pytest.mark.asyncio
    async def test_an_unserializable_value_still_logs(
        self, mocker, ld_client, user_context, caplog
    ):
        """A mismatch record must never be the thing that raises."""
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = object()
        stub_posthog(mocker, value=True)

        with caplog.at_level(
            logging.WARNING, logger="backend.util.feature_flag.mismatch"
        ):
            await ff.get_feature_flag_value("copilot-cost-limits", "u-1")
            await drain_shadow_evaluations()

        assert [
            r for r in caplog.records if r.name == "backend.util.feature_flag.mismatch"
        ]

    def test_lifecycle_starts_both(self, mocker):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        launchdarkly = mocker.patch.object(ff, "initialize_launchdarkly")
        posthog = mocker.patch.object(ph, "initialize_posthog_flags")

        ff.initialize_feature_flags()

        launchdarkly.assert_called_once()
        posthog.assert_called_once()

    def test_readiness_follows_launchdarkly(self, mocker, ld_client):
        """Dual serves LaunchDarkly, so LaunchDarkly is what gates a route."""
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.is_initialized.return_value = False

        assert ff._flag_backend_initialized() is False


class TestDualStaysOffTheRequestPath:
    """The shadow answer is discarded, so nobody may wait for it."""

    @pytest.mark.asyncio
    async def test_the_caller_does_not_wait_for_posthog(
        self, mocker, ld_client, user_context
    ):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        posthog_started = asyncio.Event()

        async def slow_posthog(*args, **kwargs):
            posthog_started.set()
            await asyncio.sleep(30)
            return False, True

        mocker.patch.object(ph, "evaluate_flag", side_effect=slow_posthog)

        result = await asyncio.wait_for(
            evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1"), timeout=5
        )

        assert result == (True, True)
        await posthog_started.wait()
        for task in list(ff._shadow_evaluations):
            task.cancel()

    @pytest.mark.asyncio
    async def test_both_vendors_share_one_context_lookup(self, mocker, ld_client):
        """The failed-lookup path is deliberately uncached, so evaluating the
        two vendors independently would double its database reads."""
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        stub_posthog(mocker, value=True)
        lookup = mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(Context.create("u-1"), False),
        )

        await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1")
        await drain_shadow_evaluations()

        lookup.assert_called_once_with("u-1")

    @pytest.mark.asyncio
    async def test_a_probe_backlog_is_bounded(self, mocker, ld_client, user_context):
        """A slow PostHog costs the diff week its samples, not the process
        its memory."""
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        mocker.patch.object(ff, "MAX_CONCURRENT_SHADOW_EVALUATIONS", 2)

        async def never_finishes(*args, **kwargs):
            await asyncio.sleep(30)
            return False, True

        mocker.patch.object(ph, "evaluate_flag", side_effect=never_finishes)

        for _ in range(5):
            await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1")

        assert len(ff._shadow_evaluations) == 2
        for task in list(ff._shadow_evaluations):
            task.cancel()

    @pytest.mark.asyncio
    async def test_segment_targeted_flags_are_marked_as_expected(
        self, mocker, ld_client, user_context, caplog
    ):
        """Until phase 2 builds the cohorts these disagree by construction, and
        the diff-week report has to be able to set them aside."""
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        stub_posthog(mocker, value=False)

        with caplog.at_level(
            logging.WARNING, logger="backend.util.feature_flag.mismatch"
        ):
            await evaluate_feature_flag(Flag.GRAPHITI_MEMORY, "u-1")
            await drain_shadow_evaluations()

        [message] = [
            r.getMessage()
            for r in caplog.records
            if r.name == "backend.util.feature_flag.mismatch"
        ]
        assert _mismatch_record(message)["expected_until_cohorts_exist"] is True


class TestShutdownStopsShadowEvaluations:
    """A shadow read that outlives teardown rebuilds the client it just closed:
    shutdown clears the "did we try" gate so an in-process restart works, and
    ``get_flag_client()`` then constructs a fresh one with a poller thread
    nothing will close."""

    @pytest.fixture(autouse=True)
    def restart_after(self):
        yield
        ff._shadow_evaluations_stopped = False

    @pytest.mark.asyncio
    async def test_shutdown_cancels_the_in_flight_reads(
        self, mocker, ld_client, user_context
    ):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        mocker.patch.object(ff, "shutdown_launchdarkly")
        mocker.patch.object(ph, "shutdown_posthog_flags")
        ld_client.variation.return_value = True

        async def never_finishes(*args, **kwargs):
            await asyncio.sleep(30)
            return False, True

        mocker.patch.object(ph, "evaluate_flag", side_effect=never_finishes)
        await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1")
        assert len(ff._shadow_evaluations) == 1

        ff.shutdown_feature_flags()

        assert ff._shadow_evaluations == set()

    @pytest.mark.asyncio
    async def test_a_read_during_teardown_starts_no_new_client(
        self, mocker, ld_client, user_context
    ):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        mocker.patch.object(ff, "shutdown_launchdarkly")
        mocker.patch.object(ph, "shutdown_posthog_flags")
        ld_client.variation.return_value = True
        get_client = mocker.patch.object(ph, "get_flag_client")

        ff.shutdown_feature_flags()
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, True)
        await drain_shadow_evaluations()

        get_client.assert_not_called()

    def test_a_restart_resumes_shadow_evaluation(self, mocker):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        mocker.patch.object(ff, "shutdown_launchdarkly")
        mocker.patch.object(ff, "initialize_launchdarkly")
        mocker.patch.object(ph, "shutdown_posthog_flags")
        mocker.patch.object(ph, "initialize_posthog_flags")

        ff.shutdown_feature_flags()
        ff.initialize_feature_flags()

        assert ff._shadow_evaluations_stopped is False

    def test_a_failed_launchdarkly_teardown_still_closes_posthog(self, mocker):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        mocker.patch.object(
            ff, "shutdown_launchdarkly", side_effect=RuntimeError("ld teardown")
        )
        posthog_shutdown = mocker.patch.object(ph, "shutdown_posthog_flags")

        with pytest.raises(RuntimeError, match="ld teardown"):
            ff.shutdown_feature_flags()

        posthog_shutdown.assert_called_once()


class TestForcedFlagsInEveryBackend:
    @pytest.mark.parametrize("backend", list(FeatureFlagBackend))
    @pytest.mark.asyncio
    async def test_an_env_override_answers_without_any_vendor(
        self, mocker, monkeypatch: pytest.MonkeyPatch, backend
    ):
        use_backend(mocker, backend)
        mocker.patch.object(
            ff, "get_client", side_effect=Exception("set_config was not called")
        )
        posthog = mocker.patch.object(ph, "evaluate_flag")
        monkeypatch.setenv("FORCE_FLAG_HIRE_EXPERTS", "true")

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, True)
        posthog.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_all_opens_the_route_gates_under_posthog(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        """The local override sits above the vendor choice, on every gate.

        Both gates consult it ahead of the "vendor cannot answer" bail-out, so
        selecting PostHog must not put a developer's switch back behind one.
        """
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        mocker.patch.object(ff.settings.config, "app_env", AppEnvironment.LOCAL)
        mocker.patch.object(ff, "_force_all_logged", True)
        mocker.patch.object(ph, "is_configured", return_value=False)
        mocker.patch.object(ph, "get_flag_client", return_value=None)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, True)
        await ff.create_feature_flag_dependency(Flag.HIRE_EXPERTS)("u-1")
        assert await _gated_route()(user_id="u-1") == "served"

    @pytest.mark.asyncio
    async def test_posthog_answers_the_gates_again_without_the_switch(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        use_backend(mocker, FeatureFlagBackend.POSTHOG)
        mocker.patch.object(ff.settings.config, "app_env", AppEnvironment.LOCAL)
        monkeypatch.delenv("FORCE_ALL_FLAGS", raising=False)
        monkeypatch.delenv("NEXT_PUBLIC_FORCE_ALL_FLAGS", raising=False)
        mocker.patch.object(ph, "is_configured", return_value=True)
        stub_posthog(mocker, value=True)

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, True)
        await ff.create_feature_flag_dependency(Flag.HIRE_EXPERTS)("u-1")
        assert await _gated_route()(user_id="u-1") == "served"

        stub_posthog(mocker, value=False)
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, True)
        with pytest.raises(HTTPException) as off:
            await ff.create_feature_flag_dependency(Flag.HIRE_EXPERTS)("u-1")
        assert off.value.status_code == 404


class TestSentryFlagContext:
    """Every vendor's served value lands on the scope Sentry attaches to errors."""

    @pytest.fixture
    def sentry_flags(self):
        with sentry_sdk.isolation_scope() as scope:
            # A forked scope inherits whatever earlier tests recorded.
            scope.flags.clear()
            yield scope.flags

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", list(FeatureFlagBackend))
    async def test_a_boolean_flag_is_recorded_as_served(
        self, mocker, ld_client, user_context, sentry_flags, backend
    ):
        use_backend(mocker, backend)
        ld_client.variation.return_value = True
        stub_posthog(mocker, value=backend is FeatureFlagBackend.POSTHOG)

        assert await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1") is True
        assert sentry_flags.get() == [{"flag": Flag.HIRE_EXPERTS.value, "result": True}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("initialized", [True, False])
    async def test_a_non_boolean_flag_records_nothing(
        self, ld_client, user_context, sentry_flags, initialized
    ):
        ld_client.is_initialized.return_value = initialized
        ld_client.variation.return_value = {"daily": 5}

        await ff.get_feature_flag_value("copilot-cost-limits", "u-1", {"daily": 1})
        assert sentry_flags.get() == []

    @pytest.mark.asyncio
    async def test_a_fallback_is_marked_beside_its_value(
        self, mocker, user_context, sentry_flags
    ):
        client = mocker.Mock(spec=LDClient)
        client.is_initialized.return_value = False
        mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)

        await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1")

        assert sentry_flags.get() == [
            {"flag": Flag.HIRE_EXPERTS.value, "result": False},
            {"flag": f"{Flag.HIRE_EXPERTS.value}.fallback", "result": True},
        ]

    @pytest.mark.asyncio
    async def test_a_non_boolean_answer_to_a_boolean_read_records_the_default(
        self, ld_client, user_context, sentry_flags
    ):
        ld_client.variation.return_value = "on"

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1", True) == (
            True,
            False,
        )
        assert sentry_flags.get() == [
            {"flag": Flag.HIRE_EXPERTS.value, "result": True},
            {"flag": f"{Flag.HIRE_EXPERTS.value}.fallback", "result": True},
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "gate, configured, initialized, answer",
        [
            ("decorator", True, False, True),
            ("decorator", True, True, "on"),
            ("dependency", False, True, True),
            ("dependency", True, False, True),
        ],
    )
    async def test_a_gate_serving_its_default_records_it(
        self,
        mocker,
        ld_client,
        user_context,
        sentry_flags,
        gate,
        configured,
        initialized,
        answer,
    ):
        use_backend(mocker, FeatureFlagBackend.LAUNCHDARKLY)
        mocker.patch("backend.util.feature_flag.is_configured", return_value=configured)
        ld_client.is_initialized.return_value = initialized
        ld_client.variation.return_value = answer
        # A stale earlier answer must not be what an error from this route carries.
        sentry_sdk.feature_flags.add_feature_flag(Flag.HIRE_EXPERTS.value, True)

        with pytest.raises(HTTPException):
            if gate == "decorator":
                await _gated_route()(user_id="u-1")
            else:
                await ff.create_feature_flag_dependency(Flag.HIRE_EXPERTS)("u-1")

        assert {f["flag"]: f["result"] for f in sentry_flags.get()} == {
            Flag.HIRE_EXPERTS.value: False,
            f"{Flag.HIRE_EXPERTS.value}.fallback": True,
        }

    @pytest.mark.asyncio
    async def test_a_fallback_marker_stays_off_the_span(
        self, mocker, user_context, sentry_flags
    ):
        client = mocker.Mock(spec=LDClient)
        client.is_initialized.return_value = False
        mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)
        span = Span()
        scope = sentry_sdk.get_current_scope()
        previous, scope.span = scope.span, span
        try:
            await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1")
        finally:
            scope.span = previous

        assert span._flags == {f"flag.evaluation.{Flag.HIRE_EXPERTS.value}": False}

    @pytest.mark.asyncio
    async def test_a_real_answer_clears_an_earlier_fallback(
        self, ld_client, user_context, sentry_flags
    ):
        sentry_sdk.feature_flags.add_feature_flag(
            f"{Flag.HIRE_EXPERTS.value}.fallback", True
        )
        ld_client.variation.return_value = True

        await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1")

        assert {f["flag"]: f["result"] for f in sentry_flags.get()} == {
            f"{Flag.HIRE_EXPERTS.value}.fallback": False,
            Flag.HIRE_EXPERTS.value: True,
        }

    @pytest.mark.asyncio
    async def test_a_recording_failure_does_not_break_the_read(
        self, mocker, ld_client, user_context
    ):
        mocker.patch.object(
            sentry_sdk.feature_flags,
            "add_feature_flag",
            side_effect=RuntimeError("sentry down"),
        )
        ld_client.variation.return_value = True

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, True)


class TestTheCountryRuleInEveryBackend:
    """The trial's country targeting, through each vendor's real evaluator."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", list(FeatureFlagBackend))
    @pytest.mark.parametrize(
        "country,offered", [("US", True), ("BR", True), ("IN", False), (None, False)]
    )
    async def test_the_country_decides_the_offer(
        self, mocker, backend, country, offered
    ):
        use_backend(mocker, backend)
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(Context.builder("u-1").kind("user").build(), True),
        )
        ld = _launchdarkly_serving_all_but_india()
        mocker.patch("backend.util.feature_flag.ldclient.get", return_value=ld)
        posthog_client = _posthog_serving_all_but_india()
        mocker.patch.object(ph, "get_flag_client", return_value=posthog_client)
        try:
            value = await ff.get_feature_flag_value(
                _TRIAL_FLAG,
                "u-1",
                None,
                attributes={"country": country} if country else None,
            )
            await drain_shadow_evaluations()
        finally:
            ld.close()
            posthog_client.shutdown()
        assert (value == {"version": "v1"}) is offered


_TRIAL_FLAG = "card-required-trial-offer"


def _launchdarkly_serving_all_but_india() -> LDClient:
    td = TestData.data_source()
    td.update(
        td.flag(_TRIAL_FLAG)
        .variations({"enabled": False}, {"version": "v1"})
        .fallthrough_variation(0)
        .if_not_match("country", "IN")
        .then_return(1)
    )
    return LDClient(LDConfig("sdk-test", update_processor_class=td, send_events=False))


def _posthog_serving_all_but_india() -> Posthog:
    """What the sync script ports that LaunchDarkly rule to, evaluated locally."""
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


def _no_remote_flags(*args, **kwargs):
    raise ConnectionError("remote /flags is unavailable in tests")


def _gated_route():
    """A route behind the decorator, which holds a raw flag key rather than a Flag."""

    @feature_flag(Flag.HIRE_EXPERTS.value)
    async def route(user_id: str) -> str:
        return "served"

    return route
