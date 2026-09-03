"""Backend selection: launchdarkly (default), posthog, and the dual diff run."""

import json
import logging
import uuid

import pytest
from ldclient import Context, LDClient

import backend.util.feature_flag as ff
import backend.util.feature_flag_posthog as ph
from backend.util.feature_flag import Flag, evaluate_feature_flag, is_feature_enabled
from backend.util.settings import FeatureFlagBackend


@pytest.fixture(autouse=True)
def no_env_override(monkeypatch: pytest.MonkeyPatch):
    """`.env` may force flags; pin the flag under test to the vendors."""
    monkeypatch.delenv("FORCE_FLAG_HIRE_EXPERTS", raising=False)
    monkeypatch.delenv("NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS", raising=False)


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
        flag silently evaluates against a user without them."""
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
            "email": "x@agpt.co",
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
    async def test_a_posthog_failure_cannot_break_the_read(
        self, mocker, ld_client, user_context
    ):
        use_backend(mocker, FeatureFlagBackend.DUAL)
        ld_client.variation.return_value = True
        mocker.patch.object(ph, "evaluate_flag", side_effect=Exception("boom"))

        assert await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1") is True

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
