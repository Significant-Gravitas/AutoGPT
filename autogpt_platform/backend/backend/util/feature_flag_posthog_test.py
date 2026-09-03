"""The PostHog evaluator itself — client lifecycle and the raw read."""

import pytest

import backend.util.feature_flag_posthog as ph


@pytest.fixture(autouse=True)
def fresh_client(mocker):
    """The singleton is module state; each test starts without one."""
    mocker.patch.object(ph, "_client", None)
    mocker.patch.object(ph, "_init_attempted", False)


def configure(mocker, *, api_key="phc_test", personal_api_key=""):
    mocker.patch.object(ph.settings.secrets, "posthog_api_key", api_key)
    mocker.patch.object(
        ph.settings.secrets, "posthog_personal_api_key", personal_api_key
    )


def snapshot(mocker, *, value, payload=None):
    result = mocker.Mock()
    result.get_flag.return_value = value
    result.get_flag_payload.return_value = payload
    return result


class TestClientConstruction:
    def test_no_project_key_means_no_client(self, mocker):
        configure(mocker, api_key="")
        assert ph.get_flag_client() is None

    def test_the_personal_key_turns_on_local_evaluation(self, mocker):
        configure(mocker, personal_api_key="phx_personal")
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()

        _, kwargs = posthog.call_args
        assert kwargs["personal_api_key"] == "phx_personal"
        assert kwargs["enable_local_evaluation"] is True

    def test_without_it_evaluation_stays_remote(self, mocker):
        configure(mocker)
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()

        _, kwargs = posthog.call_args
        assert kwargs["personal_api_key"] is None
        assert kwargs["enable_local_evaluation"] is False

    def test_the_client_is_built_once(self, mocker):
        configure(mocker)
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()
        ph.get_flag_client()

        posthog.assert_called_once()

    def test_an_unconfigured_deployment_does_not_retry_forever(self, mocker):
        """Same reason LaunchDarkly gates on "did we try": a warning and a
        construction attempt per flag read on deployments shipping without
        PostHog."""
        configure(mocker, api_key="")
        warn = mocker.patch.object(ph.logger, "warning")

        ph.get_flag_client()
        ph.get_flag_client()

        assert warn.call_count == 1

    def test_shutdown_releases_the_singleton(self, mocker):
        configure(mocker)
        client = mocker.patch.object(ph, "Posthog").return_value

        ph.get_flag_client()
        ph.shutdown_posthog_flags()

        client.shutdown.assert_called_once()
        assert ph._client is None

    def test_shutdown_without_a_client_is_a_no_op(self, mocker):
        ph.shutdown_posthog_flags()


class TestRawRead:
    @pytest.mark.asyncio
    async def test_an_unconfigured_client_returns_the_default(self, mocker):
        mocker.patch.object(ph, "get_flag_client", return_value=None)

        assert await ph.evaluate_flag("hire-experts", "u-1", default=True) == (
            True,
            False,
        )

    @pytest.mark.asyncio
    async def test_a_resolved_flag_is_evaluated(self, mocker):
        client = mocker.Mock()
        client.evaluate_flags.return_value = snapshot(mocker, value=False)
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("hire-experts", "u-1") == (False, True)

    @pytest.mark.asyncio
    async def test_an_unresolved_flag_is_not(self, mocker):
        client = mocker.Mock()
        client.evaluate_flags.return_value = snapshot(mocker, value=None)
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("hire-experts", "u-1", default=True) == (
            True,
            False,
        )

    @pytest.mark.asyncio
    async def test_a_payload_wins_over_the_flag_value(self, mocker):
        client = mocker.Mock()
        client.evaluate_flags.return_value = snapshot(
            mocker, value=True, payload={"daily": 5}
        )
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("copilot-cost-limits", "system") == (
            {"daily": 5},
            True,
        )

    @pytest.mark.asyncio
    async def test_a_variant_key_is_returned_as_the_value(self, mocker):
        client = mocker.Mock()
        client.evaluate_flags.return_value = snapshot(mocker, value="control")
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("stripe-product-id-topup", "u-1") == (
            "control",
            True,
        )

    @pytest.mark.asyncio
    async def test_the_read_is_scoped_to_one_flag(self, mocker):
        client = mocker.Mock()
        client.evaluate_flags.return_value = snapshot(mocker, value=True)
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        await ph.evaluate_flag("hire-experts", "u-1", {"role": "admin"})

        args, kwargs = client.evaluate_flags.call_args
        assert args[0] == "u-1"
        assert kwargs["flag_keys"] == ["hire-experts"]
        assert kwargs["person_properties"] == {"role": "admin"}

    @pytest.mark.asyncio
    async def test_a_failed_evaluation_returns_the_default(self, mocker):
        client = mocker.Mock()
        client.evaluate_flags.side_effect = Exception("connection refused")
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("hire-experts", "u-1", default=True) == (
            True,
            False,
        )
