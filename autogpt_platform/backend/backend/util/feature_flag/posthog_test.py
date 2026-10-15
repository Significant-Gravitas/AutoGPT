"""The PostHog evaluator itself — client lifecycle and the raw read."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from posthog import Posthog
from posthog.types import FeatureFlagResult

import backend.util.feature_flag.posthog as ph

# Taken at import, before any fixture can stub it: the test below needs the
# SDK's own capture path to show that send=False keeps events off the wire.
_REAL_CAPTURE = Posthog.capture


@pytest.fixture(autouse=True)
def fresh_client(mocker):
    """The singleton is module state; each test starts without one."""
    mocker.patch.object(ph, "_client", None)
    mocker.patch.object(ph, "_init_attempted", False)
    mocker.patch.object(ph, "_shut_down", False)


def configure(mocker, *, api_key="phc_test", secret_key="", flags_key=""):
    mocker.patch.object(ph.settings.secrets, "posthog_api_key", api_key)
    mocker.patch.object(ph.settings.secrets, "posthog_secret_key", secret_key)
    mocker.patch.object(ph.settings.secrets, "posthog_flags_api_key", flags_key)


def answer(*, value, payload=None):
    return FeatureFlagResult.from_value_and_payload("flag", value, payload)


class TestClientConstruction:
    def test_no_project_key_means_no_client(self, mocker):
        configure(mocker, api_key="")
        assert ph.get_flag_client() is None

    def test_the_secret_key_turns_on_local_evaluation(self, mocker):
        configure(mocker, secret_key="phx_personal")
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()

        _, kwargs = posthog.call_args
        assert kwargs["secret_key"] == "phx_personal"
        assert kwargs["enable_local_evaluation"] is True

    def test_without_it_evaluation_stays_remote(self, mocker):
        configure(mocker)
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()

        _, kwargs = posthog.call_args
        assert kwargs["secret_key"] is None
        assert kwargs["enable_local_evaluation"] is False

    def test_the_client_is_built_once(self, mocker):
        configure(mocker)
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()
        ph.get_flag_client()

        posthog.assert_called_once()

    def test_concurrent_first_reads_build_one_client(self, mocker):
        """The executor reads flags from two loops on two threads; a second
        client would leak a poller thread nothing closes."""
        configure(mocker)

        def slow_construction(*args, **kwargs):
            time.sleep(0.05)
            return mocker.Mock()

        posthog = mocker.patch.object(ph, "Posthog", side_effect=slow_construction)
        start = threading.Barrier(8)

        def first_read():
            start.wait()
            return ph.get_flag_client()

        with ThreadPoolExecutor(max_workers=8) as pool:
            clients = list(pool.map(lambda _: first_read(), range(8)))

        posthog.assert_called_once()
        assert all(client is clients[0] for client in clients)

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

    def test_the_client_can_be_rebuilt_after_shutdown(self, mocker):
        """An in-process restart — SpinTestServer spins the app repeatedly —
        must not leave every flag read answering with its default forever."""
        configure(mocker)
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()
        ph.shutdown_posthog_flags()
        ph.initialize_posthog_flags()

        assert ph.get_flag_client() is not None
        assert posthog.call_count == 2

    def test_a_late_read_after_shutdown_does_not_rebuild(self, mocker):
        """A shadow read runs in a worker thread that outlives the cancelled
        task; a client built there would start a poller nothing closes."""
        configure(mocker)
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()
        ph.shutdown_posthog_flags()

        assert ph.get_flag_client() is None
        assert posthog.call_count == 1

    def test_an_unconfigured_first_attempt_does_not_latch_the_gate(self, mocker):
        """Shutdown clears "did we try" even though it built no client, or
        credentials arriving later in the process can never be picked up."""
        configure(mocker, api_key="")
        mocker.patch.object(ph, "Posthog")
        assert ph.get_flag_client() is None

        ph.shutdown_posthog_flags()
        configure(mocker)
        ph.initialize_posthog_flags()

        assert ph.get_flag_client() is not None


class TestTheFlagProject:
    """Dev reads prod's flag definitions while its analytics stay in dev."""

    def test_without_a_flags_key_flags_come_from_the_analytics_project(self, mocker):
        configure(mocker, api_key="phc_analytics")
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()

        args, kwargs = posthog.call_args
        assert args[0] == "phc_analytics"
        assert kwargs["send"] is True
        assert ph.reads_another_project() is False

    def test_the_same_key_twice_is_one_project(self, mocker):
        configure(mocker, api_key="phc_prod", flags_key="phc_prod")
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()

        args, kwargs = posthog.call_args
        assert args[0] == "phc_prod"
        assert kwargs["send"] is True

    def test_a_flags_key_reads_that_project_and_sends_it_nothing(self, mocker):
        configure(
            mocker, api_key="phc_dev", flags_key="phc_prod", secret_key="phs_prod"
        )
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()

        args, kwargs = posthog.call_args
        assert args[0] == "phc_prod"
        assert kwargs["secret_key"] == "phs_prod"
        assert kwargs["send"] is False

    def test_a_flags_key_alone_configures_flag_reads(self, mocker):
        configure(mocker, api_key="", flags_key="phc_prod")

        assert ph.is_configured() is True
        assert ph.reads_another_project() is True

    def test_a_separate_project_without_the_secret_key_warns(self, mocker):
        configure(mocker, api_key="phc_dev", flags_key="phc_prod")
        mocker.patch.object(ph, "Posthog")
        warn = mocker.patch.object(ph.logger, "warning")

        ph.get_flag_client()

        assert "POSTHOG_SECRET_KEY" in warn.call_args.args[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flags_key,only_locally", [("", False), ("phc_test", False), ("phc_prod", True)]
    )
    async def test_only_a_separate_project_is_read_locally_only(
        self, mocker, flags_key, only_locally
    ):
        configure(mocker, flags_key=flags_key)
        client = mocker.Mock()
        client.get_feature_flag_result.return_value = answer(value=True)
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        await ph.evaluate_flag("hire-experts", "u-1")

        _, kwargs = client.get_feature_flag_result.call_args
        assert kwargs["only_evaluate_locally"] is only_locally


class TestASeparateFlagProjectThroughTheRealSDK:
    """What the installed posthog-python does with the client built above."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flags_key,asks_remotely", [("", True), ("phc_prod", False)]
    )
    async def test_a_flag_missing_locally_goes_remote_only_within_one_project(
        self, mocker, flags_key, asks_remotely
    ):
        """The remote fallback would hand the flag project a user's email."""
        # The flag-called events are not what this test is about.
        mocker.patch.object(Posthog, "capture")
        client, remote = _real_client(mocker, flags_key=flags_key)
        try:
            value = await ph.evaluate_flag(
                "not-in-the-definitions", "u-1", {"email": "x@agpt.co"}
            )
        finally:
            ph.shutdown_posthog_flags()

        assert value == (None, False)
        assert remote.called is asks_remotely

    @pytest.mark.asyncio
    async def test_reads_from_a_separate_project_send_it_no_events(self, mocker):
        mocker.patch.object(Posthog, "capture", _REAL_CAPTURE)
        captured = mocker.spy(Posthog, "capture")
        uploads = [
            mocker.patch("posthog.consumer.batch_post"),
            mocker.patch("posthog.consumer._send_v1_batch"),
        ]
        client, remote = _real_client(mocker, flags_key="phc_prod")
        try:
            # Exposure reads, the only ones that form $feature_flag_called.
            assert await ph.evaluate_flag("hire-experts", "u-1", exposure=True) == (
                True,
                True,
            )
            assert await ph.evaluate_flag(
                "not-in-the-definitions", "u-1", exposure=True
            ) == (None, False)
            client.flush()
        finally:
            ph.shutdown_posthog_flags()

        # The SDK still forms the events; none is uploaded.
        assert captured.called
        assert not remote.called
        assert not any(upload.called for upload in uploads)


class TestInMemoryRead:
    def test_definitions_not_loaded_yet_are_never_fetched(self, mocker):
        """Fetching them first is the SDK's default, and a network call per read."""
        client = Posthog(
            "phc-test", secret_key="phx-test", enable_local_evaluation=False
        )
        fetch = mocker.patch.object(client, "load_feature_flags")
        remote = mocker.patch.object(client, "_get_flags_decision")
        mocker.patch.object(ph, "get_flag_client", return_value=client)
        try:
            values = ph.evaluate_flags_in_memory(["hire-experts"], "visitor:v-1", {})
        finally:
            client.shutdown()

        assert (values, fetch.call_count, remote.call_count) == ({}, 0, 0)

    def test_a_matched_payload_is_served_decoded(self, mocker):
        client = Posthog(
            "phc-test", secret_key="phx-test", enable_local_evaluation=False
        )
        client.feature_flags = [
            _rolled_out("marketplace-search-terms", 100, payload='["Research"]'),
            _rolled_out("hire-experts", 100),
            _rolled_out("chat-search", 0, payload='["unused"]'),
        ]
        mocker.patch.object(ph, "get_flag_client", return_value=client)
        try:
            values = ph.evaluate_flags_in_memory(
                ["marketplace-search-terms", "hire-experts", "chat-search"],
                "visitor:v-1",
                {},
            )
        finally:
            client.shutdown()

        assert values == {
            "marketplace-search-terms": ["Research"],
            "hire-experts": True,
            "chat-search": False,
        }


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
        client.get_feature_flag_result.return_value = answer(value=False)
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("hire-experts", "u-1") == (False, True)

    @pytest.mark.asyncio
    async def test_an_unresolved_flag_is_not(self, mocker):
        client = mocker.Mock()
        client.get_feature_flag_result.return_value = answer(value=None)
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("hire-experts", "u-1", default=True) == (
            True,
            False,
        )

    @pytest.mark.asyncio
    async def test_a_payload_wins_over_the_flag_value(self, mocker):
        client = mocker.Mock()
        client.get_feature_flag_result.return_value = answer(
            value=True, payload={"daily": 5}
        )
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("copilot-cost-limits", "system") == (
            {"daily": 5},
            True,
        )

    @pytest.mark.asyncio
    async def test_a_conclusive_off_wins_over_a_stale_payload(self, mocker):
        """Serving the payload here would make an authoritative "off" read as
        a non-boolean, i.e. as "could not evaluate"."""
        client = mocker.Mock()
        client.get_feature_flag_result.return_value = answer(
            value=False, payload={"daily": 5}
        )
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("hire-experts", "u-1") == (False, True)

    @pytest.mark.asyncio
    async def test_a_variant_key_is_returned_as_the_value(self, mocker):
        client = mocker.Mock()
        client.get_feature_flag_result.return_value = answer(value="control")
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("stripe-product-id-topup", "u-1") == (
            "control",
            True,
        )

    @pytest.mark.asyncio
    async def test_the_read_is_scoped_to_one_flag(self, mocker):
        client = mocker.Mock()
        client.get_feature_flag_result.return_value = answer(value=True)
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        await ph.evaluate_flag("hire-experts", "u-1", {"role": "admin"})

        args, kwargs = client.get_feature_flag_result.call_args
        assert args == ("hire-experts", "u-1")
        assert kwargs["person_properties"] == {"role": "admin"}

    @pytest.mark.asyncio
    async def test_a_failed_evaluation_returns_the_default(self, mocker):
        client = mocker.Mock()
        client.get_feature_flag_result.side_effect = Exception("connection refused")
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("hire-experts", "u-1", default=True) == (
            True,
            False,
        )

    def test_blocked_reads_leave_the_default_executor_free(self, mocker):
        """Remote reads that hang must not starve other ``to_thread`` work."""
        release = threading.Event()
        entered = 0
        lock = threading.Lock()

        def blocking_read(*args, **kwargs):
            nonlocal entered
            with lock:
                entered += 1
            release.wait(10)
            return answer(value=True)

        client = mocker.Mock()
        client.get_feature_flag_result.side_effect = blocking_read
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        async def scenario():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(
                ThreadPoolExecutor(max_workers=ph.FLAG_READ_WORKERS)
            )
            reads = [
                asyncio.create_task(ph.evaluate_flag("hire-experts", f"u-{i}"))
                for i in range(ph.FLAG_READ_WORKERS + 1)
            ]
            try:
                deadline = time.monotonic() + 5
                while entered < ph.FLAG_READ_WORKERS:
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.01)
                return await asyncio.wait_for(asyncio.to_thread(lambda: "probe"), 2)
            finally:
                release.set()
                await asyncio.gather(*reads)
                await loop.shutdown_default_executor()

        # A private loop, so swapping its default executor touches no other test.
        loop = asyncio.new_event_loop()
        try:
            assert loop.run_until_complete(scenario()) == "probe"
        finally:
            loop.close()


def _real_client(mocker, *, flags_key: str):
    """A real flag client holding one rolled-out flag, with /flags recorded."""
    configure(mocker, api_key="phc_dev", flags_key=flags_key, secret_key="phs_test")
    mocker.patch.object(ph, "get_flag_definition_cache", return_value=None)
    client = ph.get_flag_client()
    assert client is not None
    client.feature_flags = [
        {
            "id": 1,
            "key": "hire-experts",
            "active": True,
            "filters": {"groups": [{"properties": [], "rollout_percentage": 100}]},
        }
    ]
    remote = mocker.patch.object(
        client, "_get_flags_decision", side_effect=ConnectionError("offline")
    )
    return client, remote


def _rolled_out(key: str, percentage: int, payload: str | None = None) -> dict:
    filters: dict = {"groups": [{"properties": [], "rollout_percentage": percentage}]}
    if payload is not None:
        filters["payloads"] = {"true": payload}
    return {"id": 1, "key": key, "active": True, "filters": filters}
