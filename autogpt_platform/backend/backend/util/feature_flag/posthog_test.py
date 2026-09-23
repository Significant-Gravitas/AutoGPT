"""The PostHog evaluator itself — client lifecycle and the raw read."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import backend.util.feature_flag.posthog as ph


@pytest.fixture(autouse=True)
def fresh_client(mocker):
    """The singleton is module state; each test starts without one."""
    mocker.patch.object(ph, "_client", None)
    mocker.patch.object(ph, "_init_attempted", False)
    mocker.patch.object(ph, "_shut_down", False)


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
    async def test_a_conclusive_off_wins_over_a_stale_payload(self, mocker):
        """Serving the payload here would make an authoritative "off" read as
        a non-boolean, i.e. as "could not evaluate"."""
        client = mocker.Mock()
        client.evaluate_flags.return_value = snapshot(
            mocker, value=False, payload={"daily": 5}
        )
        mocker.patch.object(ph, "get_flag_client", return_value=client)

        assert await ph.evaluate_flag("hire-experts", "u-1") == (False, True)

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
            return snapshot(mocker, value=True)

        client = mocker.Mock()
        client.evaluate_flags.side_effect = blocking_read
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
