"""The shared flag-definition cache: election, failure modes, and the SDK wiring."""

import json
import logging

import pytest
from ldclient import Context, LDClient
from posthog import Posthog
from posthog.request import GetResponse

import backend.util.feature_flag as ff
import backend.util.feature_flag_definition_cache as cache
import backend.util.feature_flag_posthog as ph
from backend.util.feature_flag import Flag, evaluate_feature_flag
from backend.util.settings import FeatureFlagBackend, FlagDefinitionCacheBackend

REFRESH = 30
LOCK_TTL = REFRESH * cache._LOCK_TTL_POLLS


class FakeRedis:
    """The four commands the provider uses, over a clock the test controls."""

    def __init__(self):
        self.now = 1_000.0
        self.fail = False
        self._values: dict[str, tuple[str, float | None]] = {}

    def set(self, key, value, nx=False, px=None, ex=None):
        self._check()
        if nx and key in self._values:
            return None
        ttl = px / 1000 if px else (float(ex) if ex else None)
        self._values[key] = (value, self.now + ttl if ttl else None)
        return True

    def get(self, key):
        self._check()
        entry = self._values.get(key)
        return entry[0] if entry else None

    def eval(self, script, numkeys, key, holder, *args):
        self._check()
        entry = self._values.get(key)
        if not entry or entry[0] != holder:
            return 0
        if script is cache._RENEW_LOCK:
            self._values[key] = (entry[0], self.now + int(args[0]) / 1000)
            return 1
        assert script is cache._RELEASE_LOCK
        del self._values[key]
        return 1

    def advance(self, seconds: float):
        self.now += seconds

    def _check(self):
        if self.fail:
            raise ConnectionError("redis is down")
        for key, (_, expires) in list(self._values.items()):
            if expires is not None and expires <= self.now:
                del self._values[key]


DEFINITIONS = {
    "flags": [
        {
            "id": 1,
            "team_id": 1,
            "name": "",
            "key": "shared-flag",
            "active": True,
            "deleted": False,
            "ensure_experience_continuity": False,
            "filters": {"groups": [{"properties": [], "rollout_percentage": 100}]},
        }
    ],
    "group_type_mapping": {},
    "cohorts": {},
}


@pytest.fixture
def redis():
    return FakeRedis()


@pytest.fixture
def logs():
    """`caplog` captures nothing for `backend.*` under the app's logging config."""
    captured: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = Capture()
    cache.logger.addHandler(handler)
    previous, cache.logger.level = cache.logger.level, logging.DEBUG
    yield captured
    cache.logger.removeHandler(handler)
    cache.logger.level = previous


def provider(redis, *, ttl=600) -> cache.RedisFlagDefinitionCache:
    return cache.RedisFlagDefinitionCache(
        refresh_interval=REFRESH, ttl=ttl, redis_factory=lambda: redis
    )


def store_definitions(redis, *, age_seconds=0.0, definitions=None):
    """Write the shared copy as the refresher would have, `age_seconds` ago."""
    import time

    redis.set(
        cache._DATA_KEY,
        json.dumps(
            {
                "fetched_at": time.time() - age_seconds,
                "definitions": definitions if definitions is not None else DEFINITIONS,
            }
        ),
    )


class TestElection:
    def test_exactly_one_process_refreshes(self, redis):
        replicas = [provider(redis) for _ in range(5)]

        assert sum(p.should_fetch_flag_definitions() for p in replicas) == 1

    def test_the_refresher_keeps_the_job_across_polls(self, redis):
        leader, follower = provider(redis), provider(redis)
        assert leader.should_fetch_flag_definitions() is True

        for _ in range(4):
            redis.advance(REFRESH)
            assert follower.should_fetch_flag_definitions() is False
            assert leader.should_fetch_flag_definitions() is True

    def test_followers_read_what_the_refresher_wrote(self, redis):
        leader, follower = provider(redis), provider(redis)
        leader.should_fetch_flag_definitions()
        leader.on_flag_definitions_received(DEFINITIONS)

        assert follower.should_fetch_flag_definitions() is False
        assert follower.get_flag_definitions() == DEFINITIONS

    def test_a_dead_refresher_hands_over_once_its_lock_expires(self, redis):
        leader, standby = provider(redis), provider(redis)
        leader.should_fetch_flag_definitions()
        assert standby.should_fetch_flag_definitions() is False

        redis.advance(LOCK_TTL + 1)  # the leader stopped renewing

        assert standby.should_fetch_flag_definitions() is True

    def test_a_graceful_shutdown_hands_over_at_once(self, redis):
        leader, standby = provider(redis), provider(redis)
        leader.should_fetch_flag_definitions()

        leader.shutdown()

        assert standby.should_fetch_flag_definitions() is True

    def test_a_follower_shutting_down_leaves_the_lock_alone(self, redis):
        leader, follower = provider(redis), provider(redis)
        leader.should_fetch_flag_definitions()
        follower.should_fetch_flag_definitions()

        follower.shutdown()

        assert provider(redis).should_fetch_flag_definitions() is False
        assert leader.should_fetch_flag_definitions() is True

    def test_becoming_the_refresher_says_so(self, redis, logs):
        provider(redis).should_fetch_flag_definitions()

        assert any("now refreshes the shared PostHog flag" in m for m in logs)


class TestFailureModes:
    def test_an_empty_cache_reads_as_a_miss(self, redis):
        assert provider(redis).get_flag_definitions() is None

    @pytest.mark.parametrize(
        "call",
        [
            lambda p: p.get_flag_definitions(),
            lambda p: p.on_flag_definitions_received(DEFINITIONS),
            lambda p: p.shutdown(),
        ],
        ids=["read", "write", "shutdown"],
    )
    def test_redis_being_down_never_raises(self, redis, call):
        p = provider(redis)
        p.should_fetch_flag_definitions()
        redis.fail = True

        assert call(p) is None

    def test_redis_being_down_falls_back_to_fetching(self, redis):
        redis.fail = True

        assert provider(redis).should_fetch_flag_definitions() is True

    def test_redis_being_down_is_logged_once(self, redis, logs):
        redis.fail = True
        p = provider(redis)

        for _ in range(10):
            p.should_fetch_flag_definitions()

        assert sum("Redis unavailable" in m for m in logs) == 1

    def test_definitions_past_the_stale_window_are_served_with_a_warning(
        self, redis, logs
    ):
        store_definitions(redis, age_seconds=REFRESH * cache._STALE_AFTER_POLLS + 60)

        assert provider(redis).get_flag_definitions() == DEFINITIONS
        assert any("are " in m and "s old" in m for m in logs)

    def test_fresh_definitions_come_with_no_warning(self, redis, logs):
        store_definitions(redis, age_seconds=1)

        assert provider(redis).get_flag_definitions() == DEFINITIONS
        assert not any("s old" in m for m in logs)

    @pytest.mark.parametrize(
        "content", ["not json", json.dumps({"definitions": DEFINITIONS})]
    )
    def test_unreadable_contents_are_discarded(self, redis, content):
        redis.set(cache._DATA_KEY, content)

        assert provider(redis).get_flag_definitions() is None

    def test_the_shared_copy_expires(self, redis):
        p = provider(redis, ttl=600)
        p.on_flag_definitions_received(DEFINITIONS)

        redis.advance(601)

        assert p.get_flag_definitions() is None


class TestMemoryCache:
    def test_it_refreshes_once_then_serves_itself(self):
        p = cache.MemoryFlagDefinitionCache(refresh_interval=REFRESH)

        assert p.should_fetch_flag_definitions() is True
        p.on_flag_definitions_received(DEFINITIONS)

        assert p.should_fetch_flag_definitions() is False
        assert p.get_flag_definitions() == DEFINITIONS


class TestSelection:
    @pytest.mark.parametrize(
        "backend, expected",
        [
            (FlagDefinitionCacheBackend.REDIS, cache.RedisFlagDefinitionCache),
            (FlagDefinitionCacheBackend.MEMORY, cache.MemoryFlagDefinitionCache),
            (FlagDefinitionCacheBackend.NONE, type(None)),
        ],
    )
    def test_the_setting_picks_the_provider(self, mocker, backend, expected):
        mocker.patch.object(
            cache.settings.config, "posthog_flag_definition_cache", backend
        )

        assert isinstance(cache.get_flag_definition_cache(), expected)

    def test_redis_is_the_default(self):
        assert (
            cache.settings.config.posthog_flag_definition_cache
            is FlagDefinitionCacheBackend.REDIS
        )


class TestClientWiring:
    """What the PostHog client is actually built with."""

    @pytest.fixture(autouse=True)
    def fresh_client(self, mocker):
        mocker.patch.object(ph, "_client", None)
        mocker.patch.object(ph, "_init_attempted", False)
        mocker.patch.object(ph.settings.secrets, "posthog_api_key", "phc_test")
        mocker.patch.object(
            ph.settings.secrets, "posthog_personal_api_key", "phx_personal"
        )

    def test_the_client_shares_its_definitions_through_the_cache(self, mocker):
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()

        _, kwargs = posthog.call_args
        assert isinstance(
            kwargs["flag_definition_cache_provider"], cache.RedisFlagDefinitionCache
        )
        assert kwargs["poll_interval"] == REFRESH

    def test_no_personal_key_means_no_poller_to_share(self, mocker):
        mocker.patch.object(ph.settings.secrets, "posthog_personal_api_key", "")
        posthog = mocker.patch.object(ph, "Posthog")

        ph.get_flag_client()

        _, kwargs = posthog.call_args
        assert kwargs["flag_definition_cache_provider"] is None
        assert kwargs["enable_local_evaluation"] is False


class TestDefaultBackendIsUntouched:
    """`launchdarkly` is the default, and it reaches none of this."""

    @pytest.fixture
    def ld_client(self, mocker):
        client = mocker.Mock(spec=LDClient)
        mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)
        client.is_initialized.return_value = True
        return client

    @pytest.fixture
    def user_context(self, mocker):
        context = Context.builder("u-1").kind("user").anonymous(False).build()
        return mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(context, True),
        )

    def test_the_default_flag_backend_is_launchdarkly(self):
        assert (
            ff.settings.config.feature_flag_backend is FeatureFlagBackend.LAUNCHDARKLY
        )

    @pytest.mark.asyncio
    async def test_the_default_builds_no_definition_cache(
        self, mocker, ld_client, user_context
    ):
        build = mocker.patch.object(cache, "get_flag_definition_cache")
        ld_client.variation.return_value = True

        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, True)
        build.assert_not_called()


class TestAgainstTheRealSDK:
    """The SDK's own poll, driven through the provider."""

    @pytest.fixture
    def fetches(self, mocker):
        """Every definitions fetch the SDK makes, with one canned answer."""
        return mocker.patch(
            "posthog.client.get",
            return_value=GetResponse(data=DEFINITIONS, etag="e1"),
        )

    def client(self, provider_impl) -> Posthog:
        return Posthog(
            "phc_test",
            personal_api_key="phx_personal",
            enable_local_evaluation=True,
            sync_mode=True,
            poll_interval=REFRESH,
            flag_definition_cache_provider=provider_impl,
        )

    def test_a_follower_evaluates_without_fetching_definitions(self, redis, fetches):
        provider(redis).should_fetch_flag_definitions()  # someone else leads
        store_definitions(redis)
        follower = self.client(provider(redis))

        follower._load_feature_flags()  # what the SDK's poller thread runs

        fetches.assert_not_called()
        snapshot = follower.evaluate_flags(
            "u-1", flag_keys=["shared-flag"], only_evaluate_locally=True
        )
        assert snapshot.get_flag("shared-flag") is True

    def test_the_refresher_fetches_and_shares_what_it_got(self, redis, fetches):
        leader = self.client(provider(redis))

        leader._load_feature_flags()

        fetches.assert_called_once()
        shared = provider(redis).get_flag_definitions()
        assert shared is not None
        assert shared["flags"] == DEFINITIONS["flags"]

    def test_an_empty_cache_falls_back_to_one_direct_fetch(self, redis, fetches):
        provider(redis).should_fetch_flag_definitions()  # someone else leads
        follower = self.client(provider(redis))  # and has written nothing yet

        follower._load_feature_flags()

        fetches.assert_called_once()

    def test_a_failed_fetch_keeps_the_previous_definitions(self, redis, fetches):
        leader = self.client(provider(redis))
        leader._load_feature_flags()
        fetches.side_effect = ConnectionError("PostHog is unreachable")

        leader._load_feature_flags()

        snapshot = leader.evaluate_flags(
            "u-1", flag_keys=["shared-flag"], only_evaluate_locally=True
        )
        assert snapshot.get_flag("shared-flag") is True

    def test_redis_being_down_leaves_every_process_fetching_for_itself(
        self, redis, fetches
    ):
        redis.fail = True
        follower = self.client(provider(redis))

        follower._load_feature_flags()

        fetches.assert_called_once()


class TestMetrics:
    def test_a_cache_read_is_counted(self, redis):
        from prometheus_client import REGISTRY

        name = "autogpt_posthog_flag_definition_cache_events_total"
        before = REGISTRY.get_sample_value(name, {"outcome": "cached"}) or 0
        store_definitions(redis)

        provider(redis).get_flag_definitions()

        assert REGISTRY.get_sample_value(name, {"outcome": "cached"}) == before + 1
