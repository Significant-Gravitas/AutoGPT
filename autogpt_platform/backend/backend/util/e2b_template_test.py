import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b.api.client.models import TemplateAliasResponse, TemplateBuildStatus

from backend.copilot.config import ChatConfig
from backend.util import e2b_template
from backend.util.e2b_template import (
    DESKTOP_IMAGE,
    TemplateSpec,
    TemplateState,
    ensure_template,
    forget_ready_templates,
    get_template_state,
)

_M = "backend.util.e2b_template"
_KEY = "e2b_test"
_OTHER_KEY = "e2b_other_team"


@pytest.fixture(autouse=True)
def _fresh_cache():
    forget_ready_templates()
    yield
    forget_ready_templates()


def _redis(lock_acquired: bool, lock_present: bool = True) -> MagicMock:
    redis = MagicMock()
    redis.set = AsyncMock(return_value=lock_acquired)
    redis.eval = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=lock_present)
    return redis


def _states(*states: TemplateState) -> AsyncMock:
    return AsyncMock(side_effect=list(states))


class TestSpec:
    def test_the_managed_image_is_the_copilot_default(self):
        assert ChatConfig().e2b_sandbox_template == DESKTOP_IMAGE.alias
        assert DESKTOP_IMAGE.cpu_count == 1 and DESKTOP_IMAGE.memory_mb == 2048

    def test_tags_are_docker_safe_and_resolve_the_bare_alias(self):
        tags = TemplateSpec(
            alias="x", source="desktop", cpu_count=2, memory_mb=4096
        ).tags
        assert tags[0] == "default"
        assert "2x4" in tags and "from-desktop" in tags
        assert all(":" not in t for t in tags)


class TestEnsureTemplate:
    @pytest.mark.asyncio
    async def test_unmanaged_template_is_left_alone(self):
        with patch(f"{_M}.get_template_state", AsyncMock()) as state:
            await ensure_template("base", _KEY)
        state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ready_template_is_checked_once_per_team_per_process(self):
        with patch(
            f"{_M}.get_template_state", _states(*[TemplateState.READY] * 2)
        ) as st:
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
            await ensure_template(DESKTOP_IMAGE.alias, _OTHER_KEY)
        # Same alias, other team: the alias may not exist there, so check again.
        assert st.await_count == 2
        assert st.await_args_list[1].args == (DESKTOP_IMAGE, _OTHER_KEY)

    @pytest.mark.asyncio
    async def test_missing_template_is_built_at_the_managed_size(self):
        redis = _redis(lock_acquired=True)
        with (
            patch(f"{_M}.get_template_state", _states(*[TemplateState.MISSING] * 2)),
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.Template") as template_cls,
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
        ):
            tpl.build = AsyncMock(return_value=MagicMock(template_id="t1"))
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)

        template_cls.return_value.from_template.assert_called_once_with("desktop")
        kwargs = tpl.build.await_args.kwargs
        assert tpl.build.await_args.args[1] == DESKTOP_IMAGE.alias
        assert kwargs["cpu_count"] == 1 and kwargs["memory_mb"] == 2048
        assert kwargs["tags"][0] == "default" and kwargs["api_key"] == _KEY
        assert f"{DESKTOP_IMAGE.alias}@" in next(iter(e2b_template._ready))

    @pytest.mark.asyncio
    async def test_lock_is_scoped_to_the_team_and_released_by_token(self):
        redis = _redis(lock_acquired=True)
        with (
            patch(f"{_M}.get_template_state", _states(*[TemplateState.MISSING] * 4)),
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.Template"),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
        ):
            tpl.build = AsyncMock(return_value=MagicMock(template_id="t1"))
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
            await ensure_template(DESKTOP_IMAGE.alias, _OTHER_KEY)

        (key_a, token_a), (key_b, token_b) = [
            c.args[:2] for c in redis.set.await_args_list
        ]
        assert key_a != key_b and _KEY not in key_a and _OTHER_KEY not in key_b
        assert redis.set.await_args_list[0].kwargs == {"nx": True, "ex": 300}
        # Compare-and-delete with the token this caller wrote, never a bare DEL.
        released = [c.args for c in redis.eval.await_args_list]
        assert released == [
            (e2b_template._UNLOCK_SCRIPT, 1, key_a, token_a),
            (e2b_template._UNLOCK_SCRIPT, 1, key_b, token_b),
        ]

    @pytest.mark.asyncio
    async def test_lock_winner_rechecks_before_building(self):
        redis = _redis(lock_acquired=True)
        with (
            patch(
                f"{_M}.get_template_state",
                _states(TemplateState.MISSING, TemplateState.READY),
            ),
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
        ):
            tpl.build = AsyncMock()
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        tpl.build.assert_not_awaited()
        redis.eval.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lock_winner_waits_for_a_build_already_in_flight(self):
        redis = _redis(lock_acquired=True)
        with (
            patch(
                f"{_M}.get_template_state",
                _states(
                    TemplateState.MISSING,
                    TemplateState.BUILDING,
                    TemplateState.BUILDING,
                    TemplateState.READY,
                ),
            ),
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}.asyncio.sleep", AsyncMock()),
        ):
            tpl.build = AsyncMock()
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        tpl.build.assert_not_awaited()
        redis.eval.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lock_winner_builds_when_the_watched_build_fails(self):
        redis = _redis(lock_acquired=True)
        with (
            patch(
                f"{_M}.get_template_state",
                _states(
                    TemplateState.MISSING,
                    TemplateState.BUILDING,
                    TemplateState.BUILDING,
                    TemplateState.MISSING,
                ),
            ),
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.Template"),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}.asyncio.sleep", AsyncMock()) as sleep,
        ):
            tpl.build = AsyncMock(return_value=MagicMock(template_id="t1"))
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        # The holder owns the lock, so failure is read from the build state
        # alone: BUILDING dropping to MISSING means build now, not wait 180 s.
        tpl.build.assert_awaited_once()
        assert sleep.await_count == 1
        redis.exists.assert_not_awaited()

    def test_followers_wait_as_long_as_the_lock_can_live(self):
        assert e2b_template._BUILD_WAIT_SECONDS >= e2b_template._BUILD_LOCK_TTL_SECONDS
        assert (
            e2b_template._BUILD_TIMEOUT_SECONDS < e2b_template._BUILD_LOCK_TTL_SECONDS
        )

    @pytest.mark.asyncio
    async def test_fingerprint_is_not_a_plain_hash_of_the_key(self):
        import hashlib

        key = e2b_template._scoped_key(DESKTOP_IMAGE, _KEY)
        assert key.startswith(f"{DESKTOP_IMAGE.alias}@") and _KEY not in key
        assert hashlib.sha256(_KEY.encode()).hexdigest()[:16] not in key
        assert key != e2b_template._scoped_key(DESKTOP_IMAGE, _OTHER_KEY)

    @pytest.mark.asyncio
    async def test_build_failure_releases_the_lock_and_propagates(self):
        redis = _redis(lock_acquired=True)
        with (
            patch(f"{_M}.get_template_state", _states(*[TemplateState.MISSING] * 2)),
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.Template"),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
        ):
            tpl.build = AsyncMock(side_effect=RuntimeError("build failed"))
            with pytest.raises(RuntimeError):
                await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        redis.eval.assert_awaited_once()
        assert not e2b_template._ready

    @pytest.mark.asyncio
    async def test_stalled_build_is_cut_off_under_the_lock_ttl(self):
        redis = _redis(lock_acquired=True)

        async def never_finishes(*_args, **_kwargs):
            await asyncio.Event().wait()

        with (
            patch(f"{_M}.get_template_state", _states(*[TemplateState.MISSING] * 2)),
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.Template"),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}._BUILD_TIMEOUT_SECONDS", 0.01),
        ):
            tpl.build = AsyncMock(side_effect=never_finishes)
            with pytest.raises(asyncio.TimeoutError):
                await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        assert (
            e2b_template._BUILD_TIMEOUT_SECONDS < e2b_template._BUILD_LOCK_TTL_SECONDS
        )
        redis.eval.assert_awaited_once()
        assert not e2b_template._ready

    @pytest.mark.asyncio
    async def test_follower_waits_for_the_other_builder(self):
        redis = _redis(lock_acquired=False)
        with (
            patch(
                f"{_M}.get_template_state",
                _states(
                    TemplateState.MISSING,
                    TemplateState.MISSING,
                    TemplateState.BUILDING,
                    TemplateState.READY,
                ),
            ) as st,
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}.asyncio.sleep", AsyncMock()),
        ):
            tpl.build = AsyncMock()
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        tpl.build.assert_not_awaited()
        redis.eval.assert_not_awaited()
        assert st.await_count == 4
        assert len(e2b_template._ready) == 1

    @pytest.mark.asyncio
    async def test_follower_stops_when_the_builder_gave_up(self):
        redis = _redis(lock_acquired=False, lock_present=False)
        with (
            patch(f"{_M}.get_template_state", _states(*[TemplateState.MISSING] * 2)),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}.asyncio.sleep", AsyncMock()) as sleep,
        ):
            with pytest.raises(RuntimeError, match="was not built"):
                await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_follower_gives_up_eventually(self):
        redis = _redis(lock_acquired=False)
        with (
            patch(
                f"{_M}.get_template_state",
                AsyncMock(return_value=TemplateState.BUILDING),
            ),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}.asyncio.sleep", AsyncMock()),
        ):
            with pytest.raises(TimeoutError):
                await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        assert not e2b_template._ready


def _client() -> MagicMock:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _response(status_code: int, parsed=None) -> MagicMock:
    return MagicMock(status_code=status_code, parsed=parsed)


def _template(*statuses: TemplateBuildStatus) -> MagicMock:
    from e2b.api.client.models import TemplateWithBuilds

    template = MagicMock(spec=TemplateWithBuilds)
    template.builds = [MagicMock(status=s) for s in statuses]
    return template


class TestGetTemplateState:
    async def _state(self, alias_response, template_response=None) -> TemplateState:
        with (
            patch(f"{_M}.get_api_client", return_value=_client()),
            patch(
                f"{_M}.get_templates_aliases_alias.asyncio_detailed",
                AsyncMock(return_value=alias_response),
            ),
            patch(
                f"{_M}.get_templates_template_id.asyncio_detailed",
                AsyncMock(return_value=template_response),
            ) as by_id,
        ):
            state = await get_template_state(DESKTOP_IMAGE, _KEY)
        if template_response is not None:
            by_id.assert_awaited_once()
            assert by_id.await_args.kwargs["template_id"] == "tpl-1"
        return state

    @pytest.mark.asyncio
    async def test_unknown_alias_is_missing(self):
        assert await self._state(_response(404)) is TemplateState.MISSING

    @pytest.mark.asyncio
    async def test_another_teams_alias_is_used_as_is(self):
        assert await self._state(_response(403)) is TemplateState.READY

    @pytest.mark.asyncio
    async def test_alias_lookup_failure_raises(self):
        with pytest.raises(RuntimeError, match="alias lookup"):
            await self._state(_response(500))

    @pytest.mark.asyncio
    async def test_a_finished_build_means_ready(self):
        alias = _response(200, TemplateAliasResponse(template_id="tpl-1", public=False))
        template = _response(
            200, _template(TemplateBuildStatus.ERROR, TemplateBuildStatus.READY)
        )
        assert await self._state(alias, template) is TemplateState.READY

    @pytest.mark.asyncio
    async def test_a_registered_alias_with_a_running_build_is_not_ready(self):
        alias = _response(200, TemplateAliasResponse(template_id="tpl-1", public=False))
        template = _response(200, _template(TemplateBuildStatus.WAITING))
        assert await self._state(alias, template) is TemplateState.BUILDING

    @pytest.mark.asyncio
    async def test_a_failed_build_leaves_the_template_missing(self):
        alias = _response(200, TemplateAliasResponse(template_id="tpl-1", public=False))
        template = _response(200, _template(TemplateBuildStatus.ERROR))
        assert await self._state(alias, template) is TemplateState.MISSING

    @pytest.mark.asyncio
    async def test_template_lookup_failure_raises(self):
        alias = _response(200, TemplateAliasResponse(template_id="tpl-1", public=False))
        with pytest.raises(RuntimeError, match="template lookup"):
            await self._state(alias, _response(500))
