from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.util import e2b_template
from backend.util.e2b_template import (
    DESKTOP_IMAGE,
    TemplateSpec,
    ensure_template,
    forget_ready_templates,
)

_M = "backend.util.e2b_template"
_KEY = "e2b_test"


@pytest.fixture(autouse=True)
def _fresh_cache():
    forget_ready_templates()
    yield
    forget_ready_templates()


def _redis(lock_acquired: bool) -> MagicMock:
    redis = MagicMock()
    redis.set = AsyncMock(return_value=lock_acquired)
    redis.delete = AsyncMock()
    return redis


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
        with patch(f"{_M}.AsyncTemplate") as tpl:
            tpl.alias_exists = AsyncMock()
            await ensure_template("base", _KEY)
        tpl.alias_exists.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_existing_alias_is_checked_once_per_process(self):
        with patch(f"{_M}.AsyncTemplate") as tpl:
            tpl.alias_exists = AsyncMock(return_value=True)
            tpl.build = AsyncMock()
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        tpl.alias_exists.assert_awaited_once_with(DESKTOP_IMAGE.alias, api_key=_KEY)
        tpl.build.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_alias_is_built_at_the_managed_size(self):
        redis = _redis(lock_acquired=True)
        with (
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.Template") as template_cls,
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
        ):
            tpl.alias_exists = AsyncMock(return_value=False)
            tpl.build = AsyncMock(return_value=MagicMock(template_id="t1"))
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)

        template_cls.return_value.from_template.assert_called_once_with("desktop")
        kwargs = tpl.build.await_args.kwargs
        assert tpl.build.await_args.args[1] == DESKTOP_IMAGE.alias
        assert kwargs["cpu_count"] == 1 and kwargs["memory_mb"] == 2048
        assert kwargs["tags"][0] == "default" and kwargs["api_key"] == _KEY
        # The lock is released even though nothing else needs it now.
        redis.delete.assert_awaited_once()
        assert DESKTOP_IMAGE.alias in e2b_template._ready

    @pytest.mark.asyncio
    async def test_build_failure_releases_the_lock_and_propagates(self):
        redis = _redis(lock_acquired=True)
        with (
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.Template"),
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
        ):
            tpl.alias_exists = AsyncMock(return_value=False)
            tpl.build = AsyncMock(side_effect=RuntimeError("build failed"))
            with pytest.raises(RuntimeError):
                await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        redis.delete.assert_awaited_once()
        assert DESKTOP_IMAGE.alias not in e2b_template._ready

    @pytest.mark.asyncio
    async def test_follower_waits_for_the_other_builder(self):
        redis = _redis(lock_acquired=False)
        with (
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}.asyncio.sleep", AsyncMock()),
        ):
            tpl.alias_exists = AsyncMock(side_effect=[False, False, True])
            tpl.build = AsyncMock()
            await ensure_template(DESKTOP_IMAGE.alias, _KEY)
        tpl.build.assert_not_awaited()
        assert tpl.alias_exists.await_count == 3
        assert DESKTOP_IMAGE.alias in e2b_template._ready

    @pytest.mark.asyncio
    async def test_follower_gives_up_eventually(self):
        redis = _redis(lock_acquired=False)
        with (
            patch(f"{_M}.AsyncTemplate") as tpl,
            patch(f"{_M}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_M}.asyncio.sleep", AsyncMock()),
        ):
            tpl.alias_exists = AsyncMock(return_value=False)
            with pytest.raises(TimeoutError):
                await ensure_template(DESKTOP_IMAGE.alias, _KEY)
