"""Tests for the ask_question choice-button token store."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot import choices

_KEY = "copilot-bot:choice:discord:tok"


def _stored(options: list[str], owner: str = "u1") -> str:
    return json.dumps({"options": options, "owner": owner})


def _redis(*, get=None, getdel=None) -> MagicMock:
    redis = MagicMock()
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    redis.get = AsyncMock(return_value=get)
    redis.getdel = AsyncMock(return_value=getdel)
    return redis


def _patched(redis: MagicMock):
    return patch(
        "backend.copilot.bot.choices.get_redis_async",
        new=AsyncMock(return_value=redis),
    )


@pytest.mark.asyncio
async def test_store_choice_writes_options_and_owner_with_ttl():
    redis = _redis()
    with _patched(redis):
        token = await choices.store_choice("discord", ["US", "EU"], "u1")

    assert len(token) == 12
    key, value = redis.set.await_args.args
    assert key == f"copilot-bot:choice:discord:{token}"
    assert json.loads(value) == {"options": ["US", "EU"], "owner": "u1"}
    assert redis.set.await_args.kwargs["ex"] == choices.CHOICE_TTL


@pytest.mark.asyncio
async def test_resolve_choice_returns_option_by_index():
    redis = _redis(get=_stored(["US", "EU"]), getdel=_stored(["US", "EU"]))
    with _patched(redis):
        resolved = await choices.resolve_choice("discord", "tok", 1, "u1")

    assert resolved.text == "EU"
    assert resolved.refused is False
    redis.getdel.assert_awaited_once_with(_KEY)


@pytest.mark.asyncio
async def test_resolve_choice_missing_token_returns_none():
    redis = _redis(get=None)
    with _patched(redis):
        resolved = await choices.resolve_choice("discord", "gone", 0, "u1")

    assert resolved.text is None
    assert resolved.refused is False
    redis.getdel.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_choice_out_of_range_index_returns_none():
    """A forged or stale payload must not index-error into a 500."""
    redis = _redis(get=_stored(["US", "EU"]), getdel=_stored(["US", "EU"]))
    with _patched(redis):
        resolved = await choices.resolve_choice("discord", "tok", 5, "u1")

    assert resolved.text is None


@pytest.mark.asyncio
async def test_a_bystanders_click_is_refused_and_consumes_nothing():
    """The buttons are visible to a whole channel and are far easier to hit
    than typing a reply. A passer-by must not be able to destroy the token
    and leave the person who was actually asked with "expired"."""
    redis = _redis(get=_stored(["US", "EU"], owner="asker"))
    with _patched(redis):
        resolved = await choices.resolve_choice("discord", "tok", 1, "bystander")

    assert resolved.text is None
    assert resolved.refused is True
    redis.getdel.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_asker_can_still_answer_after_a_bystander_clicked():
    store = {_KEY: _stored(["US", "EU"], owner="asker")}

    async def fake_get(key: str):
        return store.get(key)

    async def fake_getdel(key: str):
        return store.pop(key, None)

    redis = MagicMock()
    redis.get = AsyncMock(side_effect=fake_get)
    redis.getdel = AsyncMock(side_effect=fake_getdel)
    with _patched(redis):
        assert (
            await choices.resolve_choice("discord", "tok", 0, "bystander")
        ).refused is True
        assert (await choices.resolve_choice("discord", "tok", 1, "asker")).text == "EU"


@pytest.mark.asyncio
async def test_resolve_choice_is_single_use_for_the_owner():
    """GETDEL keeps the owner's own double-click (or a platform delivery
    retry racing itself) from resolving twice — each resolve continues the
    paused AutoPilot turn, which is billable."""
    store = {_KEY: _stored(["US"], owner="u1")}

    async def fake_get(key: str):
        return store.get(key)

    async def fake_getdel(key: str):
        return store.pop(key, None)

    redis = MagicMock()
    redis.get = AsyncMock(side_effect=fake_get)
    redis.getdel = AsyncMock(side_effect=fake_getdel)
    with _patched(redis):
        assert (await choices.resolve_choice("discord", "tok", 0, "u1")).text == "US"
        second = await choices.resolve_choice("discord", "tok", 0, "u1")

    assert second.text is None
    assert second.refused is False


@pytest.mark.asyncio
async def test_a_malformed_record_is_refused_rather_than_consumed():
    redis = _redis(get="not json")
    with _patched(redis):
        resolved = await choices.resolve_choice("discord", "tok", 0, "u1")

    assert resolved.text is None
    assert resolved.refused is True
    redis.getdel.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_choice_deletes_key():
    redis = _redis()
    with _patched(redis):
        await choices.clear_choice("discord", "tok")

    redis.delete.assert_awaited_once_with(_KEY)
