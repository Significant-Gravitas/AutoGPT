"""The skill index a turn reads is cached for minutes, so every write that
changes an owner's skills has to reach the next read, including a write that
lands while a rebuild is still reading the folder."""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.tools import skills
from backend.copilot.tools.skills import (
    SKILLS_INDEX_CACHE_TTL_S,
    ParsedSkill,
    _claim_skills_cache,
    _read_skills_cache,
    _skills_cache_key,
    _write_skills_cache,
    delete_user_skill,
    invalidate_skills_index_cache,
    invalidate_skills_index_for_path,
    list_user_skills,
    render_skill_markdown,
    store_user_skill,
)
from backend.copilot.tools.skills_test import (
    _cached_skills,
    _FakeRedis,
    _FakeWorkspaceManager,
)
from backend.data import redis_client
from backend.data.redis_client import get_redis_async
from backend.util.testing import is_tcp_port_reachable

USER = "user-1"

requires_redis = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis reachable; the write-if-claimed script needs one",
)


def _skill(name: str) -> bytes:
    return render_skill_markdown(
        ParsedSkill(name=name, description=f"{name} description", body="steps")
    ).encode()


async def _names(expert_id: str | None = None) -> list[str]:
    return [s.name for s in await list_user_skills(USER, expert_id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("expert_id", [None, "expert-1"])
async def test_a_cached_index_shows_a_store_and_a_delete_on_the_next_read(
    expert_id,
):
    with _cached_skills(_FakeWorkspaceManager()) as redis:
        assert await _names(expert_id) == []
        # Cached for the whole TTL, so a write that left it in place would
        # go unseen by every read below.
        key = _skills_cache_key(USER, expert_id)
        assert redis.ttls[key] == SKILLS_INDEX_CACHE_TTL_S

        await store_user_skill(
            USER, name="fresh", description="d", body="b", expert_id=expert_id
        )
        assert await _names(expert_id) == ["fresh"]

        await delete_user_skill(USER, "fresh", expert_id=expert_id)
        assert await _names(expert_id) == []


@pytest.mark.asyncio
async def test_a_skill_stored_while_the_index_is_rebuilt_shows_on_the_next_read():
    """The rebuild was reading the folder when the skill landed. Caching what
    it read would hide the skill for the whole TTL."""
    fake = _FakeWorkspaceManager()
    fake.files["/skills/old/SKILL.md"] = _skill("old")
    listed, resume = asyncio.Event(), asyncio.Event()
    read_folder = skills._list_user_skills_from_workspace

    async def read_then_wait(*args, **kwargs):
        found = await read_folder(*args, **kwargs)
        listed.set()
        await resume.wait()
        return found

    with _cached_skills(fake):
        with patch.object(skills, "_list_user_skills_from_workspace", read_then_wait):
            rebuild = asyncio.create_task(list_user_skills(USER))
            await listed.wait()
        await store_user_skill(USER, name="new", description="d", body="b")
        resume.set()

        assert [s.name for s in await rebuild] == ["old"]
        assert await _names() == ["new", "old"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "owner"),
    [
        ("/skills/brief/SKILL.md", None),
        ("/skills/brief/references/tone.md", None),
        ("/experts/expert-1/skills/brief/SKILL.md", "expert-1"),
        ("/experts/expert-1/skills/brief/scripts/run.py", "expert-1"),
    ],
)
async def test_a_file_changed_in_a_skill_folder_drops_that_owners_index(path, owner):
    owners = [None, "expert-1", "expert-2"]
    redis = _FakeRedis()
    for each in owners:
        redis.store[_skills_cache_key(USER, each)] = "[]"
    with patch(
        "backend.copilot.tools.skills.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        await invalidate_skills_index_for_path(USER, path)

    dropped = [o for o in owners if _skills_cache_key(USER, o) not in redis.store]
    assert dropped == [owner]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/notes.md",
        "/sessions/session-1/skills/brief/SKILL.md",
        "/experts/expert-1/notes.md",
        "/skillset/brief.md",
    ],
)
async def test_a_file_outside_every_skill_folder_drops_no_index(path):
    redis = _FakeRedis()
    keys = [_skills_cache_key(USER, owner) for owner in (None, "expert-1")]
    for key in keys:
        redis.store[key] = "[]"
    with patch(
        "backend.copilot.tools.skills.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        await invalidate_skills_index_for_path(USER, path)

    assert sorted(redis.store) == sorted(keys)


@pytest.mark.asyncio
@pytest.mark.parametrize("held", ["finished index", "other claim"])
async def test_a_rebuild_takes_no_claim_over_an_entry_another_holds(held):
    """Two turns can miss at once. The one that claims second must leave the
    index the first just finished, or the first's claim, where it is."""
    redis = _FakeRedis()
    key = _skills_cache_key(USER)
    redis.store[key] = (
        "[]" if held == "finished index" else f"{skills._REBUILDING_MARKER}first"
    )
    before = redis.store[key]
    with patch(
        "backend.copilot.tools.skills.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        refused = await _claim_skills_cache(USER)
        after_refusal = redis.store[key]
        del redis.store[key]
        taken = await _claim_skills_cache(USER)

    assert refused is None and after_refusal == before
    assert taken is not None and redis.store[key] == taken


@requires_redis
@pytest.mark.asyncio
async def test_on_redis_an_index_is_written_only_while_its_claim_holds(server):
    """The fake stands in for the write-if-claimed script everywhere else;
    this runs the script itself."""
    user = f"skills-index-{uuid.uuid4().hex}"
    key = _skills_cache_key(user)
    redis = await get_redis_async()
    try:
        claim = await _claim_skills_cache(user)
        assert claim is not None
        # Another turn reading meanwhile rebuilds rather than taking the claim
        # for an index.
        assert await _read_skills_cache(user) is None
        await _write_skills_cache(
            user, [ParsedSkill(name="kept", description="d", body="")], claim=claim
        )
        cached = await _read_skills_cache(user)
        assert cached is not None and [s.name for s in cached] == ["kept"]
        assert 0 < await redis.ttl(key) <= SKILLS_INDEX_CACHE_TTL_S

        # A rebuild that missed before this index landed takes no claim over it.
        assert await _claim_skills_cache(user) is None
        cached = await _read_skills_cache(user)
        assert cached is not None and [s.name for s in cached] == ["kept"]

        await redis.delete(key)
        lost = await _claim_skills_cache(user)
        assert lost is not None
        await invalidate_skills_index_cache(user)
        await _write_skills_cache(
            user, [ParsedSkill(name="stale", description="d", body="")], claim=lost
        )
        assert await redis.get(key) is None
    finally:
        await redis.delete(key)
