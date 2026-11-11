"""A turn reads skill eligibility after listing, then reads queued updates
beside the chat's skill versions. The versions wait for the registry read,
because a cold one can update a marketplace copy and record its new version."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.copilot.learning import retrieval
from backend.copilot.tools import skills
from backend.util.testing import Rendezvous


def _history(index: str) -> list[str]:
    return [f"<available_skills>\n{index}\n</available_skills>\n\nhello"]


@pytest.fixture
def skills_on(monkeypatch):
    monkeypatch.setattr(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )


@pytest.mark.asyncio
async def test_a_turn_reads_updates_and_versions_together_after_listing(
    skills_on, fake_learning_store, monkeypatch
):
    """Queued updates, heads and seen versions meet after the listing."""
    versions = Rendezvous(3)
    owned = AsyncMock(
        return_value=[
            skills.ParsedSkill(name="mine", description="d", body="x"),
            skills.ParsedSkill(name="stopped", description="d", body="x"),
        ]
    )
    paused = AsyncMock(return_value={"stopped"})
    queued = AsyncMock(side_effect=versions.returning([]))
    heads = AsyncMock(side_effect=versions.returning([]))
    seen = AsyncMock(side_effect=versions.returning(None))
    monkeypatch.setattr(skills, "list_user_skills", owned)
    monkeypatch.setattr(skills, "paused_skill_names", paused)
    monkeypatch.setattr(skills, "_consume_updated", queued)
    monkeypatch.setattr(fake_learning_store, "list_heads", heads)
    monkeypatch.setattr(skills, "read_seen_index_revision", seen)
    defaults = skills.render_skills_index(skills.get_default_skills_for_index())

    notice = await skills.build_skills_update_notice(
        "user-1", "expert-1", _history(defaults), session_id="chat-1"
    )

    assert "New skills: mine." in notice
    assert "stopped" not in notice
    owned.assert_awaited_once_with("user-1", "expert-1", None)
    paused.assert_awaited_once_with("user-1", "expert-1")
    queued.assert_awaited_once_with("user-1", "expert-1")
    heads.assert_awaited_once_with("user-1", "expert-1")
    seen.assert_awaited_once_with("chat-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("unreadable", ["heads", "seen"])
async def test_unreadable_versions_skip_only_the_version_check(
    skills_on, fake_learning_store, monkeypatch, unreadable
):
    """Queued updates are still announced, and the versions the chat saw are
    left as they were for a later turn to compare against."""
    mine = skills.ParsedSkill(name="mine", description="d", body="x")
    failure = AsyncMock(side_effect=OSError("registry unavailable"))
    monkeypatch.setattr(skills, "list_user_skills", AsyncMock(return_value=[mine]))
    monkeypatch.setattr(skills, "paused_skill_names", AsyncMock(return_value=set()))
    monkeypatch.setattr(skills, "_consume_updated", AsyncMock(return_value=["mine"]))
    if unreadable == "heads":
        monkeypatch.setattr(fake_learning_store, "list_heads", failure)
    else:
        redis = AsyncMock()
        redis.get = failure
        monkeypatch.setattr(retrieval, "get_redis_async", AsyncMock(return_value=redis))
        monkeypatch.setattr(
            skills, "read_seen_index_revision", retrieval.read_seen_index_revision
        )
    mark = AsyncMock()
    monkeypatch.setattr(skills, "mark_index_revision_seen", mark)
    index = skills.render_skills_index([*skills.get_default_skills_for_index(), mine])

    notice = await skills.build_skills_update_notice(
        "user-1", prior_contents=_history(index), session_id="chat-1"
    )

    assert "Updated skills: mine." in notice
    failure.assert_awaited_once()
    mark.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_turn", [True, False])
async def test_a_skill_paused_during_listing_is_not_offered(
    skills_on, monkeypatch, first_turn
):
    """A pause committed before listing finishes affects this turn's index."""
    started = asyncio.Event()
    release = asyncio.Event()
    paused_names: set[str] = set()
    mine = skills.ParsedSkill(name="mine", description="d", body="x")

    async def listing(*_args, **_kwargs):
        started.set()
        await release.wait()
        return [mine]

    async def paused(*_args, **_kwargs):
        return set(paused_names)

    monkeypatch.setattr(skills, "list_user_skills", listing)
    monkeypatch.setattr(skills, "paused_skill_names", paused)
    index = skills.render_skills_index([*skills.get_default_skills_for_index(), mine])
    turn = asyncio.create_task(
        skills.build_skills_context("user-1")
        if first_turn
        else skills.build_skills_update_notice("user-1", prior_contents=_history(index))
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        paused_names.add("mine")
    finally:
        release.set()
        await asyncio.gather(turn, return_exceptions=True)
    result = await turn
    if first_turn:
        assert "name: mine" not in result
    else:
        assert "Removed skills: mine." in result


@pytest.mark.asyncio
async def test_a_cancelled_turn_ends_once_its_reads_have_wound_down(
    skills_on, fake_learning_store, monkeypatch
):
    """Cancelling a turn mid-read cancels every read it started and waits for
    each to wind down; the cancellation reaches the caller instead of
    becoming a turn without a notice."""
    started: dict[str, asyncio.Task] = {}
    wound_down: list[str] = []
    all_started = asyncio.Event()

    def blocked(name: str, cleanup_steps: int):
        async def read(*_args, **_kwargs):
            task = asyncio.current_task()
            assert task is not None
            started[name] = task
            if len(started) == 3:
                all_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # A client closing its request, which takes a few loop turns.
                for _ in range(cleanup_steps):
                    await asyncio.sleep(0)
                wound_down.append(name)
                raise

        return read

    monkeypatch.setattr(skills, "list_user_skills", AsyncMock(return_value=[]))
    monkeypatch.setattr(skills, "paused_skill_names", AsyncMock(return_value=set()))
    monkeypatch.setattr(skills, "_consume_updated", blocked("queued", 0))
    monkeypatch.setattr(fake_learning_store, "list_heads", blocked("heads", 5))
    monkeypatch.setattr(skills, "read_seen_index_revision", blocked("seen", 2))

    turn = asyncio.create_task(
        skills.build_skills_update_notice("user-1", session_id="chat-1")
    )
    left_running: list[str] = []
    turn.add_done_callback(
        lambda _: left_running.extend(n for n, t in started.items() if not t.done())
    )
    try:
        await asyncio.wait_for(all_started.wait(), timeout=10)
    finally:
        turn.cancel()
        (outcome,) = await asyncio.gather(turn, return_exceptions=True)
        await asyncio.gather(*started.values(), return_exceptions=True)

    assert isinstance(outcome, asyncio.CancelledError)
    assert set(started) == {"queued", "heads", "seen"}
    assert sorted(wound_down) == sorted(started)
    assert left_running == []
