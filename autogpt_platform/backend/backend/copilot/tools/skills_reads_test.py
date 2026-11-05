"""How a turn reads its skills: the owner's skills beside the paused names,
then the queued updates beside the chat's skill versions. The versions wait
for the registry read, because a cold one can update a marketplace copy and
record its new version."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools import skills
from backend.copilot.tools.skill_updates_test import (
    USER,
    _install,
    _marketplace,
    _package,
)
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path
from backend.util.testing import Rendezvous


def _history(index: str) -> list[str]:
    return [f"<available_skills>\n{index}\n</available_skills>\n\nhello"]


@pytest.fixture
def skills_on(monkeypatch):
    monkeypatch.setattr(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )


@pytest.fixture
def update_queue(monkeypatch):
    """The names a write queues for the next notice, held in memory the way
    the Redis set holds them."""
    queued: set[str] = set()

    async def remember(_user_id, _expert_id, names):
        queued.update(names)

    async def consume(_user_id, _expert_id):
        names = sorted(queued)
        queued.clear()
        return names

    monkeypatch.setattr(skills, "_remember_updated", remember)
    monkeypatch.setattr(skills, "_consume_updated", consume)


def _after_other_ready_work(lookup: AsyncMock) -> AsyncMock:
    """*lookup*, answering only once the loop has run every other ready task,
    as a database round trip lets it."""

    async def answer(*args, **kwargs):
        for _ in range(10):
            await asyncio.sleep(0)
        return await lookup(*args, **kwargs)

    return AsyncMock(side_effect=answer)


@pytest.mark.asyncio
async def test_a_turn_reads_its_skills_in_two_overlapping_stages(
    skills_on, fake_learning_store, monkeypatch
):
    """The owner's skills meet the paused names; after them the queued
    updates, the heads and the versions the chat saw meet each other. A read
    awaited after another never meets its partners and fails the test."""
    registry = Rendezvous(2)
    versions = Rendezvous(3)
    owned = AsyncMock(
        side_effect=registry.returning(
            [
                skills.ParsedSkill(name="mine", description="d", body="x"),
                skills.ParsedSkill(name="stopped", description="d", body="x"),
            ]
        )
    )
    paused = AsyncMock(side_effect=registry.returning({"stopped"}))
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
async def test_a_turn_that_updates_a_copy_announces_the_update_once(
    skills_on, update_queue
):
    """A cold turn that brings a marketplace copy up to date records the new
    version as the one the chat has seen. Heads read while the copy is still
    being updated (here, while its marketplace lookup is out) would record
    the old one, and the next turn would announce the same update again."""
    manager = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n")
    v2 = _package("cold-email", "v2", "# Cold, revised\n")
    with _patch_skills_path(manager) as patched:
        await _install(v1)
        history = _history(await skills.build_skills_context(USER))
        # Takes the install's own queued name and records v1 as seen.
        await skills.build_skills_update_notice(
            USER, prior_contents=history, session_id="chat-1"
        )
        _marketplace(patched, v2)
        patched.skill_db.get_active_versions = _after_other_ready_work(
            patched.skill_db.get_active_versions
        )

        updating = await skills.build_skills_update_notice(
            USER, prior_contents=history, session_id="chat-1"
        )
        after = await skills.build_skills_update_notice(
            USER, prior_contents=history, session_id="chat-1"
        )

    assert manager.files["/skills/cold-email/SKILL.md"] == v2.skill_markdown.encode()
    assert "Updated skills: cold-email." in updating
    assert after == ""


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
        monkeypatch.setattr(skills, "read_seen_index_revision", failure)
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
async def test_a_failed_listing_reaches_the_caller_once_no_read_is_left_running(
    monkeypatch,
):
    """The caller gets the listing's own error, as when the two reads ran in
    turn, and only once the paused read has ended: nothing runs on behind."""
    release = asyncio.Event()
    paused_reads: list[asyncio.Task] = []

    async def listing(*_args, **_kwargs):
        raise RuntimeError("storage unavailable")

    async def paused(*_args, **_kwargs):
        """Answers no names: unknown as well, which on its own would raise a
        different error."""
        task = asyncio.current_task()
        assert task is not None
        paused_reads.append(task)
        await release.wait()

    monkeypatch.setattr(skills, "list_user_skills", listing)
    monkeypatch.setattr(skills, "paused_skill_names", paused)

    listed = asyncio.create_task(
        skills.list_all_skills("user-1", require_complete=True)
    )
    left_running: list[asyncio.Task] = []
    listed.add_done_callback(
        lambda _: left_running.extend(t for t in paused_reads if not t.done())
    )
    for _ in range(10):
        await asyncio.sleep(0)
    release.set()
    (outcome,) = await asyncio.gather(listed, return_exceptions=True)
    await asyncio.gather(*paused_reads, return_exceptions=True)

    assert isinstance(outcome, RuntimeError)
    assert str(outcome) == "storage unavailable"
    assert left_running == []


@pytest.mark.asyncio
async def test_a_cancelled_turn_ends_once_its_reads_have_wound_down(
    skills_on, monkeypatch
):
    """Cancelling a turn mid-read cancels every read it started and waits for
    each to wind down; the cancellation reaches the caller instead of
    becoming a turn without a notice."""
    started: dict[str, asyncio.Task] = {}
    wound_down: list[str] = []

    def blocked(name: str, cleanup_steps: int):
        async def read(*_args, **_kwargs):
            task = asyncio.current_task()
            assert task is not None
            started[name] = task
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # A client closing its request, which takes a few loop turns.
                for _ in range(cleanup_steps):
                    await asyncio.sleep(0)
                wound_down.append(name)
                raise

        return read

    monkeypatch.setattr(skills, "list_user_skills", blocked("files", 0))
    monkeypatch.setattr(skills, "paused_skill_names", blocked("paused", 5))

    turn = asyncio.create_task(
        skills.build_skills_update_notice("user-1", session_id="chat-1")
    )
    for _ in range(10):
        await asyncio.sleep(0)
    left_running: list[str] = []
    turn.add_done_callback(
        lambda _: left_running.extend(n for n, t in started.items() if not t.done())
    )
    turn.cancel()
    (outcome,) = await asyncio.gather(turn, return_exceptions=True)
    await asyncio.gather(*started.values(), return_exceptions=True)

    assert isinstance(outcome, asyncio.CancelledError)
    assert "files" in started
    assert sorted(wound_down) == sorted(started)
    assert left_running == []
