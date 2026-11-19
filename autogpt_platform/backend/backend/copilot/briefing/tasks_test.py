"""The brief's task fields: composing them, storing them, and every reader of
a stored brief (narrative, thread post, home) handling rows with and
without them."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import TaskCreator, TaskStatus

from backend.api.features.experts.models import Expert
from backend.api.features.home.briefing import compose_briefing as compose_home
from backend.copilot.briefing import generate as generate_module
from backend.copilot.briefing.generate import compose_briefing
from backend.copilot.briefing.models import BriefingContent, BriefingTasks
from backend.copilot.briefing.narrative import _facts_block
from backend.copilot.briefing.render import render_briefing_markdown
from backend.copilot.briefing.tasks import MAX_TASK_ITEMS, compose_task_summary
from backend.data.task import Task

NOW = datetime(2026, 11, 14, 8, 0, tzinfo=UTC)
SINCE = NOW - timedelta(hours=24)


def make_task(title: str, status: TaskStatus, hours_ago: float = 2, **fields) -> Task:
    at = NOW - timedelta(hours=hours_ago)
    settled = status in (TaskStatus.DONE, TaskStatus.FAILED)
    return Task(
        id=fields.pop("id", title),
        user_id="user-1",
        title=title,
        status=status,
        created_by=TaskCreator.USER,
        created_at=at - timedelta(hours=1),
        updated_at=at,
        completed_at=at if settled else None,
        **fields,
    )


def make_expert() -> Expert:
    return Expert(
        id="exp-1",
        name="Ana",
        avatar_url="https://a/ana.png",
        role="Researcher",
        tagline=None,
        bio=None,
        skills=[],
        identity="",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=[],
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )


def summary() -> BriefingTasks:
    return compose_task_summary(
        tasks=[
            make_task(
                "Pricing comparison",
                TaskStatus.DONE,
                owner_expert_id="exp-1",
                result="Plan B is 20% cheaper.",
            ),
            make_task("Old report", TaskStatus.DONE, hours_ago=48),
            make_task("Scrape leads", TaskStatus.FAILED, failure_reason="Site down"),
            make_task("Pick a venue", TaskStatus.NEEDS_INPUT, status_text="A or B?"),
            make_task("Draft post", TaskStatus.IN_PROGRESS),
            make_task(
                "Part of comparison",
                TaskStatus.DONE,
                parent_task_id="Pricing comparison",
            ),
        ],
        experts=[make_expert()],
        since=SINCE,
    )


def make_content(tasks: BriefingTasks | None) -> BriefingContent:
    return BriefingContent(
        generated_at=NOW,
        timezone="UTC",
        zero_expert_fallback=False,
        run_items=[],
        decision_items=[],
        tasks=tasks,
    )


def test_summary_reports_what_finished_failed_and_waits_since_the_last_brief():
    tasks = summary()
    [done] = tasks.finished
    assert (done.title, done.owner_name, done.line) == (
        "Pricing comparison",
        "Ana",
        "Plan B is 20% cheaper.",
    )
    assert done.owner_avatar_url == "https://a/ana.png"
    assert [t.line for t in tasks.failed] == ["Site down"]
    assert [t.line for t in tasks.waiting] == ["A or B?"]
    assert (tasks.finished_total, tasks.failed_total, tasks.waiting_total) == (1, 1, 1)
    assert tasks.in_progress_total == 1


def test_summary_caps_its_lists_but_counts_everything():
    tasks = compose_task_summary(
        tasks=[
            make_task(f"Done {n}", TaskStatus.DONE, hours_ago=n + 1)
            for n in range(MAX_TASK_ITEMS + 2)
        ],
        experts=[],
        since=SINCE,
    )
    assert len(tasks.finished) == MAX_TASK_ITEMS
    assert tasks.finished_total == MAX_TASK_ITEMS + 2
    assert tasks.finished[0].title == "Done 0"


def test_a_stored_row_from_before_tasks_still_validates():
    stored = make_content(None).model_dump(mode="json")
    del stored["tasks"]
    content = BriefingContent.model_validate(stored)
    assert content.tasks is None


def test_task_fields_round_trip_through_storage():
    content = make_content(summary())
    stored = BriefingContent.model_validate(content.model_dump(mode="json"))
    assert stored.tasks == content.tasks


def test_a_brief_with_only_task_news_is_still_written():
    def compose(tasks: BriefingTasks):
        return compose_briefing(
            experts=[],
            executions=[],
            reviews=[],
            agent_info_by_graph_id={},
            generated_at=NOW,
            tz_name="UTC",
            task_summary=tasks,
        )

    assert compose(BriefingTasks()) is None
    content = compose(summary())
    assert content is not None and content.tasks == summary()


def test_the_narrative_reads_tasks_escaped():
    tasks = summary().model_copy(
        update={"waiting": [summary().waiting[0].model_copy(update={"title": "<x>"})]}
    )
    facts = _facts_block(make_content(tasks))
    assert "Tasks finished: 1" in facts
    assert "[waiting on the user] &lt;x&gt;" in facts
    assert "[finished] Ana / Pricing comparison: Plan B is 20% cheaper." in facts
    assert "Tasks" not in _facts_block(make_content(None))


def test_the_thread_post_lists_tasks():
    posted = render_briefing_markdown(make_content(summary()))
    assert "**Your tasks**" in posted
    assert "- Pick a venue — waiting on you: A or B?" in posted
    assert "**Your tasks**" not in render_briefing_markdown(make_content(None))


@pytest.mark.parametrize("tasks", [None, summary()])
def test_home_shows_the_stored_tasks_or_none_for_an_older_brief(tasks):
    home = compose_home(
        now=NOW,
        executions=[],
        expert_by_id={},
        agent_by_graph={},
        persisted=make_content(tasks),
    )
    assert home.tasks == tasks


@pytest.mark.asyncio
async def test_a_failed_draft_never_holds_the_brief_back(monkeypatch):
    fake_datetime = MagicMock(wraps=datetime)
    fake_datetime.now.return_value = NOW
    monkeypatch.setattr(generate_module, "datetime", fake_datetime)
    monkeypatch.setattr(
        generate_module, "is_user_paywalled", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        generate_module, "compose_narrative", AsyncMock(return_value=None)
    )
    drafting = AsyncMock(side_effect=RuntimeError("db down"))
    monkeypatch.setattr(generate_module, "draft_task_suggestions", drafting)
    user = MagicMock(timezone="UTC")
    monkeypatch.setattr(
        generate_module,
        "user_db",
        lambda: MagicMock(get_user_by_id=AsyncMock(return_value=user)),
    )
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=None),
        create_briefing=AsyncMock(return_value=MagicMock(id="briefing-1")),
        append_plain_session_message=AsyncMock(return_value="session-1"),
        mark_briefing_delivered=AsyncMock(),
    )
    monkeypatch.setattr(
        generate_module, "get_database_manager_async_client", lambda **kw: client
    )
    board = [make_task("Pick a venue", TaskStatus.NEEDS_INPUT, status_text="A or B?")]
    for name, mock in {
        "experts_db": MagicMock(list_experts=AsyncMock(return_value=[])),
        "execution_db": MagicMock(get_graph_executions=AsyncMock(return_value=[])),
        "review_db": MagicMock(get_pending_reviews_for_user=AsyncMock(return_value=[])),
        "library_db": MagicMock(
            get_library_agent_refs_by_graph_ids=AsyncMock(return_value=[])
        ),
        "task_db": MagicMock(list_tasks=AsyncMock(return_value=board)),
    }.items():
        monkeypatch.setattr(generate_module, name, lambda mock=mock: mock)

    result = await generate_module.generate_and_deliver_briefing("user-1")

    assert result["status"] == "delivered"
    drafting.assert_awaited_once_with("user-1")
    stored = client.create_briefing.await_args.args[2]
    assert stored["tasks"]["waiting"][0]["title"] == "Pick a venue"
