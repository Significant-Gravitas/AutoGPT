import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts.models import (
    Expert,
    ExpertActivity,
    ExpertActivityDay,
    ExpertRun,
)
from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.api.features.memory.models import MemoryFact
from backend.copilot.tools.design_page import PageResponse
from backend.data.execution import ExecutionStatus, GraphExecution, GraphExecutionMeta

from .design import ExpertStatusSection, PageDesign
from .design_test import DRAFT, IDEAS, agent
from .resolve import _EXPERT_KINDS, COULD_NOT_LOAD, GONE, resolve_page
from .resolved import ResolvedPage
from .sources import PageSources

_MODULE = "backend.api.features.pages.resolve"
NOW = datetime(2026, 11, 21, 12, tzinfo=UTC)
EXPERT = Expert.model_construct(
    id="nadia", name="Nadia", is_archived=False, is_template=False
)


def meta(
    run_id: str,
    graph_id: str = "graph-ideas",
    *,
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    inputs: dict[str, Any] | None = None,
    cost: int = 0,
    summary: str | None = None,
    minutes_ago: int = 0,
) -> GraphExecutionMeta:
    started = NOW - timedelta(minutes=minutes_ago)
    return GraphExecutionMeta.model_construct(
        id=run_id,
        graph_id=graph_id,
        status=status,
        inputs=inputs or {},
        is_dry_run=False,
        started_at=started,
        ended_at=started + timedelta(seconds=30),
        stats=GraphExecutionMeta.Stats.model_construct(
            cost=cost, duration=30.0, activity_status=summary
        ),
    )


def run(
    run_id: str, outputs: dict[str, list[Any]], graph_id: str = "graph-ideas"
) -> GraphExecution:
    return GraphExecution.model_construct(
        **dict(meta(run_id, graph_id)), outputs=outputs
    )


KEYWORDS = [{"keyword": "remote work"}, {"keyword": "four-day week"}]


class Fakes:
    """The data each section reads, keyed the way the readers ask for it."""

    def __init__(self) -> None:
        self.recent: dict[str, list[GraphExecutionMeta]] = {}
        self.latest: dict[str, GraphExecution] = {}
        self.full: dict[str, GraphExecution] = {}
        self.reviews: list[PendingHumanReviewModel] = []
        self.paused_by_expert: list[GraphExecutionMeta] = []
        self.presets: list[Any] = []
        self.facts: list[MemoryFact] = []
        self.expert_runs: list[ExpertRun] = []
        self.activity = ExpertActivity(timezone="UTC", days=[])

    async def get_graph_executions(self, **kw: Any) -> list[GraphExecutionMeta]:
        if kw.get("execution_ids"):
            return [self.full[i] for i in kw["execution_ids"] if i in self.full]
        if kw.get("expert_id"):
            statuses = kw.get("statuses") or []
            if ExecutionStatus.REVIEW in statuses:
                return self.paused_by_expert
            return [] if statuses else [meta("e1", cost=120)]
        runs = self.recent.get(kw["graph_id"], [])
        if kw.get("statuses") == [ExecutionStatus.COMPLETED]:
            done = self.latest.get(kw["graph_id"])
            return [done] if done else []
        return runs

    async def get_graph_execution(
        self, user_id: str, execution_id: str
    ) -> GraphExecution | None:
        return self.full.get(execution_id) or next(
            (r for r in self.latest.values() if r.id == execution_id), None
        )


@pytest.fixture
def fakes() -> Any:
    data = Fakes()
    execution = MagicMock(
        get_graph_executions=AsyncMock(side_effect=data.get_graph_executions),
        get_graph_execution=AsyncMock(side_effect=data.get_graph_execution),
    )
    library = MagicMock(
        list_presets=AsyncMock(
            side_effect=lambda *a, **k: MagicMock(presets=data.presets)
        )
    )
    reviews = MagicMock(
        get_pending_reviews_for_user=AsyncMock(side_effect=lambda *a: data.reviews)
    )
    experts = MagicMock(
        list_expert_runs=AsyncMock(side_effect=lambda *a: data.expert_runs),
        get_expert_activity=AsyncMock(side_effect=lambda *a: data.activity),
    )
    with (
        patch(f"{_MODULE}.execution_db", return_value=execution),
        patch(f"{_MODULE}.library_db", return_value=library),
        patch(f"{_MODULE}.review_db", return_value=reviews),
        patch(f"{_MODULE}.experts_db", return_value=experts),
        patch(
            f"{_MODULE}.read_live_facts", AsyncMock(side_effect=lambda *a: data.facts)
        ),
    ):
        yield data


SOURCES = PageSources(
    workflows={"ideas": IDEAS, "draft": DRAFT}, experts={"nadia": EXPERT}
)


async def resolve(*sections: dict[str, Any], **kw: Any) -> ResolvedPage:
    design = PageDesign.model_validate({"title": "Posts", "sections": list(sections)})
    return await resolve_page("user-1", design, kw.pop("sources", SOURCES), **kw)


ROW_TABLE = {
    "kind": "latest_result",
    "workflow": "ideas",
    "cards": [
        {
            "type": "table",
            "output": "keywords",
            "columns": [{"label": "Keyword", "key": "keyword"}],
            "row_action": {
                "label": "Draft",
                "workflow": "draft",
                "input": "keyword",
                "column": "keyword",
            },
        }
    ],
}
BOUND_FORM = {
    "kind": "run_form",
    "workflow": "draft",
    "fields": [
        {"input": "keyword", "from_output": {"workflow": "ideas", "output": "topic"}}
    ],
}


@pytest.mark.asyncio
async def test_a_row_button_carries_each_rows_value_and_a_bound_field_starts_from_the_output(
    fakes: Fakes,
) -> None:
    fakes.latest["graph-ideas"] = run(
        "ideas-1", {"keywords": [KEYWORDS], "topic": ["work"]}
    )

    table, form = (await resolve(ROW_TABLE, BOUND_FORM)).sections

    assert table.kind == "latest_result" and form.kind == "run_form"
    (row_run,) = table.row_runs
    assert (row_run.library_agent_id, row_run.input, row_run.values) == (
        "draft",
        "keyword",
        ["remote work", "four-day week"],
    )
    assert form.fields[0].default == "work"


@pytest.mark.asyncio
async def test_a_followed_run_replaces_the_latest_for_its_workflow_only(
    fakes: Fakes,
) -> None:
    fakes.latest["graph-ideas"] = run("old", {"keywords": [KEYWORDS], "topic": ["old"]})
    followed = run("new", {"keywords": [KEYWORDS[:1]], "topic": ["new"]})

    table, form = (await resolve(ROW_TABLE, BOUND_FORM, followed=[followed])).sections

    assert table.run_id == "new" and table.row_runs[0].values == ["remote work"]
    assert form.fields[0].default == "new"


@pytest.mark.asyncio
async def test_a_form_starts_from_the_last_runs_inputs_and_keeps_optional_inputs(
    fakes: Fakes,
) -> None:
    wide = agent(
        "wide", {"a": {"type": "string"}, "b": {"type": "string", "default": "x"}}
    )
    fakes.recent["graph-wide"] = [meta("r1", "graph-wide", inputs={"a": "last"})]

    (form,) = (
        await resolve(
            {"kind": "run_form", "workflow": "wide", "fields": [{"input": "a"}]},
            sources=PageSources(workflows={"wide": wide}),
        )
    ).sections

    assert [(f.input, f.default, f.group) for f in form.fields] == [
        ("a", "last", None),
        ("b", "x", "More options"),
    ]


@pytest.mark.asyncio
async def test_run_history_numbers_and_trend_read_the_recent_runs(fakes: Fakes) -> None:
    runs = [meta(f"d{i}", "graph-draft", cost=10 * i, minutes_ago=i) for i in range(3)]
    fakes.recent["graph-draft"] = runs
    fakes.full = {
        r.id: run(r.id, {"title": [f"Post {r.id}"]}, "graph-draft") for r in runs
    }

    history, numbers, trend = (
        await resolve(
            {
                "kind": "run_history",
                "workflow": "draft",
                "columns": [{"label": "Title", "output": "title"}],
            },
            {"kind": "run_numbers", "workflow": "draft", "period": "7d"},
            {"kind": "run_trend", "workflow": "draft", "value": "cost", "runs": 5},
        )
    ).sections

    assert [row.cells for row in history.rows] == [
        ["Post d0"],
        ["Post d1"],
        ["Post d2"],
    ]
    assert [(i.label, i.value) for i in numbers.block.items] == [
        ("Runs", "3"),
        ("Succeeded", "100%"),
        ("Spent", "$0.30"),
    ]
    assert trend.block is not None and trend.block.series[0].values == [0.2, 0.1, 0.0]


@pytest.mark.asyncio
async def test_quick_runs_list_only_active_templates_without_a_trigger(
    fakes: Fakes,
) -> None:
    def preset(name: str, active: bool = True, webhook: str | None = None) -> Any:
        return MagicMock(
            id=name, is_active=active, webhook_id=webhook, **{"name": name}
        )

    fakes.presets = [
        preset("a"),
        preset("off", active=False),
        preset("hook", webhook="w"),
        *map(preset, "bcde"),
    ]
    for p in fakes.presets:
        p.name = p.id

    (quick,) = (await resolve({"kind": "quick_runs", "workflow": "ideas"})).sections

    assert [t.name for t in quick.templates] == ["a", "b", "c", "d"]


def review(exec_id: str, graph_id: str) -> PendingHumanReviewModel:
    return PendingHumanReviewModel.model_construct(
        node_exec_id=f"n-{exec_id}",
        graph_exec_id=exec_id,
        graph_id=graph_id,
        payload={"amount": 2400},
    )


@pytest.mark.asyncio
async def test_waiting_is_scoped_to_its_workflow_its_expert_or_nothing(
    fakes: Fakes,
) -> None:
    fakes.reviews = [
        review("x1", "graph-ideas"),
        review("x2", "graph-other"),
        review("x3", "graph-z"),
    ]
    fakes.paused_by_expert = [meta("x3", "graph-z", status=ExecutionStatus.REVIEW)]

    by_workflow, by_expert, everything = (
        await resolve(
            {"kind": "waiting", "workflow": "ideas"},
            {"kind": "waiting", "expert": "nadia"},
            {"kind": "waiting"},
        )
    ).sections

    assert [r.graph_exec_id for r in by_workflow.reviews] == ["x1"]
    assert [r.graph_exec_id for r in by_expert.reviews] == ["x3"]
    assert [r.graph_exec_id for r in everything.reviews] == ["x1", "x2", "x3"]
    assert (by_workflow.scope_name, by_expert.scope_name, everything.scope_name) == (
        "Ideas",
        "Nadia",
        None,
    )


@pytest.mark.asyncio
async def test_expert_sections_read_its_work_memory_and_activity(fakes: Fakes) -> None:
    fakes.expert_runs = [
        ExpertRun.model_construct(
            execution_id="w1",
            agent_name="Competitor watch",
            status="completed",
            started_at=NOW,
            link="/x",
        )
    ]
    fakes.full["w1"] = meta("w1", summary="Found two pricing changes.")
    fakes.facts = [
        MemoryFact(uuid="f", fact="Acme cut prices.", name="n", source="a", target="b")
    ]
    fakes.activity = ExpertActivity(
        timezone="UTC",
        days=[
            ExpertActivityDay(day=date(2026, 11, d), sessions=1, runs=d)
            for d in (20, 21)
        ],
    )
    composed = MagicMock(status="ready", detail="Ready", spend_cents=120)

    with patch(f"{_MODULE}.compose_agent_statuses", return_value=[composed]) as compose:
        status, work, learned, activity = (
            await resolve(
                {"kind": "expert_status", "expert": "nadia"},
                {"kind": "expert_work", "expert": "nadia"},
                {"kind": "expert_learned", "expert": "nadia"},
                {"kind": "expert_activity", "expert": "nadia"},
            )
        ).sections

    assert compose.call_args.kwargs["spend_by_expert"] == {"nadia": 120}
    assert (status.status, status.spend_cents) == ("ready", 120)
    assert work.items[0].summary == "Found two pricing changes."
    assert [f.fact for f in learned.facts] == ["Acme cut prices."]
    assert activity.block is not None and activity.block.series[1].values == [
        20.0,
        21.0,
    ]


def going(run_id: str, status: ExecutionStatus) -> GraphExecution:
    return GraphExecution.model_construct(
        **dict(meta(run_id, status=status)), outputs={}
    )


@pytest.mark.parametrize(
    "case, sections, followed, work, expert_status, expected",
    [
        ("followed run, form only", [BOUND_FORM], ["RUNNING"], [], "ready", True),
        ("followed run finished", [BOUND_FORM], ["COMPLETED"], [], "ready", False),
        ("followed run waits on a person", [ROW_TABLE], ["REVIEW"], [], "ready", False),
        (
            "expert working",
            [{"kind": "expert_status", "expert": "nadia"}],
            [],
            [],
            "working",
            True,
        ),
        (
            "expert work queued",
            [{"kind": "expert_work", "expert": "nadia"}],
            [],
            ["queued"],
            "ready",
            True,
        ),
        (
            "expert work done",
            [{"kind": "expert_work", "expert": "nadia"}],
            [],
            ["completed"],
            "ready",
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_the_page_reads_as_running_only_while_a_run_it_shows_or_follows_is_going(
    fakes: Fakes,
    case: str,
    sections: list[dict[str, Any]],
    followed: list[str],
    work: list[str],
    expert_status: str,
    expected: bool,
) -> None:
    fakes.expert_runs = [
        ExpertRun.model_construct(
            execution_id=f"w{i}",
            agent_name="Watch",
            status=status,
            started_at=NOW,
            link="/x",
        )
        for i, status in enumerate(work)
    ]
    composed = MagicMock(status=expert_status, detail="", spend_cents=0)
    runs = [going(f"f{i}", ExecutionStatus(s)) for i, s in enumerate(followed)]

    with patch(f"{_MODULE}.compose_agent_statuses", return_value=[composed]):
        page = await resolve(*sections, followed=runs)

    assert page.running is expected, case


@pytest.mark.asyncio
async def test_one_section_that_fails_reads_couldnt_load_and_the_rest_still_show(
    fakes: Fakes,
) -> None:
    fakes.latest["graph-ideas"] = run(
        "ideas-1", {"keywords": [KEYWORDS], "topic": ["work"]}
    )
    boom = AsyncMock(side_effect=RuntimeError("boom"))
    with patch.dict(_EXPERT_KINDS, {ExpertStatusSection: boom}):
        text, broken, table, form = (
            await resolve(
                {"kind": "text", "body": "Hello"},
                {"kind": "expert_status", "expert": "nadia"},
                ROW_TABLE,
                BOUND_FORM,
            )
        ).sections

    assert (text.kind, table.kind, form.kind) == ("text", "latest_result", "run_form")
    assert (broken.kind, broken.reason, broken.message) == (
        "notice",
        "failed",
        COULD_NOT_LOAD,
    )


@pytest.mark.asyncio
async def test_a_gone_source_and_one_that_failed_to_load_read_differently(
    fakes: Fakes,
) -> None:
    sources = PageSources(workflows={"ideas": None, "draft": None}, failed={"draft"})

    gone, failed = (
        await resolve(
            {"kind": "schedule", "workflow": "ideas"},
            {"kind": "schedule", "workflow": "draft"},
            sources=sources,
        )
    ).sections

    assert (gone.reason, gone.message) == ("stale", GONE.format(source="workflow"))
    assert (failed.reason, failed.message) == ("failed", COULD_NOT_LOAD)


FIXTURE = (
    Path(__file__).parents[5]
    / "frontend/src/app/(platform)/copilot/tools/DesignPage/__tests__"
    / "pageOutput.json"
)


@pytest.mark.asyncio
async def test_the_tool_output_matches_the_frontend_fixture(fakes: Fakes) -> None:
    """The chat card's test renders this same file, so the two can't drift."""
    fakes.latest["graph-ideas"] = run(
        "ideas-1", {"keywords": [KEYWORDS], "topic": ["work"]}
    )
    page = await resolve(
        {"kind": "text", "body": "Pick a keyword to draft."}, ROW_TABLE, BOUND_FORM
    )
    response = PageResponse(message="Preview shown to the user.", page=page)
    output = json.loads(response.model_dump_json(exclude_none=True))

    if os.environ.get("UPDATE_PAGE_FIXTURE"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(output, indent=2) + "\n")

    assert output == json.loads(FIXTURE.read_text())
