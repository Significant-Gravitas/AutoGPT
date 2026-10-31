"""The plan job: the greeting's prompts ranked, owned and stored with the team.

The row lives in memory and every read hands back a copy, so a write made
from a stale read really loses; the conditional write it stands in for is
``db.replace_plan``. Jev is mocked where the job asks it.
"""

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import (
    db,
    intro,
    plan,
    plan_job,
    role_tasks,
    suggested_tasks,
)
from backend.api.features.onboarding_dump.models import (
    ExpertRecommendations,
    RecommendedExpert,
    SuggestedPrompt,
)

USER_ID = "test-user-id"
RECORDING_ID = "rec-1"
TRANSCRIPT = "I sell stickers on Etsy and the orders eat my week."

MARIA = RecommendedExpert(template_id="tpl-maria", name="Maria", role="Marketing")
MAX = RecommendedExpert(template_id="tpl-max", name="Max", role="Sales")
MINA = RecommendedExpert(template_id="tpl-mina", name="Mina", role="Bookkeeping")
NADIA = RecommendedExpert(template_id="tpl-nadia", name="Nadia", role="Research")
TEAM = [MARIA, MAX, MINA, NADIA]

# What the greeting stored, before anyone acted on it. The icons alone name
# an owner for the first three: marketing, sales, bookkeeping.
GREETING_ROWS = [
    {"title": "Rewrite your Etsy listings", "prompt": "Rewrite.", "icon": "megaphone"},
    {"title": "Find 20 gift shops to pitch", "prompt": "Find.", "icon": "target"},
    {"title": "Track orders and costs", "prompt": "Track.", "icon": "chart-bar"},
    {"title": "Plan a holiday drop", "prompt": "Plan it.", "icon": "sparkle"},
]
GREETING_PROMPTS = [SuggestedPrompt.model_validate(row) for row in GREETING_ROWS]
TITLES = [prompt.title for prompt in GREETING_PROMPTS]


def _team(experts: list[RecommendedExpert] = TEAM) -> dict[str, Any]:
    return ExpertRecommendations(
        diagnosis="Orders eat your week.", experts=experts, source="llm"
    ).model_dump(mode="json")


class RowStore:
    """The one brain-dump row, with ``updatedAt`` moving on every write.

    ``race`` lands other writes between the job's read and its write, each
    setting the given fields, so the job's conditional write loses to them.
    """

    def __init__(self) -> None:
        self.row: OnboardingBrainDump | None = None
        self.writes = 0
        self.racing: list[dict[str, Any]] = []

    def put(self, prompts: list[Any], **fields: Any) -> None:
        self.row = OnboardingBrainDump.model_construct(
            **{
                "userId": USER_ID,
                "recordingId": RECORDING_ID,
                "status": BrainDumpStatus.extracting,
                "inputMode": BrainDumpInputMode.voice,
                "transcript": TRANSCRIPT,
                "greeting": "Let's get those orders off your plate.",
                "greetingSeen": False,
                "suggestedPrompts": prompts,
                "recommendedExperts": _team(),
                "updatedAt": datetime(2026, 10, 1, tzinfo=UTC),
                **fields,
            }
        )

    def race(self, times: int = 1, **fields: Any) -> None:
        self.racing.extend([fields] * times)

    @property
    def prompts(self) -> list[Any]:
        assert self.row is not None
        return self.row.suggestedPrompts

    @property
    def team(self) -> ExpertRecommendations:
        assert self.row is not None
        return ExpertRecommendations.model_validate(self.row.recommendedExperts)

    async def get_dump(self, user_id: str) -> OnboardingBrainDump | None:
        return self.row.model_copy(deep=True) if self.row else None

    async def replace_plan(
        self,
        user_id: str,
        recording_id: str,
        read_at: datetime,
        prompts: list[dict[str, Any]],
        team: dict[str, Any],
    ) -> bool:
        assert self.row is not None
        if self.racing:
            # Another write landed between the job's read and this write.
            for field, value in self.racing.pop(0).items():
                setattr(self.row, field, value)
            self.row.updatedAt += timedelta(milliseconds=1)
        if self.row.recordingId != recording_id or self.row.updatedAt != read_at:
            return False
        self.row.suggestedPrompts = prompts
        self.row.recommendedExperts = team
        self.row.updatedAt += timedelta(milliseconds=1)
        self.writes += 1
        return True


@pytest.fixture(autouse=True)
def rows(mocker: MockerFixture) -> RowStore:
    store = RowStore()
    mocker.patch.object(db, "get_dump", new=store.get_dump)
    mocker.patch.object(db, "replace_plan", new=store.replace_plan)
    return store


@pytest.fixture(autouse=True)
def judge(mocker: MockerFixture) -> AsyncMock:
    """Jev unavailable unless a test gives it an answer."""
    mock = AsyncMock(return_value=None)
    mocker.patch.object(plan_job.plan, "judge_plan", new=mock)
    return mock


async def _run(
    user_role: str | None = None, pain_points: list[str] | None = None
) -> None:
    await plan_job.store_ranked_plan(
        USER_ID, RECORDING_ID, TRANSCRIPT, user_role, pain_points or []
    )


def _owned(prompt: SuggestedPrompt, owner: RecommendedExpert | None) -> SuggestedPrompt:
    return prompt.model_copy(
        update={"expert_template_id": owner.template_id if owner else None}
    )


def _judged(
    order: list[int], owners: list[RecommendedExpert | None]
) -> plan.JudgedPlan:
    return plan.JudgedPlan(
        prompts=[
            _owned(GREETING_PROMPTS[index], owner)
            for index, owner in zip(order, owners)
        ],
        order=order,
        scores=[4.0, 3.0, 2.0, 1.0][: len(order)],
        owner_probabilities=[0.9] * len(order),
    )


def _plan_log(caplog: pytest.LogCaptureFixture) -> str:
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Brain dump plan for user")
    ]
    assert len(lines) == 1, caplog.text
    return lines[0]


async def test_without_jev_the_greeting_order_is_kept_with_hinted_owners(
    rows: RowStore, judge: AsyncMock
):
    rows.put(GREETING_ROWS)

    await _run("Founder/CEO", ["Orders"])

    judge.assert_awaited_once_with(
        TRANSCRIPT, "Founder/CEO", ["Orders"], GREETING_PROMPTS, TEAM
    )
    assert rows.writes == 1
    assert [
        (t.id, t.title, t.expert_template_id)
        for t in suggested_tasks.stored_tasks(rows.prompts)
    ] == [
        ("0", TITLES[0], "tpl-maria"),
        ("1", TITLES[1], "tpl-max"),
        ("2", TITLES[2], "tpl-mina"),
        ("3", TITLES[3], None),
    ]
    assert [e.name for e in rows.team.experts] == ["Maria", "Max", "Mina"]


async def test_a_jev_reorder_keeps_each_task_under_the_id_it_was_handed_out_with(
    rows: RowStore, judge: AsyncMock
):
    rows.put(GREETING_ROWS)
    handed_out = {t.id: t.title for t in suggested_tasks.stored_tasks(rows.prompts)}
    judge.return_value = _judged([1, 0, 3, 2], [MAX, MARIA, None, MINA])

    await _run()

    tasks = suggested_tasks.stored_tasks(rows.prompts)
    assert [(t.id, t.title, t.expert_template_id) for t in tasks] == [
        ("1", TITLES[1], "tpl-max"),
        ("0", TITLES[0], "tpl-maria"),
        ("3", TITLES[3], None),
        ("2", TITLES[2], "tpl-mina"),
    ]
    assert {t.id: t.title for t in tasks} == handed_out


async def test_the_team_is_capped_at_three_and_a_task_outside_it_goes_to_otto(
    rows: RowStore, judge: AsyncMock
):
    rows.put(
        [
            *GREETING_ROWS[:3],
            {"title": "Watch competitors", "prompt": "Watch.", "icon": "bell"},
        ]
    )

    await _run()

    # Nadia would take the fourth task, but the first three tasks' owners
    # fill the team, so she is not on it and that task is Otto's.
    assert [e.name for e in rows.team.experts] == ["Maria", "Max", "Mina"]
    assert rows.team.diagnosis == "Orders eat your week."
    assert [row["expert_template_id"] for row in rows.prompts] == [
        "tpl-maria",
        "tpl-max",
        "tpl-mina",
        None,
    ]


async def test_a_jev_owner_left_off_the_capped_team_goes_to_otto(
    rows: RowStore, judge: AsyncMock
):
    rows.put(GREETING_ROWS)
    judge.return_value = _judged([0, 1, 2, 3], [MINA, MAX, MARIA, NADIA])

    await _run()

    assert [e.name for e in rows.team.experts] == ["Mina", "Max", "Maria"]
    assert [row["expert_template_id"] for row in rows.prompts] == [
        "tpl-mina",
        "tpl-max",
        "tpl-maria",
        None,
    ]


async def test_a_lost_write_is_retried_without_asking_jev_again(
    rows: RowStore, judge: AsyncMock
):
    rows.put(GREETING_ROWS)
    rows.race(times=2)
    judge.return_value = _judged([1, 0, 3, 2], [MAX, MARIA, None, MINA])

    await _run()

    judge.assert_awaited_once()
    assert rows.writes == 1
    assert [row["title"] for row in rows.prompts] == [
        TITLES[1],
        TITLES[0],
        TITLES[3],
        TITLES[2],
    ]


async def test_three_lost_writes_store_nothing_and_say_so(
    rows: RowStore, judge: AsyncMock, caplog: pytest.LogCaptureFixture
):
    rows.put(GREETING_ROWS)
    rows.race(times=3)

    with caplog.at_level(logging.INFO, logger=plan_job.logger.name):
        await _run()

    judge.assert_awaited_once()
    assert rows.writes == 0
    assert rows.prompts == GREETING_ROWS
    assert rows.team == ExpertRecommendations.model_validate(_team())
    assert "gave up after 3 lost writes" in caplog.text
    assert "Brain dump plan for user" not in caplog.text


async def test_a_list_rewritten_mid_job_is_ranked_afresh(
    rows: RowStore, judge: AsyncMock
):
    rows.put(GREETING_ROWS)
    rewritten = GREETING_ROWS[:2]
    rows.race(suggestedPrompts=rewritten)

    await _run()

    assert [call.args[3] for call in judge.await_args_list] == [
        GREETING_PROMPTS,
        GREETING_PROMPTS[:2],
    ]
    assert [row["title"] for row in rows.prompts] == TITLES[:2]


async def test_a_task_acted_on_mid_job_keeps_its_state_and_nothing_is_written(
    rows: RowStore, judge: AsyncMock
):
    rows.put(GREETING_ROWS)
    dismissed = [{**GREETING_ROWS[0], "id": "0", "state": "dismissed"}]
    rows.race(suggestedPrompts=[*dismissed, *GREETING_ROWS[1:]])

    await _run()

    assert rows.writes == 0
    assert rows.prompts[0]["state"] == "dismissed"
    assert [row["title"] for row in rows.prompts] == TITLES


async def test_a_new_take_mid_job_is_left_alone(rows: RowStore, judge: AsyncMock):
    rows.put(GREETING_ROWS)
    rows.race(recordingId="rec-2", suggestedPrompts=[])

    await _run()

    assert rows.writes == 0
    assert rows.prompts == []


async def test_rows_that_already_carry_a_state_are_left_alone(
    rows: RowStore, judge: AsyncMock
):
    stored = [
        {**row, "id": str(index), "state": "open"}
        for index, row in enumerate(GREETING_ROWS)
    ]
    stored[1]["state"] = "sent"
    rows.put(stored)

    await _run()

    judge.assert_not_awaited()
    assert rows.writes == 0
    assert rows.prompts == stored


@pytest.mark.parametrize(
    "fields",
    [
        {"recordingId": "rec-2"},
        {"suggestedPrompts": []},
        {"suggestedPrompts": [{"title": 3}]},
        {"recommendedExperts": None},
        {"recommendedExperts": {"experts": "not a list"}},
    ],
    ids=[
        "another take",
        "no prompts",
        "unparseable prompts",
        "no team",
        "unparseable team",
    ],
)
async def test_nothing_is_written_without_this_takes_prompts_and_team(
    rows: RowStore, judge: AsyncMock, fields: dict[str, Any]
):
    rows.put(GREETING_ROWS, **fields)
    before = rows.row.model_copy(deep=True) if rows.row else None

    await _run()

    judge.assert_not_awaited()
    assert rows.writes == 0
    assert rows.row == before


async def test_a_user_without_a_dump_gets_nothing_written(
    rows: RowStore, judge: AsyncMock
):
    await _run()

    judge.assert_not_awaited()
    assert rows.row is None


async def test_a_user_with_a_role_gets_their_role_set_over_the_generic_starters(
    rows: RowStore, judge: AsyncMock
):
    rows.put([p.model_dump(mode="json") for p in intro.fallback_prompts()])

    await _run("Founder/CEO", ["Finding leads"])

    judge.assert_not_awaited()
    tasks = suggested_tasks.stored_tasks(rows.prompts)
    assert [t.title for t in tasks] == [
        t.title for t in role_tasks.ROLE_FALLBACK_TASKS["Founder/CEO"]
    ]
    assert [t.id for t in tasks] == ["0", "1", "2", "3", "4"]
    # Sales and marketing tasks go to the stored team's Max and Maria.
    assert [t.expert_template_id for t in tasks] == [
        "tpl-max",
        "tpl-maria",
        None,
        None,
        "tpl-maria",
    ]
    assert tasks[0].why == "Most founders start here."
    assert [e.name for e in rows.team.experts] == ["Max", "Maria", "Mina"]


async def test_a_role_set_task_whose_role_is_not_on_the_stored_team_is_otto_s(
    rows: RowStore, judge: AsyncMock
):
    rows.put(
        [p.model_dump(mode="json") for p in intro.fallback_prompts()],
        recommendedExperts=_team([MARIA, MINA, NADIA]),
    )

    await _run("Founder/CEO", [])

    assert [row["expert_template_id"] for row in rows.prompts] == [
        None,
        "tpl-maria",
        None,
        None,
        "tpl-maria",
    ]
    assert [e.name for e in rows.team.experts] == ["Maria", "Mina", "Nadia"]


async def test_a_user_without_a_role_keeps_the_generic_starters_ranked(
    rows: RowStore, judge: AsyncMock
):
    generic = intro.fallback_prompts()
    rows.put([p.model_dump(mode="json") for p in generic])

    await _run(None, [])

    judge.assert_awaited_once()
    assert judge.await_args is not None
    assert judge.await_args.args[3] == generic
    assert rows.writes == 1
    assert [row["title"] for row in rows.prompts] == [p.title for p in generic]


async def test_a_greeting_written_from_the_dump_is_ranked_even_with_a_role(
    rows: RowStore, judge: AsyncMock
):
    rows.put(GREETING_ROWS)

    await _run("Marketing", [])

    judge.assert_awaited_once()
    assert [row["title"] for row in rows.prompts] == TITLES


@pytest.mark.parametrize(
    "broken", ["get_dump", "replace_plan"], ids=["read fails", "write fails"]
)
async def test_a_database_failure_never_escapes_the_job(
    rows: RowStore,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    broken: str,
):
    rows.put(GREETING_ROWS)
    mocker.patch.object(
        db, broken, new=AsyncMock(side_effect=RuntimeError("database down"))
    )

    with caplog.at_level(logging.WARNING, logger=plan_job.logger.name):
        await _run()

    assert rows.prompts == GREETING_ROWS
    assert f"Brain dump plan not stored for user {USER_ID}" in caplog.text


async def test_the_log_line_names_jev_s_answer_and_how_long_it_took(
    rows: RowStore, judge: AsyncMock, caplog: pytest.LogCaptureFixture
):
    rows.put(GREETING_ROWS)
    judge.return_value = _judged([1, 0, 3, 2], [MAX, MARIA, None, MINA])

    with caplog.at_level(logging.INFO, logger=plan_job.logger.name):
        await _run()

    line = _plan_log(caplog)
    assert "source=llm, 4 tasks, 3 owned, fallback=False" in line
    assert re.search(r"jev \d+ms", line)
    assert "scores [4.0, 3.0, 2.0, 1.0]" in line


async def test_the_log_line_says_when_jev_failed_and_the_order_fell_back(
    rows: RowStore, judge: AsyncMock, caplog: pytest.LogCaptureFixture
):
    rows.put(GREETING_ROWS)

    with caplog.at_level(logging.INFO, logger=plan_job.logger.name):
        await _run()

    line = _plan_log(caplog)
    assert "source=llm" in line
    assert "fallback=True" in line
    assert re.search(r"jev \d+ms", line)
    assert line.endswith("fallback order")


async def test_the_log_line_says_jev_was_not_asked_for_a_role_set(
    rows: RowStore, judge: AsyncMock, caplog: pytest.LogCaptureFixture
):
    rows.put([p.model_dump(mode="json") for p in intro.fallback_prompts()])

    with caplog.at_level(logging.INFO, logger=plan_job.logger.name):
        await _run("Founder/CEO", [])

    line = _plan_log(caplog)
    assert "source=role set" in line
    assert "jev not asked" in line
    assert not re.search(r"jev \d+ms", line)
