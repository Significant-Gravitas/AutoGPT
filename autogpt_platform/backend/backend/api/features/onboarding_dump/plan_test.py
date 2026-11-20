"""The plan's Jev call: answers read into an order and owners, every failure
a None, and a state that always fits the request whole."""

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import intro, plan
from backend.api.features.onboarding_dump.models import (
    RecommendedExpert,
    SuggestedPrompt,
)
from backend.blocks.typesafe._budget import prepare_state
from backend.blocks.typesafe._client import JevCallResult

MARIA = RecommendedExpert(
    template_id="tpl-maria",
    name="Maria",
    role="Marketing",
    workflow_names=["Etsy listing rewrite"],
)
MAX = RecommendedExpert(template_id="tpl-max", name="Max", role="Sales")
MINA = RecommendedExpert(template_id="tpl-mina", name="Mina", role="Bookkeeping")
NADIA = RecommendedExpert(template_id="tpl-nadia", name="Nadia", role="Research")
TEAM = [MARIA, MAX, MINA, NADIA]

PROMPTS = [
    SuggestedPrompt(title="Rewrite your Etsy listings", prompt="Rewrite them."),
    SuggestedPrompt(title="Find 20 gift shops", prompt="Find shops."),
    SuggestedPrompt(title="Track orders and costs", prompt="Track them."),
    SuggestedPrompt(title="Plan a holiday drop", prompt="Plan it."),
]


def _result(answers: dict[str, Any], **overrides: Any) -> JevCallResult:
    fields: dict[str, Any] = {
        "answers": answers,
        "request": "{}",
        "response": "{}",
        "latency_ms": 120.0,
        "input_tokens": 10,
        "output_tokens": 5,
        "request_id": "req-1",
        "truncated": False,
        "truncation_note": "",
    }
    return JevCallResult(**{**fields, **overrides})


def _owner(
    choice: str,
    probability: float | None,
    *,
    confidence: float,
    others: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Jev's answer to one owner question.

    ``probability`` is ``probabilities[choice]``, kept apart from
    ``confidence`` so a test shows which of the two decided; None leaves
    the choice out of ``probabilities``, which then lists only ``others``.
    """
    probabilities = dict(others or {})
    if probability is not None:
        probabilities[choice] = probability
    return {
        "type": "choice",
        "choice": choice,
        "confidence": confidence,
        "probabilities": probabilities,
    }


OTTO_SURE = _owner("otto", 0.9, confidence=0.7)


def _answers(
    scores: list[float], owners: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    for index, (score, owner) in enumerate(zip(scores, owners)):
        answers[f"rank_{index}"] = {"type": "score", "score": score}
        answers[f"owner_{index}"] = owner
    return answers


@pytest.fixture
def jev(mocker: MockerFixture) -> AsyncMock:
    mocker.patch.object(plan, "_api_key", "test-key")
    mock = AsyncMock()
    mocker.patch.object(plan, "call_jev", new=mock)
    return mock


async def _judge(prompts=PROMPTS, team=TEAM) -> plan.JudgedPlan | None:
    return await plan.judge_plan(
        "I sell stickers on Etsy.", "Founder/CEO", ["Social media"], prompts, team
    )


async def test_jev_answers_order_the_tasks_and_name_their_owners(jev: AsyncMock):
    jev.return_value = _result(
        _answers(
            [2.0, 4.0, 2.0, 3.1],
            [
                _owner("tpl-maria", 0.9, confidence=0.6),
                _owner("tpl-max", 0.8, confidence=0.5),
                _owner("tpl-mina", 0.7, confidence=0.45),
                _owner("otto", 0.6, confidence=0.55),
            ],
        )
    )

    judged = await _judge()

    assert judged is not None
    # Highest score first; the two tasks that tie keep the greeting's order.
    assert [p.title for p in judged.prompts] == [
        "Find 20 gift shops",
        "Plan a holiday drop",
        "Rewrite your Etsy listings",
        "Track orders and costs",
    ]
    assert [p.expert_template_id for p in judged.prompts] == [
        "tpl-max",
        None,
        "tpl-maria",
        "tpl-mina",
    ]
    assert judged.scores == [4.0, 3.1, 2.0, 2.0]
    assert judged.owner_probabilities == [0.8, 0.6, 0.9, 0.7]


async def test_the_order_maps_each_ranked_task_back_to_its_place_in_the_greeting(
    jev: AsyncMock,
):
    jev.return_value = _result(_answers([0.0, 1.0, 4.0, 2.0], [OTTO_SURE] * 4))

    judged = await _judge()

    assert judged is not None
    assert judged.order == [2, 3, 1, 0]
    assert [PROMPTS[index].title for index in judged.order] == [
        p.title for p in judged.prompts
    ]


async def test_one_call_asks_a_rank_and_an_owner_for_every_task(jev: AsyncMock):
    jev.return_value = _result(_answers([1.0] * 4, [OTTO_SURE] * 4))

    await _judge()

    jev.assert_awaited_once()
    key, state, questions = jev.await_args_list[0].args
    assert key == "test-key"
    assert sorted(questions) == sorted(
        [f"rank_{i}" for i in range(4)] + [f"owner_{i}" for i in range(4)]
    )
    assert questions["rank_2"].criteria == plan.RANK_LEVELS
    assert set(questions["owner_0"].criteria) == {
        "tpl-maria",
        "tpl-max",
        "tpl-mina",
        "tpl-nadia",
        "otto",
    }
    assert questions["owner_0"].criteria["tpl-maria"] == (
        "Maria, Marketing: Etsy listing rewrite"
    )
    # The rubric comes first, then what Jev judges against.
    assert list(state)[:2] == ["rubric", "role"]
    assert state["tasks"][1] == {
        "index": 1,
        "title": "Find 20 gift shops",
        "prompt": "Find shops.",
    }


async def test_the_choice_s_own_probability_decides_an_owner_not_the_confidence(
    jev: AsyncMock,
):
    jev.return_value = _result(
        _answers(
            [3.0, 2.0, 1.0, 0.0],
            [
                _owner("tpl-maria", 0.39, confidence=0.95),
                _owner("tpl-max", 0.4, confidence=0.1),
                _owner("tpl-mina", 0.95, confidence=0.39),
                _owner("tpl-nadia", 0.1, confidence=0.9),
            ],
        )
    )

    judged = await _judge()

    assert judged is not None
    # Maria and Nadia are below the bar on their own probability, however
    # sure the confidence reads; Max and Mina clear it however unsure.
    assert [p.expert_template_id for p in judged.prompts] == [
        None,
        "tpl-max",
        "tpl-mina",
        None,
    ]
    assert judged.owner_probabilities == [0.39, 0.4, 0.95, 0.1]


async def test_the_confidence_decides_an_owner_the_probabilities_leave_out(
    jev: AsyncMock,
):
    jev.return_value = _result(
        _answers(
            [3.0, 2.0, 1.0, 0.0],
            [
                _owner("tpl-maria", None, confidence=0.8, others={"otto": 0.05}),
                _owner("tpl-max", None, confidence=0.2, others={"tpl-mina": 0.9}),
                _owner("tpl-mina", None, confidence=0.6),
                {"type": "choice", "choice": "tpl-nadia", "confidence": 0.3},
            ],
        )
    )

    judged = await _judge()

    assert judged is not None
    assert [p.expert_template_id for p in judged.prompts] == [
        "tpl-maria",
        None,
        "tpl-mina",
        None,
    ]
    assert judged.owner_probabilities == [0.8, 0.2, 0.6, 0.3]


async def test_a_complete_answer_with_a_soft_error_is_still_used(jev: AsyncMock):
    jev.return_value = _result(
        _answers([1.0, 3.0, 2.0, 0.0], [_owner("tpl-max", 0.9, confidence=0.8)] * 4),
        request_id="",
        error="Jev response did not include a request ID.",
    )

    judged = await _judge()

    assert judged is not None
    assert [p.title for p in judged.prompts] == [
        "Find 20 gift shops",
        "Track orders and costs",
        "Rewrite your Etsy listings",
        "Plan a holiday drop",
    ]
    assert {p.expert_template_id for p in judged.prompts} == {"tpl-max"}


@pytest.mark.parametrize(
    "failure",
    [
        {"answers": {}, "error": "Jev API request failed (HTTP 429)."},
        {"answers": {}},
        {"truncated": True, "truncation_note": "State truncated."},
    ],
    ids=["failed call", "no answers", "cut state"],
)
async def test_a_failed_empty_or_cut_call_returns_none(
    jev: AsyncMock, failure: dict[str, Any]
):
    complete = _answers([1.0] * 4, [OTTO_SURE] * 4)
    jev.return_value = _result(**{"answers": complete, **failure})

    assert await _judge() is None


@pytest.mark.parametrize("raised", [TimeoutError(), RuntimeError("connection reset")])
async def test_a_timeout_or_a_raise_returns_none(
    jev: AsyncMock, raised: Exception, caplog: pytest.LogCaptureFixture
):
    jev.side_effect = raised

    with caplog.at_level(logging.WARNING, logger=plan.logger.name):
        assert await _judge() is None

    # One line per failed plan, so a Jev outage reads as one per dump.
    assert len(caplog.records) == 1


async def test_a_call_that_outlives_the_timeout_returns_none(
    jev: AsyncMock, mocker: MockerFixture
):
    mocker.patch.object(plan, "_TIMEOUT_SECONDS", 0.01)

    # A complete answer, but too late: only the timeout makes this None.
    async def slow(*args: Any, **kwargs: Any) -> JevCallResult:
        await asyncio.sleep(5)
        return _result(_answers([1.0] * len(PROMPTS), [OTTO_SURE] * len(PROMPTS)))

    jev.side_effect = slow

    assert await _judge() is None


async def test_no_key_returns_none_without_calling(
    jev: AsyncMock, mocker: MockerFixture
):
    mocker.patch.object(plan, "_api_key", "")

    assert await _judge() is None
    jev.assert_not_awaited()


async def test_no_prompts_returns_none_without_calling(jev: AsyncMock):
    assert await _judge(prompts=[]) is None
    jev.assert_not_awaited()


@pytest.mark.parametrize(
    "drop", ["rank_2", "owner_0"], ids=["missing rank", "missing owner"]
)
async def test_a_missing_answer_returns_none(jev: AsyncMock, drop: str):
    answers = _answers([1.0] * 4, [OTTO_SURE] * 4)
    del answers[drop]
    jev.return_value = _result(answers)

    assert await _judge() is None


@pytest.mark.parametrize(
    "owner",
    [
        {"choice": "otto", "confidence": 0.9, "probabilities": [0.9]},
        {"choice": ["otto"], "confidence": 0.9},
        {"choice": "otto", "confidence": "high"},
    ],
    ids=["probabilities a list", "choice a list", "confidence text"],
)
async def test_an_answer_of_the_wrong_shape_returns_none_rather_than_raising(
    jev: AsyncMock, owner: dict[str, Any]
):
    answers: dict[str, Any] = _answers([1.0] * len(PROMPTS), [OTTO_SURE] * len(PROMPTS))
    answers["owner_0"] = owner
    jev.return_value = _result(answers)

    assert await _judge() is None


async def test_an_owner_outside_the_choices_returns_none(jev: AsyncMock):
    jev.return_value = _result(
        _answers(
            [1.0] * 4,
            [_owner("tpl-somebody-else", 0.9, confidence=0.8)] + [OTTO_SURE] * 3,
        )
    )

    assert await _judge() is None


async def test_a_state_too_long_to_send_whole_returns_none(
    jev: AsyncMock, mocker: MockerFixture
):
    mocker.patch.object(plan, "MAX_TRANSCRIPT_CHARS", 40_000)
    mocker.patch.object(plan, "TRANSCRIPT_BYTE_STEPS", (40_000,))
    long_transcript = "word " * 8_000

    judged = await plan.judge_plan(long_transcript, "Founder/CEO", [], PROMPTS, TEAM)

    assert judged is None
    jev.assert_not_awaited()


def _big_team(size: int) -> list[RecommendedExpert]:
    return [
        RecommendedExpert(
            template_id=f"tpl-{i:02d}-0000-0000-0000-000000000000",
            name=f"Expert {i}",
            role="Marketing",
            workflow_names=[f"Workflow {i} with a long descriptive name"] * 3,
        )
        for i in range(size)
    ]


async def test_a_long_team_offers_jev_its_first_six(jev: AsyncMock):
    jev.return_value = _result(_answers([1.0] * 4, [OTTO_SURE] * 4))

    await _judge(team=_big_team(16))

    _, state, questions = jev.await_args_list[0].args
    assert len(state["team"]) == plan.MAX_OWNER_CANDIDATES
    assert len(questions["owner_0"].criteria) == plan.MAX_OWNER_CANDIDATES + 1


# A recommendation as the team job stores it: uuid template ids, roster
# roles and up to three workflow names each.
REALISTIC_TEAM = [
    RecommendedExpert(
        template_id=f"{index:08x}-4a1b-4c2d-8e3f-0123456789ab",
        name=name,
        role=role,
        workflow_names=workflows,
    )
    for index, (name, role, workflows) in enumerate(
        [
            (
                "Maria",
                "Marketing",
                [
                    "Weekly SEO content calendar and briefs",
                    "Rewrite product listings for search",
                    "Social post drafts in your voice",
                ],
            ),
            (
                "Max",
                "Sales",
                [
                    "Find and qualify 25 lookalike companies",
                    "Personalised cold outreach sequences",
                    "CRM cleanup and deal follow-ups",
                ],
            ),
            (
                "Mina",
                "Bookkeeping",
                [
                    "Monthly expense categorisation",
                    "Invoice chasing with polite reminders",
                    "Cash flow summary every Monday",
                ],
            ),
            (
                "Olive",
                "Ops",
                [
                    "Weekly checklist tracker that chases owners",
                    "Vendor and supplier follow-ups",
                    "Monday metrics summary from your tools",
                ],
            ),
            (
                "Eli",
                "Executive Assistant",
                [
                    "Inbox triage with draft replies",
                    "Calendar scheduling without back and forth",
                    "Meeting prep briefs",
                ],
            ),
            (
                "Rae",
                "Recruiting",
                [
                    "Screen applicants against the role",
                    "Draft job posts",
                    "Interview scheduling and reminders",
                ],
            ),
        ]
    )
]


@pytest.mark.parametrize(
    "transcript",
    ["日" * 9_000, "😀" * 9_000, "é" * 9_000, "word " * 9_000],
    ids=["three-byte script", "four-byte emoji", "two-byte accents", "ascii"],
)
def test_the_longest_plan_the_pipeline_writes_fits_the_jev_budget(transcript: str):
    prompts = [
        SuggestedPrompt(
            title="t" * intro.MAX_TITLE_CHARS, prompt="p" * intro.MAX_PROMPT_CHARS
        )
        for _ in range(intro.MAX_PROMPTS)
    ]
    pain_points = ["Social media", "Finding leads", "Reports & data", "Scheduling"]

    candidates = REALISTIC_TEAM[: plan.MAX_OWNER_CANDIDATES]
    questions = plan._questions(prompts, candidates)
    state = plan._fitting_state(
        transcript, "Founder/CEO", pain_points, prompts, candidates, questions
    )

    assert state is not None
    assert len(state["team"]) == plan.MAX_OWNER_CANDIDATES
    assert all(
        len(task["prompt"]) == plan.MAX_PROMPT_STATE_CHARS for task in state["tasks"]
    )
    assert len(state["transcript"]) <= plan.MAX_TRANSCRIPT_CHARS
    assert len(state["transcript"].encode()) <= plan.MAX_TRANSCRIPT_BYTES
    assert not prepare_state(state, questions).truncated


async def test_long_tasks_in_a_three_byte_script_still_reach_jev_with_less_transcript(
    jev: AsyncMock,
):
    # Tasks at their length caps in Japanese leave too little room for the
    # whole transcript cut: it gives way, and Jev is still asked.
    prompts = [
        SuggestedPrompt(
            title="日" * intro.MAX_TITLE_CHARS, prompt="日" * intro.MAX_PROMPT_CHARS
        )
        for _ in range(intro.MAX_PROMPTS)
    ]
    jev.return_value = _result(
        _answers(
            [3.0] * len(prompts),
            [_owner(REALISTIC_TEAM[0].template_id, 0.9, confidence=0.9)] * len(prompts),
        )
    )

    judged = await plan.judge_plan(
        "日" * 9_000, "Founder/CEO", [], prompts, REALISTIC_TEAM
    )

    assert judged is not None
    jev.assert_awaited_once()
    sent_state = jev.await_args.args[1]
    assert len(sent_state["transcript"].encode()) < plan.MAX_TRANSCRIPT_BYTES
    assert all(
        len(task["prompt"].encode()) <= plan.MAX_PROMPT_STATE_BYTES
        for task in sent_state["tasks"]
    )


@pytest.mark.parametrize(
    "transcript",
    [
        "日" * 9_000,
        "😀" * 9_000,
        "a" + "日" * 9_000,
        "a" + "😀" * 9_000,
        "ab" + "😀" * 9_000,
        "é" * 9_000,
        "word " * 2_000,
    ],
    ids=[
        "three-byte script",
        "four-byte emoji",
        "three-byte script off the byte boundary",
        "four-byte emoji off the byte boundary",
        "four-byte emoji two bytes off",
        "two-byte accents",
        "ascii",
    ],
)
def test_a_long_transcript_is_cut_to_its_opening_on_a_whole_character(
    transcript: str,
):
    clipped = plan.clip_transcript(transcript)

    assert len(clipped) <= plan.MAX_TRANSCRIPT_CHARS
    assert len(clipped.encode()) <= plan.MAX_TRANSCRIPT_BYTES
    # Every kept character is whole and in place, and the cut keeps all the
    # characters that fit: the next one would cross a limit.
    assert transcript.startswith(clipped)
    following = transcript[len(clipped)]
    assert (
        len(clipped) == plan.MAX_TRANSCRIPT_CHARS
        or len((clipped + following).encode()) > plan.MAX_TRANSCRIPT_BYTES
    )


def test_a_short_transcript_is_sent_whole_without_its_outer_whitespace():
    assert plan.clip_transcript("  I sell stickers on Etsy. 日本\n") == (
        "I sell stickers on Etsy. 日本"
    )
