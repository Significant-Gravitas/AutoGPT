"""Tests for the feature eval's runs, on fakes: the worker adapter and the
settle table, the live tier on an in-memory task API, skipping, scoring and
review, the dry run and the report."""

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from backend.copilot.eval.tasks.__main__ import load_worlds
from backend.copilot.eval.tasks.models import (
    CheckResult,
    Review,
    Run,
    ToolCall,
    TurnRecord,
    World,
)
from backend.copilot.eval.tasks.runner import Models, turn_text
from backend.copilot.eval.tasks.simulator import Simulator
from backend.copilot.task_run import run_prompt
from backend.data.task import Task

from .__main__ import dry_run, load_cases
from .controls import (
    brief_checks,
    settle_table,
    settled_as,
    worker_case,
    worker_extra_checks,
)
from .cost import price_of
from .fakes import FakeApi, FakeBackend
from .judge import rate
from .live import run_live
from .models import FeatureCase, FeatureChecks, TaskSpec
from .play import Settings, play
from .records import make_task
from .report import write_report

WORLDS = load_worlds()


def _failed(checks: list[CheckResult]) -> list[str]:
    return [c.name for c in checks if not c.passed]


# -- controls ----------------------------------------------------------------------


def _task(**fields: Any) -> Task:
    spec = {
        "title": "Staff notice",
        "description": "Write it.",
        "goal": "Shifts know",
        "done_when": "Saved",
    }
    return make_task("c1", TaskSpec.model_validate(spec | fields))


def test_brief_checks_follow_run_prompt():
    task = _task()
    checks = brief_checks(
        FeatureChecks(brief_has=["title", "goal", "note"]), task, "Shorter please"
    )
    by_name = {c.name: c.passed for c in checks}
    prompt = run_prompt(task, "Shorter please")
    assert by_name["brief_has:title"] and by_name["brief_has:note"]
    assert by_name["brief_has:goal"] == ("Shifts know" in prompt)
    assert not brief_checks(FeatureChecks(brief_has=["note"]), task, None)[0].passed


def test_settle_table_covers_every_ending():
    checks = settle_table()
    # The mapping, every ending x result x open subtask, and the worker's words.
    assert len(checks) == 1 + 4 * 2 * 2 + 1
    assert checks[0].name == "turn_ending_mapping"
    assert checks[-1].name == "settle:completed:no_result:worker_words"


@pytest.mark.parametrize("note", [None, "Use the adbench sources"])
def test_worker_case_sends_production_run_prompt(note: str | None):
    case = FeatureCase(
        id="controls_x",
        area="controls",
        world="saffron_kitchens",
        title="t",
        needs="model",
        note=note,
    )
    task = _task()
    played = worker_case(case, task)
    sim = Simulator(WORLDS["saffron_kitchens"])
    assert turn_text(played.turns[0], sim) == run_prompt(task, note)
    assert sim.record.working


def _run(*calls: ToolCall, ended_by: str = "finished") -> Run:
    return Run(
        case_id="c",
        turns=[TurnRecord(user="u", tool_calls=list(calls), ended_by=ended_by)],
    )


def test_settled_as():
    assert (
        settled_as(_run(ToolCall(name="finish_task", arguments={"summary": "s"})))
        == "DONE"
    )
    assert (
        settled_as(_run(ToolCall(name="finish_task", failed=True), ended_by="reply"))
        == "FAILED"
    )
    assert settled_as(_run(ended_by="question")) == "NEEDS_INPUT"


@pytest.mark.parametrize(
    "closing,passed",
    [
        ("Shopify is not connected, so I could not pull the orders.", True),
        ("", False),  # nothing said: the generic reason is all there is
    ],
)
def test_fails_with_reason_reads_the_workers_last_words(closing: str, passed: bool):
    run = Run(
        case_id="c",
        turns=[TurnRecord(user="u", reply=closing, final_reply=closing)],
    )
    checks = worker_extra_checks(FeatureChecks(fails_with_reason=True), run)
    assert checks[0].passed is passed


async def test_worker_case_plays_on_the_engine_and_is_checked():
    case = next(c for c in load_cases(ids=["controls_008"]))

    async def engine(played, world: World, models: Models) -> Run:
        return _run(
            ToolCall(name="report_progress", arguments={"status_text": "Writing"}),
            ToolCall(
                name="write_workspace_file",
                arguments={"filename": "one-pager.md", "content": "adbench: iOS gap"},
            ),
            ToolCall(
                name="finish_task",
                arguments={"summary": "Done", "file": "one-pager.md"},
            ),
        )

    settings = Settings(WORLDS, models=Models(FakeBackend(), "o", "u"), engine=engine)
    result = await play(case, settings)
    names = {c.name: c.passed for c in result.checks}
    assert names["settles_as"] and names["mentions_any"] and names["finishes_task"]


# -- live tier ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,api,failed",
    [
        ("controls_009", FakeApi(), []),
        ("controls_010", FakeApi(), []),
        ("controls_011", FakeApi(), []),
        (
            "controls_011",
            FakeApi(keep_working_after_stop=True),
            ["nothing_after_stop"],
        ),
        ("suggestions_010", FakeApi(), []),
    ],
)
async def test_live_actions_on_a_fake_api(
    case_id: str, api: FakeApi, failed: list[str]
):
    case = load_cases(ids=[case_id])[0]
    checks, record = await run_live(case, api, "user-1")
    assert _failed(checks) == failed
    assert api.archived == [next(iter(api.tasks))]


async def test_play_skips_what_cannot_run():
    settings = Settings(WORLDS)
    live = await play(load_cases(ids=["controls_009"])[0], settings)
    model = await play(load_cases(ids=["suggestions_005"])[0], settings)
    assert "live database" in (live.skipped or "")
    assert "model" in (model.skipped or "")


# -- scoring, review, cost, report -------------------------------------------------


def test_rate():
    good, bad = CheckResult(name="a", passed=True), CheckResult(name="b", passed=False)
    assert rate([good], None).verdict == "pass"
    assert rate([good, bad], None).verdict == "fail"
    review = Review(verdict="pass", score=8, reasons=[])
    assert rate([good, bad], review).verdict == "partial"
    assert rate([good, bad], review).score == 65.0


async def test_suggestions_case_is_reviewed():
    backend = FakeBackend('{"verdict": "pass", "score": 9, "reasons": ["clear"]}')
    case = load_cases(ids=["suggestions_003"])[0].model_copy(update={"review": True})
    settings = Settings(WORLDS, reviewers=[("fake:m", backend, "m")])
    result = await play(case, settings)
    assert result.reviewed and result.ratings["fake:m"].review.reasons == ["clear"]
    assert "Ask Aarey Pure Dairy for a December paneer quote" in backend.sent[0]


def test_dry_run_prices_every_case():
    args = argparse.Namespace(
        backend="claude-native",
        model="claude-opus-5-5",
        drafter="claude:claude-haiku-5-5",
        reviewer=None,
        live_model="claude-opus-5-5",
    )
    out = dry_run(load_cases(), args)
    assert all(c.id in out for c in load_cases())
    assert "Estimated API-equivalent cost: $" in out
    assert price_of("claude:claude-opus-5-5") is not None
    assert price_of("cursor:grok-4.7-medium-fast") is None


async def test_report_is_written(tmp_path: Path):
    """A passed case is counted, a failed one is listed with its failed
    check, a skipped one is listed with why."""
    settings = Settings(WORLDS)
    passing = load_cases(ids=["suggestions_003"])[0]
    skipped = load_cases(ids=["controls_009"])[0]
    failing = passing.model_copy(
        update={
            "id": "suggestions_failing",
            "expected": passing.expected.model_copy(
                update={"checks": FeatureChecks(max_new=0)}
            ),
        }
    )
    results = [await play(c, settings) for c in (passing, failing, skipped)]
    summary = write_report(
        tmp_path, results, {"backend": "x", "tiers": ["none"]}, settings.names
    )
    report = (tmp_path / "report.md").read_text()
    assert "**suggestions_failing**" in report and "`max_new`" in report
    assert "**suggestions_003**" not in report.split("## Skipped")[0]
    assert "## Skipped" in report and "controls_009" in report
    assert summary["overall"]["code"]["cases"] == 2
    assert json.loads((tmp_path / "summary.json").read_text())["skipped"]
    assert (tmp_path / "report.html").read_text().startswith("<!doctype html>")
