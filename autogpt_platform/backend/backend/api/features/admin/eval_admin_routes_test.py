"""Tests for the admin Evals routes, on a small eval folder written per test
and on the eval's own checked-in baselines."""

import json
from pathlib import Path
from typing import Any

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from . import eval_files
from .eval_admin_routes import router

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

REVIEWERS = ["claude:claude-opus-5-5", "cursor:grok-4.7-medium-fast"]
MODELS = {
    "backend": "claude-native",
    "otto": "claude-opus-5-5",
    "user": "claude-sonnet-5-5",
    "reviewers": REVIEWERS,
}


@pytest.fixture(autouse=True)
def admin_auth(mock_jwt_admin):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def eval_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A baseline with its report, then a local run of the same setup."""
    root = tmp_path / "tasks"
    cases = root / "data" / "cases"
    cases.mkdir(parents=True)
    (cases / "must_ask.json").write_text(json.dumps([_case_def("must_ask_001")]))
    (cases / "simple_qa.json").write_text(json.dumps([_case_def("simple_qa_001")]))
    _write(root / "baseline_native_run1.json", _summary(pass_rate=50.0, fail=1))
    (root / "report_native_run1.md").write_text(BASELINE_REPORT)
    run = root / "runs" / "20261110-090000"
    run.mkdir(parents=True)
    _write(run / "summary.json", _summary(pass_rate=50.0, fail=1))
    lines = [
        _result("must_ask_001", "fail", 30.0, "Otto never asked."),
        _result("simple_qa_001", "pass", 100.0, "Answered directly."),
    ]
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    monkeypatch.setattr(eval_files, "EVAL_DIR", root)
    return root


def test_lists_runs_newest_local_first(eval_dir: Path) -> None:
    response = client.get("/admin/evals/runs")
    assert response.status_code == 200
    runs = response.json()["runs"]
    assert [r["id"] for r in runs] == [
        "task-20261110-090000",
        "task-baseline_native_run1",
    ]
    assert runs[0]["case_coverage"] == "all"
    assert runs[1]["case_coverage"] == "not_passed"
    assert runs[1]["label"] == "Baseline native run 1"
    assert runs[1]["started_at"].startswith("2026-11-09T08:41:47")
    assert response.json()["commands"]


def test_local_run_parses_with_its_comparison(eval_dir: Path) -> None:
    response = client.get("/admin/evals/runs/task-20261110-090000")
    assert response.status_code == 200
    detail = response.json()
    failed = detail["cases"][0]
    assert failed["case_id"] == "must_ask_001"
    assert failed["verdict"] == "fail"
    assert failed["ratings"][0]["reasons"] == ["Otto never asked."]
    assert failed["observed"] == "start_task"
    assert failed["expected"]["behaviour"] == "ask"
    assert failed["request"] == "can you book it"
    assert failed["checks"] == [
        {"name": "asks_question", "passed": False, "detail": ""}
    ]
    assert failed["turns"][0]["tool_calls"][0]["name"] == "start_task"
    # The baseline report listed simple_qa_001 as failed and left must_ask_001
    # out, so it was passed by every reviewer then.
    assert failed["previous_verdict"] == "pass"
    comparison = detail["comparison"]
    assert comparison["previous_id"] == "task-baseline_native_run1"
    assert comparison["cases_compared"] is True
    assert [c["case_id"] for c in comparison["regressed"]] == ["must_ask_001"]
    assert [c["case_id"] for c in comparison["improved"]] == ["simple_qa_001"]
    assert [p["current"] for p in detail["history"]] == [False, True]
    assert "Repeat this run" in [c["label"] for c in detail["commands"]]


def test_baseline_reads_its_report(eval_dir: Path) -> None:
    response = client.get("/admin/evals/runs/task-baseline_native_run1")
    assert response.status_code == 200
    detail = response.json()
    assert detail["comparison"] is None
    assert detail["note"]
    [case] = detail["cases"]
    assert case["case_id"] == "simple_qa_001"
    assert case["detail"] == "report"
    assert case["category"] == "simple_qa"
    assert case["checks"] == [{"name": "max_tool_calls", "passed": False, "detail": ""}]
    assert [r["reviewer"] for r in case["ratings"]] == REVIEWERS
    assert case["ratings"][0]["reasons"] == ["It searched the web first."]
    assert case["ratings"][1]["score"] == 35.0


@pytest.mark.parametrize(
    "run_id, status",
    [
        ("task-unknown", 404),
        ("task-..", 422),
        ("task-baseline_native_run1.json", 422),
        ("..%2F..%2Fetc%2Fpasswd", 404),
        ("features-%2E%2E", 422),
        ("other-baseline_native_run1", 422),
    ],
)
def test_refuses_names_not_in_the_listing(
    eval_dir: Path, run_id: str, status: int
) -> None:
    assert client.get(f"/admin/evals/runs/{run_id}").status_code == status


def test_non_admin_is_refused(eval_dir: Path, mock_jwt_user) -> None:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    assert client.get("/admin/evals/runs").status_code == 403
    assert client.get("/admin/evals/runs/task-20261110-090000").status_code == 403


def test_checked_in_baselines_parse() -> None:
    runs = client.get("/admin/evals/runs").json()["runs"]
    by_id = {r["id"]: r for r in runs}
    assert by_id["task-baseline_native_run2"]["case_coverage"] == "not_passed"
    assert by_id["task-baseline_run1"]["case_coverage"] == "none"
    detail = client.get("/admin/evals/runs/task-baseline_native_run2").json()
    assert detail["run"]["stats"]["pass_rate"] == 79.6
    assert len(detail["categories"]) == 12
    assert detail["cases"] and all(c["verdict"] for c in detail["cases"])
    assert detail["comparison"]["previous_id"] == "task-baseline_native_run1"


BASELINE_REPORT = """# Task eval — 20261109-084147

## Cases not passed by every reviewer

- **simple_qa_001** (claude-opus-5-5 fail 40.0, grok-4.7-medium-fast fail 35.0) Where is Delhi
  - failed checks: max_tool_calls
  - claude-opus-5-5: It searched the web first.
  - grok-4.7-medium-fast: A web search for a fact it knew.
"""


def _summary(pass_rate: float, fail: int) -> dict[str, Any]:
    stats = {"cases": 2, "score": 65.0, "pass_rate": pass_rate, "partial": 0}
    stats["fail"] = fail
    return {
        "models": MODELS,
        "overall": {r: stats for r in REVIEWERS},
        "categories": {
            "must_ask": {r: stats for r in REVIEWERS},
            "simple_qa": {r: stats for r in REVIEWERS},
        },
        "tokens": {"prompt": 10, "completion": 2},
        "seconds_per_turn": 3.5,
    }


def _case_def(case_id: str) -> dict[str, Any]:
    category = case_id.rsplit("_", 1)[0]
    return {
        "id": case_id,
        "category": category,
        "world": "lumen_skincare",
        "title": f"{category} case",
        "turns": [{"user": "can you book it"}],
        "expected": {
            "behaviour": "ask" if category == "must_ask" else "reply",
            "must": ["asks which day"],
            "must_not": ["books a day itself"],
        },
        "human_example": {"reply": "Which day suits you?", "why": "Only Maya knows."},
    }


def _result(case_id: str, verdict: str, score: float, reason: str) -> dict[str, Any]:
    review = {"verdict": verdict, "score": score / 10, "reasons": [reason]}
    rating = {"review": review, "score": score, "verdict": verdict}
    return {
        "case_id": case_id,
        "category": case_id.rsplit("_", 1)[0],
        "title": f"{case_id} title",
        "world": "lumen_skincare",
        "run": {
            "case_id": case_id,
            "turns": [
                {
                    "user": "can you book it",
                    "reply": "On it.",
                    "tool_calls": [{"name": "start_task", "arguments": {"t": 1}}],
                    "ended_by": "task_started",
                }
            ],
        },
        "checks": (
            [] if verdict == "pass" else [{"name": "asks_question", "passed": False}]
        ),
        "review": review,
        "score": score,
        "verdict": verdict,
        "ratings": {r: rating for r in REVIEWERS},
    }


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data))
