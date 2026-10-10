"""Regression tests for GithubGetCIResultsBlock's overall verdict (#15291)."""

from unittest import mock

import pytest

from backend.blocks.github._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.github.ci import GithubGetCIResultsBlock


def _run(conclusion: str | None, status: str = "completed") -> dict:
    return {"id": 1, "name": "check", "status": status, "conclusion": conclusion}


async def _outputs(check_runs: list[dict]) -> dict:
    block = GithubGetCIResultsBlock()
    with mock.patch.object(
        GithubGetCIResultsBlock,
        "get_ci_results",
        mock.AsyncMock(
            return_value={"check_runs": check_runs, "total_count": len(check_runs)}
        ),
    ):
        input_data = block.Input(
            repo="owner/repo", target="abc123def", credentials=TEST_CREDENTIALS_INPUT
        )
        return {
            name: value
            async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS)
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conclusions",
    [
        ["cancelled"],
        ["success", "cancelled"],
        ["success", "stale"],
        ["success", "startup_failure"],
        ["success", "action_required"],
        ["success", "timed_out"],
        ["success", "failure"],
        ["success", None],
    ],
)
async def test_non_passing_conclusion_is_failure(conclusions):
    out = await _outputs([_run(c) for c in conclusions])
    assert out["overall_status"] == "completed"
    assert out["overall_conclusion"] == "failure"
    assert out["failed_checks"] == 1
    assert out["passed_checks"] + out["failed_checks"] == out["total_checks"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conclusions",
    [["success"], ["success", "skipped"], ["success", "neutral"]],
)
async def test_passing_conclusions_are_success(conclusions):
    out = await _outputs([_run(c) for c in conclusions])
    assert out["overall_conclusion"] == "success"
    assert out["failed_checks"] == 0
    assert out["passed_checks"] == len(conclusions)


@pytest.mark.asyncio
async def test_in_progress_run_is_pending_and_not_counted_as_failed():
    out = await _outputs([_run("success"), _run(None, status="in_progress")])
    assert out["overall_status"] == "pending"
    assert out["overall_conclusion"] == "pending"
    assert out["failed_checks"] == 0
    assert out["passed_checks"] == 1
