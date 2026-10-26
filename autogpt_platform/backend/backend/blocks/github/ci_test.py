from unittest import mock

import pytest

from backend.blocks.github import ci
from backend.blocks.github._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.github.ci import (
    SEARCH_PATTERN_MAX_LENGTH,
    SEARCH_TIMEOUT_SECONDS,
    GithubGetCIResultsBlock,
)

CHECK_RUNS = [
    {"name": "lint", "output_text": "starting\nerror: unused import\ndone"},
    {"name": "test", "output_text": ""},
]


class _RecordingPattern:
    """Stands in for a compiled pattern: records each search's timeout and
    advances the test's clock by ``seconds_per_search``."""

    def __init__(
        self,
        clock: "_Clock | None" = None,
        seconds_per_search: float = 0.0,
        raises: type[Exception] | None = None,
    ):
        self.timeouts: list[float] = []
        self.clock = clock
        self.seconds_per_search = seconds_per_search
        self.raises = raises

    def search(self, line: str, *, timeout: float) -> None:
        self.timeouts.append(timeout)
        if self.clock:
            self.clock.now += self.seconds_per_search
        if self.raises:
            raise self.raises


class _Clock:
    now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_search_in_logs_finds_matching_lines_with_context():
    matched = await GithubGetCIResultsBlock.search_in_logs(CHECK_RUNS, r"ERROR:")

    assert matched == [
        {
            "check_name": "lint",
            "line_number": 2,
            "line": "error: unused import",
            "context": ["starting", "error: unused import", "done"],
        }
    ]


@pytest.mark.asyncio
async def test_search_in_logs_shares_one_time_budget_across_the_lines(mocker):
    clock = _Clock()
    mocker.patch("backend.blocks.github.ci.monotonic", clock.monotonic)
    pattern = _RecordingPattern(clock, seconds_per_search=SEARCH_TIMEOUT_SECONDS / 2)
    mocker.patch("backend.blocks.github.ci.regex.compile", return_value=pattern)

    # The lint check has three lines: each search gets what the last left,
    # and the third is refused before it's searched
    with pytest.raises(ValueError, match="timed out"):
        await GithubGetCIResultsBlock.search_in_logs(CHECK_RUNS[:1], "error")

    assert pattern.timeouts == [SEARCH_TIMEOUT_SECONDS, SEARCH_TIMEOUT_SECONDS / 2]


@pytest.mark.asyncio
async def test_search_in_logs_stops_once_the_time_is_used_up(mocker):
    mocker.patch.object(ci, "SEARCH_TIMEOUT_SECONDS", 0)

    with pytest.raises(ValueError, match="timed out"):
        await GithubGetCIResultsBlock.search_in_logs(CHECK_RUNS, "error")


@pytest.mark.asyncio
async def test_search_in_logs_surfaces_a_regex_timeout_as_an_input_error(mocker):
    mocker.patch(
        "backend.blocks.github.ci.regex.compile",
        return_value=_RecordingPattern(raises=TimeoutError),
    )

    with pytest.raises(ValueError, match="timed out"):
        await GithubGetCIResultsBlock.search_in_logs(CHECK_RUNS, "error")


@pytest.mark.asyncio
async def test_search_in_logs_reports_an_invalid_pattern():
    with pytest.raises(ValueError, match="Invalid search pattern"):
        await GithubGetCIResultsBlock.search_in_logs(CHECK_RUNS, "(")


@pytest.mark.asyncio
async def test_search_in_logs_rejects_an_oversized_pattern_before_compiling(mocker):
    compile_pattern = mocker.patch("backend.blocks.github.ci.regex.compile")

    with pytest.raises(ValueError, match="too large"):
        await GithubGetCIResultsBlock.search_in_logs(
            CHECK_RUNS, "a" * (SEARCH_PATTERN_MAX_LENGTH + 1)
        )

    compile_pattern.assert_not_called()


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
