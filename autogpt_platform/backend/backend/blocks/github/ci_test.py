import pytest

from backend.blocks.github import ci
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
    """Stands in for a compiled pattern and records each search's timeout."""

    def __init__(self, raises: type[Exception] | None = None):
        self.timeouts: list[float] = []
        self.raises = raises

    def search(self, line: str, *, timeout: float) -> None:
        self.timeouts.append(timeout)
        if self.raises:
            raise self.raises


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
async def test_search_in_logs_gives_every_search_the_time_left(mocker):
    pattern = _RecordingPattern()
    mocker.patch("backend.blocks.github.ci.regex.compile", return_value=pattern)

    await GithubGetCIResultsBlock.search_in_logs(CHECK_RUNS, "error")

    assert len(pattern.timeouts) == 3
    assert all(0 < timeout <= SEARCH_TIMEOUT_SECONDS for timeout in pattern.timeouts)
    assert pattern.timeouts == sorted(pattern.timeouts, reverse=True)


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
async def test_search_in_logs_rejects_an_oversized_pattern_before_compiling(mocker):
    compile_pattern = mocker.patch("backend.blocks.github.ci.regex.compile")

    with pytest.raises(ValueError, match="too large"):
        await GithubGetCIResultsBlock.search_in_logs(
            CHECK_RUNS, "a" * (SEARCH_PATTERN_MAX_LENGTH + 1)
        )

    compile_pattern.assert_not_called()
