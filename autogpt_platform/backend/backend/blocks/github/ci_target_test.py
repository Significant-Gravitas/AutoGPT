"""GithubGetCIResultsBlock target resolution: SHA vs PR number (#15292)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks.github.ci import GithubGetCIResultsBlock
from backend.util.request import HTTPClientError

REPO = "owner/repo"
COMMITS = f"https://api.github.com/repos/{REPO}/commits/"
PULLS = f"https://api.github.com/repos/{REPO}/pulls/"


def _api(routes: dict[str, dict | int]) -> MagicMock:
    """Fake GitHub client: URL -> JSON body, or an int HTTP error status."""

    async def get(url, *args, **kwargs):
        result = routes.get(url, 404)
        if isinstance(result, int):
            raise HTTPClientError(f"HTTP {result}", result)
        response = MagicMock()
        response.json.return_value = result
        return response

    api = MagicMock()
    api.get = AsyncMock(side_effect=get)
    return api


def _urls(api: MagicMock) -> list[str]:
    return [call.args[0] for call in api.get.await_args_list]


@pytest.mark.asyncio
async def test_all_digit_short_sha_resolves_as_commit():
    api = _api({COMMITS + "1234567": {"sha": "1234567" + "0" * 33}})
    sha = await GithubGetCIResultsBlock.get_commit_sha(api, REPO, "1234567")
    assert sha == "1234567" + "0" * 33
    assert _urls(api) == [COMMITS + "1234567"]


@pytest.mark.asyncio
async def test_all_digit_long_falls_back_to_pr_when_no_such_commit():
    api = _api({PULLS + "1234567": {"head": {"sha": "prhead"}}})
    sha = await GithubGetCIResultsBlock.get_commit_sha(api, REPO, "1234567")
    assert sha == "prhead"
    assert _urls(api) == [COMMITS + "1234567", PULLS + "1234567"]


@pytest.mark.asyncio
async def test_short_digit_string_is_pr_first():
    api = _api({PULLS + "123": {"head": {"sha": "prhead"}}})
    assert await GithubGetCIResultsBlock.get_commit_sha(api, REPO, "123") == "prhead"
    assert _urls(api) == [PULLS + "123"]


@pytest.mark.asyncio
async def test_short_digit_string_falls_back_to_commit():
    api = _api({COMMITS + "123456": {"sha": "123456abc"}})
    sha = await GithubGetCIResultsBlock.get_commit_sha(api, REPO, "123456")
    assert sha == "123456abc"
    assert _urls(api) == [PULLS + "123456", COMMITS + "123456"]


@pytest.mark.asyncio
async def test_int_and_hash_prefix_are_prs():
    api = _api({PULLS + "42": {"head": {"sha": "prhead"}}})
    assert await GithubGetCIResultsBlock.get_commit_sha(api, REPO, 42) == "prhead"
    assert await GithubGetCIResultsBlock.get_commit_sha(api, REPO, "#42") == "prhead"
    assert _urls(api) == [PULLS + "42", PULLS + "42"]


@pytest.mark.asyncio
async def test_hex_sha_is_returned_unchanged():
    api = _api({})
    assert await GithubGetCIResultsBlock.get_commit_sha(api, REPO, "abc123d") == (
        "abc123d"
    )
    api.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_404_error_is_not_swallowed():
    api = _api({COMMITS + "1234567": 401})
    with pytest.raises(HTTPClientError):
        await GithubGetCIResultsBlock.get_commit_sha(api, REPO, "1234567")


@pytest.mark.asyncio
async def test_run_passes_digit_string_through_unchanged(mocker):
    block = GithubGetCIResultsBlock()
    get_ci_results = mocker.patch.object(
        GithubGetCIResultsBlock,
        "get_ci_results",
        AsyncMock(return_value={"check_runs": [], "total_count": 0}),
    )
    from backend.blocks.github._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT

    input_data = GithubGetCIResultsBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT,  # type: ignore[arg-type]
        repo=REPO,
        target="1234567",
    )
    async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
        pass
    assert get_ci_results.await_args.args[2] == "1234567"
