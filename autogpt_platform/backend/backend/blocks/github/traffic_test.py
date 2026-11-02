import asyncio
from http import HTTPStatus
from typing import Any
from unittest import mock

import pytest

from backend.blocks.github import traffic
from backend.blocks.github._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.github.traffic import (
    TEST_CLONES_PAYLOAD,
    TEST_POPULAR_PATHS_PAYLOAD,
    TEST_REFERRERS_PAYLOAD,
    TEST_VIEWS_PAYLOAD,
    GithubGetRepositoryTrafficBlock,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import (
    BlockExecutionError,
    BlockInputError,
    BlockUnknownError,
)
from backend.util.request import HTTPClientError, http_status_error

TRAFFIC_URL = "https://api.github.com/repos/owner/repo/traffic"
VIEWS_URL = f"{TRAFFIC_URL}/views"
CLONES_URL = f"{TRAFFIC_URL}/clones"
REFERRERS_URL = f"{TRAFFIC_URL}/popular/referrers"
PATHS_URL = f"{TRAFFIC_URL}/popular/paths"

NO_PUSH_ACCESS_MESSAGE = (
    "GitHub only shows traffic to people with push access to owner/repo. "
    "Connect a GitHub account that can push to it; a fine-grained token also "
    "needs the Administration (read) permission."
)


def _github_error(status: int, message: str) -> HTTPClientError:
    """The error `Requests` raises for a GitHub error response."""
    body = (
        f'{{"message":"{message}","documentation_url":'
        f'"https://docs.github.com/rest/metrics/traffic#get-page-views",'
        f'"status":"{status}"}}'
    )
    error = http_status_error(status, HTTPStatus(status).phrase, body.encode())
    assert isinstance(error, HTTPClientError)
    return error


class _FakeApi:
    """Serves canned responses by URL and records every request."""

    def __init__(
        self,
        views: Any = TEST_VIEWS_PAYLOAD,
        clones: Any = TEST_CLONES_PAYLOAD,
        referrers: Any = TEST_REFERRERS_PAYLOAD,
        paths: Any = TEST_POPULAR_PATHS_PAYLOAD,
        error: Exception | None = None,
    ):
        self.responses = {
            VIEWS_URL: views,
            CLONES_URL: clones,
            REFERRERS_URL: referrers,
            PATHS_URL: paths,
        }
        self.error = error
        self.calls: dict[str, dict] = {}

    async def get(self, url: str, **kwargs):
        self.calls[url] = kwargs
        if self.error:
            raise self.error
        payload = self.responses[url]
        return mock.Mock(json=lambda: payload)


async def _run(api: _FakeApi, **inputs) -> dict[str, Any]:
    """Run the block end to end with only the HTTP layer faked."""
    with mock.patch.object(traffic, "get_api", return_value=api):
        return await _execute(**inputs)


async def _execute(**inputs) -> dict[str, Any]:
    block = GithubGetRepositoryTrafficBlock()
    input_data = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "repo_url": "owner/repo",
        **inputs,
    }
    return {
        name: value
        async for name, value in block.execute(
            input_data,
            credentials=TEST_CREDENTIALS,
            execution_context=ExecutionContext(),
        )
    }


# ── Requests ──


class TestRequests:
    async def test_reads_all_four_traffic_endpoints(self):
        api = _FakeApi()
        await _run(api)
        assert set(api.calls) == {VIEWS_URL, CLONES_URL, REFERRERS_URL, PATHS_URL}

    async def test_views_and_clones_are_daily_by_default(self):
        api = _FakeApi()
        await _run(api)
        assert api.calls[VIEWS_URL]["params"] == {"per": "day"}
        assert api.calls[CLONES_URL]["params"] == {"per": "day"}

    async def test_weekly_period_is_sent_as_per_week(self):
        api = _FakeApi()
        await _run(api, period="week")
        assert api.calls[VIEWS_URL]["params"] == {"per": "week"}
        assert api.calls[CLONES_URL]["params"] == {"per": "week"}

    async def test_referrers_and_paths_take_no_period(self):
        api = _FakeApi()
        await _run(api, period="week")
        assert api.calls[REFERRERS_URL]["params"] is None
        assert api.calls[PATHS_URL]["params"] is None

    async def test_urls_are_sent_as_built(self):
        # convert_urls would rewrite an api.github.com URL as if it were a
        # github.com one, turning /repos/owner/repo into /repos/repos/owner.
        with mock.patch.object(traffic, "get_api", return_value=_FakeApi()) as get_api:
            await _execute()
        assert (
            get_api.call_args_list
            == [mock.call(TEST_CREDENTIALS, convert_urls=False)] * 4
        )

    async def test_the_four_requests_are_in_flight_together(self):
        everyone_asked = asyncio.Event()

        class WaitForAllApi(_FakeApi):
            async def get(self, url: str, **kwargs):
                self.calls[url] = kwargs
                if len(self.calls) == 4:
                    everyone_asked.set()
                # Requests made one at a time would never get here.
                await asyncio.wait_for(everyone_asked.wait(), timeout=5)
                return await super().get(url, **kwargs)

        outputs = await _run(WaitForAllApi())
        assert outputs["views"] == TEST_VIEWS_PAYLOAD["count"]


# ── Repository input ──


class TestRepositoryInput:
    @pytest.mark.parametrize(
        "repo_url",
        [
            "owner/repo",
            "https://github.com/owner/repo",
            "https://github.com/owner/repo/",
            "http://www.github.com/owner/repo",
            "github.com/owner/repo.git",
            "  https://github.com/owner/repo  ",
        ],
    )
    async def test_urls_and_owner_repo_name_the_same_repository(self, repo_url):
        api = _FakeApi()
        await _run(api, repo_url=repo_url)
        assert all(url.startswith(f"{TRAFFIC_URL}/") for url in api.calls)

    @pytest.mark.parametrize(
        "repo_url",
        [
            "",
            "owner",
            "owner/repo/issues",
            "owner/../../user",
            "https://github.com/owner/repo/tree/main",
            "https://gitlab.com/owner/repo",
        ],
    )
    async def test_invalid_repository_is_rejected_before_any_request(self, repo_url):
        api = _FakeApi()
        with pytest.raises(BlockInputError, match="repository"):
            await _run(api, repo_url=repo_url)
        assert api.calls == {}

    async def test_unknown_period_is_rejected(self):
        api = _FakeApi()
        with pytest.raises(BlockInputError):
            await _run(api, period="month")
        assert api.calls == {}


# ── Parsing ──


class TestParsing:
    async def test_maps_github_responses_to_outputs(self):
        assert await _run(_FakeApi()) == {
            "views": 3270,
            "unique_visitors": 1416,
            "views_by_period": [
                {"timestamp": "2026-09-28T00:00:00Z", "count": 1810, "uniques": 840},
                {"timestamp": "2026-09-29T00:00:00Z", "count": 1460, "uniques": 692},
            ],
            "clones": 412,
            "unique_cloners": 87,
            "clones_by_period": [
                {"timestamp": "2026-09-28T00:00:00Z", "count": 230, "uniques": 51},
                {"timestamp": "2026-09-29T00:00:00Z", "count": 182, "uniques": 44},
            ],
            "referrers": [
                {"referrer": "github.com", "count": 1203, "uniques": 655},
                {"referrer": "Google", "count": 421, "uniques": 302},
            ],
            "popular_content": [
                {
                    "path": "/owner/repo",
                    "title": "Overview",
                    "url": "https://github.com/owner/repo",
                    "count": 2010,
                    "uniques": 1033,
                },
                {
                    "path": "/owner/repo/issues",
                    "title": "/issues",
                    "url": "https://github.com/owner/repo/issues",
                    "count": 288,
                    "uniques": 120,
                },
            ],
        }

    async def test_repository_without_traffic(self):
        outputs = await _run(
            _FakeApi(
                views={"count": 0, "uniques": 0, "views": []},
                clones={"count": 0, "uniques": 0, "clones": []},
                referrers=[],
                paths=[],
            )
        )
        assert outputs == {
            "views": 0,
            "unique_visitors": 0,
            "views_by_period": [],
            "clones": 0,
            "unique_cloners": 0,
            "clones_by_period": [],
            "referrers": [],
            "popular_content": [],
        }

    async def test_missing_and_null_fields_degrade_to_zero_and_empty(self):
        outputs = await _run(
            _FakeApi(
                views={},
                clones={
                    "count": None,
                    "clones": [{"timestamp": "2026-09-28T00:00:00Z"}],
                },
                referrers=[{"referrer": "Google"}],
                paths=[{"count": 3, "title": None}],
            )
        )
        assert outputs["views"] == 0
        assert outputs["unique_visitors"] == 0
        assert outputs["views_by_period"] == []
        assert outputs["clones"] == 0
        assert outputs["clones_by_period"] == [
            {"timestamp": "2026-09-28T00:00:00Z", "count": 0, "uniques": 0}
        ]
        assert outputs["referrers"] == [
            {"referrer": "Google", "count": 0, "uniques": 0}
        ]
        assert outputs["popular_content"] == [
            {"path": "", "title": "", "url": "", "count": 3, "uniques": 0}
        ]

    @pytest.mark.parametrize(
        ("path", "url"),
        [
            (
                "/owner/repo/blob/main/README.md",
                "https://github.com/owner/repo/blob/main/README.md",
            ),
            ("owner/repo/wiki", "https://github.com/owner/repo/wiki"),
        ],
    )
    async def test_popular_content_links_to_the_page(self, path, url):
        outputs = await _run(_FakeApi(paths=[{"path": path, "count": 1, "uniques": 1}]))
        assert outputs["popular_content"][0]["url"] == url


# ── Errors ──


class TestErrors:
    @pytest.mark.parametrize(
        "message",
        [
            # What GitHub sends a token without push access (checked live)
            "Must have push access to repository",
            # A fine-grained token without the Administration (read) permission
            "Resource not accessible by personal access token",
        ],
    )
    async def test_no_access_to_traffic_explains_what_is_needed(self, message):
        error = _github_error(403, message)
        with pytest.raises(BlockExecutionError) as caught:
            await _run(_FakeApi(error=error))
        assert str(caught.value) == NO_PUSH_ACCESS_MESSAGE
        assert caught.value.__cause__ is error

    async def test_repository_not_found(self):
        error = _github_error(404, "Not Found")
        with pytest.raises(BlockExecutionError) as caught:
            await _run(_FakeApi(error=error))
        assert str(caught.value) == (
            "GitHub can't find owner/repo, or the connected GitHub account can't "
            "see it. Check the repository name, and that the account or token has "
            "access to the repository."
        )
        assert caught.value.__cause__ is error

    async def test_rate_limit_keeps_githubs_message(self):
        # A rate-limited 403 is not about access, so it must not say it is.
        error = _github_error(403, "API rate limit exceeded for user ID 1.")
        with pytest.raises(BlockUnknownError, match="API rate limit exceeded"):
            await _run(_FakeApi(error=error))

    async def test_rejected_credentials_keep_their_status(self):
        # AutoPilot asks the user to reconnect when it finds a 401 on the
        # error's cause chain, so a 401 must reach it unchanged.
        error = _github_error(401, "Bad credentials")
        with pytest.raises(BlockUnknownError) as caught:
            await _run(_FakeApi(error=error))
        assert caught.value.__cause__ is error
        assert error.status_code == 401
