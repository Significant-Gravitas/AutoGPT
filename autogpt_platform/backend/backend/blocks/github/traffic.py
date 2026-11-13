import asyncio
from typing import Any, Literal

from typing_extensions import TypedDict

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.request import HTTPClientError

from ._api import GITHUB_API_URL, get_api
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)
from ._utils import normalize_repo_path

GITHUB_WEB_URL = "https://github.com"

TrafficPeriod = Literal["day", "week"]

# What GitHub's 403 says when the caller may not read traffic: no push access
# ("Must have push access to repository"), or a fine-grained token without
# Administration (read) ("Resource not accessible by personal access token").
# Other 403s, such as rate limits or SSO enforcement, keep GitHub's own message.
_NO_TRAFFIC_ACCESS_MESSAGES = ("must have push access", "resource not accessible")


class TrafficCount(TypedDict):
    timestamp: str
    count: int
    uniques: int


class TrafficReferrer(TypedDict):
    referrer: str
    count: int
    uniques: int


class PopularContent(TypedDict):
    path: str
    title: str
    url: str
    count: int
    uniques: int


# Responses in the shapes the traffic endpoints return, for the self-test
TEST_VIEWS_PAYLOAD = {
    "count": 3270,
    "uniques": 1416,
    "views": [
        {"timestamp": "2026-09-28T00:00:00Z", "count": 1810, "uniques": 840},
        {"timestamp": "2026-09-29T00:00:00Z", "count": 1460, "uniques": 692},
    ],
}
TEST_CLONES_PAYLOAD = {
    "count": 412,
    "uniques": 87,
    "clones": [
        {"timestamp": "2026-09-28T00:00:00Z", "count": 230, "uniques": 51},
        {"timestamp": "2026-09-29T00:00:00Z", "count": 182, "uniques": 44},
    ],
}
TEST_REFERRERS_PAYLOAD = [
    {"referrer": "github.com", "count": 1203, "uniques": 655},
    {"referrer": "Google", "count": 421, "uniques": 302},
]
TEST_POPULAR_PATHS_PAYLOAD = [
    {"path": "/owner/repo", "title": "Overview", "count": 2010, "uniques": 1033},
    {"path": "/owner/repo/issues", "title": "/issues", "count": 288, "uniques": 120},
]


class GithubGetRepositoryTrafficBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="Repository URL or '{owner}/{repo}'",
            placeholder="https://github.com/owner/repo",
        )
        period: TrafficPeriod = SchemaField(
            description="Whether to break the views and clones down by day or by "
            "week. Days and weeks start at midnight UTC, and weeks start on Monday.",
            default="day",
        )

    class Output(BlockSchemaOutput):
        views: int = SchemaField(description="Views in the last 14 days")
        unique_visitors: int = SchemaField(
            description="Unique visitors in the last 14 days. Someone who visited "
            "on several days counts once."
        )
        views_by_period: list[TrafficCount] = SchemaField(
            description="Views and unique visitors for each day or week, "
            "oldest first"
        )
        clones: int = SchemaField(
            description="Git clones in the last 14 days. Counts full clones, "
            "not fetches."
        )
        unique_cloners: int = SchemaField(
            description="Unique cloners in the last 14 days"
        )
        clones_by_period: list[TrafficCount] = SchemaField(
            description="Clones and unique cloners for each day or week, "
            "oldest first"
        )
        referrers: list[TrafficReferrer] = SchemaField(
            description="Top 10 sites that sent visitors in the last 14 days, "
            "with the views and unique visitors from each"
        )
        popular_content: list[PopularContent] = SchemaField(
            description="Top 10 most viewed pages of the repository in the last "
            "14 days, with the views, unique visitors and github.com link of each"
        )

    def __init__(self):
        super().__init__(
            id="aa46e092-b04f-492a-ae13-bb920f1469c2",
            description="Get a GitHub repository's views and clones over the last "
            "14 days, with unique visitors and cloners, plus its top referring "
            "sites and most viewed pages. Views and clones are also broken down "
            "by day or week. GitHub only shows traffic to accounts with push "
            "access to the repository.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.MARKETING},
            input_schema=GithubGetRepositoryTrafficBlock.Input,
            output_schema=GithubGetRepositoryTrafficBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("views", 3270),
                ("unique_visitors", 1416),
                ("views_by_period", TEST_VIEWS_PAYLOAD["views"]),
                ("clones", 412),
                ("unique_cloners", 87),
                ("clones_by_period", TEST_CLONES_PAYLOAD["clones"]),
                ("referrers", TEST_REFERRERS_PAYLOAD),
                (
                    "popular_content",
                    [
                        {
                            **TEST_POPULAR_PATHS_PAYLOAD[0],
                            "url": "https://github.com/owner/repo",
                        },
                        {
                            **TEST_POPULAR_PATHS_PAYLOAD[1],
                            "url": "https://github.com/owner/repo/issues",
                        },
                    ],
                ),
            ],
            test_mock={
                "get_views": lambda *args, **kwargs: TEST_VIEWS_PAYLOAD,
                "get_clones": lambda *args, **kwargs: TEST_CLONES_PAYLOAD,
                "get_referrers": lambda *args, **kwargs: TEST_REFERRERS_PAYLOAD,
                "get_popular_content": lambda *args, **kwargs: (
                    TEST_POPULAR_PATHS_PAYLOAD
                ),
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def get_views(
        credentials: GithubCredentials, repo: str, period: TrafficPeriod
    ) -> dict:
        return await _get_traffic(credentials, repo, "views", {"per": period})

    @staticmethod
    async def get_clones(
        credentials: GithubCredentials, repo: str, period: TrafficPeriod
    ) -> dict:
        return await _get_traffic(credentials, repo, "clones", {"per": period})

    @staticmethod
    async def get_referrers(credentials: GithubCredentials, repo: str) -> list[dict]:
        return await _get_traffic(credentials, repo, "popular/referrers")

    @staticmethod
    async def get_popular_content(
        credentials: GithubCredentials, repo: str
    ) -> list[dict]:
        return await _get_traffic(credentials, repo, "popular/paths")

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            repo = normalize_repo_path(input_data.repo_url)
        except ValueError as e:
            raise BlockInputError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e

        try:
            views, clones, referrers, paths = await asyncio.gather(
                self.get_views(credentials, repo, input_data.period),
                self.get_clones(credentials, repo, input_data.period),
                self.get_referrers(credentials, repo),
                self.get_popular_content(credentials, repo),
            )
        except HTTPClientError as e:
            if message := _traffic_error_message(e, repo):
                raise BlockExecutionError(
                    message=message, block_name=self.name, block_id=self.id
                ) from e
            raise

        yield "views", views.get("count") or 0
        yield "unique_visitors", views.get("uniques") or 0
        yield "views_by_period", [
            _to_traffic_count(item) for item in views.get("views") or []
        ]
        yield "clones", clones.get("count") or 0
        yield "unique_cloners", clones.get("uniques") or 0
        yield "clones_by_period", [
            _to_traffic_count(item) for item in clones.get("clones") or []
        ]
        yield "referrers", [_to_referrer(item) for item in referrers or []]
        yield "popular_content", [_to_popular_content(item) for item in paths or []]


async def _get_traffic(
    credentials: GithubCredentials,
    repo: str,
    endpoint: str,
    params: dict[str, str] | None = None,
) -> Any:
    api = get_api(credentials, convert_urls=False)
    response = await api.get(
        f"{GITHUB_API_URL}/repos/{repo}/traffic/{endpoint}", params=params
    )
    return response.json()


def _traffic_error_message(error: HTTPClientError, repo: str) -> str | None:
    """A message for the errors the user can fix, or None to keep GitHub's own."""
    detail = str(error).lower()
    if error.status_code == 403 and any(
        message in detail for message in _NO_TRAFFIC_ACCESS_MESSAGES
    ):
        return (
            f"GitHub only shows traffic to people with push access to {repo}. "
            "Connect a GitHub account that can push to it; a fine-grained token "
            "also needs the Administration (read) permission."
        )
    if error.status_code == 404:
        return (
            f"GitHub can't find {repo}, or the connected GitHub account can't see "
            "it. Check the repository name, and that the account or token has "
            "access to the repository."
        )
    return None


def _to_traffic_count(item: dict) -> TrafficCount:
    return {
        "timestamp": item.get("timestamp") or "",
        "count": item.get("count") or 0,
        "uniques": item.get("uniques") or 0,
    }


def _to_referrer(item: dict) -> TrafficReferrer:
    return {
        "referrer": item.get("referrer") or "",
        "count": item.get("count") or 0,
        "uniques": item.get("uniques") or 0,
    }


def _to_popular_content(item: dict) -> PopularContent:
    path = item.get("path") or ""
    return {
        "path": path,
        "title": item.get("title") or "",
        "url": f"{GITHUB_WEB_URL}/{path.lstrip('/')}" if path else "",
        "count": item.get("count") or 0,
        "uniques": item.get("uniques") or 0,
    }
