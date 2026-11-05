"""
DataForSEO Site Audit block: crawl a site and report its technical SEO issues.
"""

from asyncio import sleep
from typing import Any

from backend.sdk import (
    BaseModel,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
    UserPasswordCredentials,
)

from ._api import DataForSeoClient, record_cost
from ._config import dataforseo

TASK_POST_PATH = "/v3/on_page/task_post"
SUMMARY_PATH = "/v3/on_page/summary/{task_id}"
PAGES_PATH = "/v3/on_page/pages"
POLL_INTERVAL_SECONDS = 15

# On-Page checks where a true value (or a non-zero page count) is a problem.
# The rest, such as is_https or seo_friendly_url, are true on a healthy page.
ISSUE_CHECKS = frozenset(
    {
        "no_title", "title_too_long", "title_too_short", "duplicate_title_tag",
        "irrelevant_title", "no_description", "irrelevant_description",
        "duplicate_meta_tags", "no_h1_tag", "is_4xx_code", "is_5xx_code",
        "is_broken", "is_redirect", "redirect_chain", "has_meta_refresh_redirect",
        "has_links_to_redirects", "canonical_to_broken", "canonical_to_redirect",
        "canonical_chain", "recursive_canonical", "is_orphan_page", "is_http",
        "https_to_http_links", "is_link_relation_conflict", "no_image_alt",
        "low_content_rate", "low_character_count", "low_readability_rate",
        "has_micromarkup_errors", "high_loading_time", "high_waiting_time",
        "large_page_size", "size_greater_than_3mb", "has_render_blocking_resources",
        "no_doctype", "no_encoding_meta_tag", "no_favicon", "deprecated_html_tags",
        "lorem_ipsum", "flash", "frame",
    }
)  # fmt: skip
# Page-level flags that sit beside the checks rather than inside them.
PAGE_FLAGS = ("duplicate_title", "duplicate_description", "duplicate_content", "broken_links", "broken_resources")  # fmt: skip


class AuditedPage(BaseModel):
    url: str
    status_code: int | None
    title: str
    description: str
    h1: list[str]
    canonical: str
    click_depth: int | None
    onpage_score: float | None
    issues: list[str]


class DataForSeoSiteAuditBlock(Block):
    """Crawls a website and lists each page's technical SEO issues."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = dataforseo.credentials_field(
            description="DataForSEO credentials (username and password)"
        )
        target: str = SchemaField(
            description="Domain to crawl, without https:// or www. (e.g. 'example.com')"
        )
        max_crawl_pages: int = SchemaField(
            description="Most pages to crawl (billed per page crawled)",
            default=100,
            ge=1,
            le=1000,
        )
        max_wait_seconds: int = SchemaField(
            description=(
                "How long to wait for the crawl. If it is still running, the pages "
                "crawled so far are returned; pass task_id to collect the rest later"
            ),
            default=300,
            ge=0,
            le=900,
        )
        task_id: str = SchemaField(
            description="Collect the results of an earlier crawl instead of starting one",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        task_id: str = SchemaField(description="The crawl's ID, kept for 30 days")
        finished: bool = SchemaField(description="Whether the crawl has finished")
        pages_crawled: int = SchemaField(description="How many pages were crawled")
        onpage_score: float | None = SchemaField(
            description="Site score, 0-100, where 100 means no critical issues"
        )
        issue_counts: dict[str, int] = SchemaField(
            description="How many crawled pages have each issue, most common first"
        )
        pages: list[AuditedPage] = SchemaField(
            description=(
                "Crawled pages, lowest score first, each with its issues (On-Page "
                "check names such as no_h1_tag, is_4xx_code, duplicate_title) and "
                "click_depth: clicks from the home page"
            )
        )

    def __init__(self):
        super().__init__(
            id="546df8df-3650-462b-8638-3ae9c055fa1d",
            description=(
                "Crawl a website with DataForSEO On-Page and report technical SEO "
                "issues per page: broken pages, redirects, titles, descriptions, H1s, "
                "canonicals, duplicates, orphan pages, speed and click depth"
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": dataforseo.get_test_credentials().model_dump(),
                "target": "example.com",
                "max_crawl_pages": 5,
            },
            test_credentials=dataforseo.get_test_credentials(),
            test_output=[
                ("task_id", "task-1"),
                ("finished", True),
                ("pages_crawled", 5),
                ("onpage_score", 91.5),
                ("issue_counts", {"no_image_alt": 5, "no_h1_tag": 1}),
                ("pages", [_TEST_AUDITED_PAGE]),
            ],
            test_mock={
                "_start_crawl": lambda *args, **kwargs: "task-1",
                "_summary": lambda *args, **kwargs: _TEST_SUMMARY,
                "_pages": lambda *args, **kwargs: [_TEST_PAGE],
            },
            effect=BlockEffect.READ,
        )

    async def _start_crawl(self, client: DataForSeoClient, input_data: Input) -> str:
        await client.post_task(
            TASK_POST_PATH,
            {
                "target": input_data.target.strip(),
                "max_crawl_pages": input_data.max_crawl_pages,
            },
        )
        record_cost(self, client)
        return client.last_task_id

    async def _summary(self, client: DataForSeoClient, task_id: str) -> dict:
        results = await client.get_task(SUMMARY_PATH.format(task_id=task_id))
        # A crawl that has not started yet has no result.
        return (results[0] if results else None) or {}

    async def _pages(self, client: DataForSeoClient, task_id: str, limit: int) -> list:
        results = await client.post_task(
            PAGES_PATH,
            {"id": task_id, "limit": limit, "order_by": ["onpage_score,asc"]},
        )
        return (results[0].get("items") if results else None) or []

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        client = DataForSeoClient(credentials)
        task_id = input_data.task_id.strip() or await self._start_crawl(
            client, input_data
        )
        summary = await self._wait_for_crawl(
            client, task_id, input_data.max_wait_seconds
        )
        pages = await self._pages(client, task_id, input_data.max_crawl_pages)
        metrics = summary.get("page_metrics") or {}
        yield "task_id", task_id
        yield "finished", summary.get("crawl_progress") == "finished"
        yield "pages_crawled", (summary.get("crawl_status") or {}).get(
            "pages_crawled"
        ) or 0
        yield "onpage_score", metrics.get("onpage_score")
        yield "issue_counts", count_issues(metrics)
        yield "pages", [parse_page(p) for p in pages]

    async def _wait_for_crawl(
        self, client: DataForSeoClient, task_id: str, max_wait_seconds: int
    ) -> dict:
        waited = 0
        summary = await self._summary(client, task_id)
        while summary.get("crawl_progress") != "finished" and waited < max_wait_seconds:
            await sleep(POLL_INTERVAL_SECONDS)
            waited += POLL_INTERVAL_SECONDS
            summary = await self._summary(client, task_id)
        return summary


def count_issues(metrics: dict[str, Any]) -> dict[str, int]:
    counts = {
        name: count
        for name, count in (metrics.get("checks") or {}).items()
        if name in ISSUE_CHECKS and count
    }
    counts |= {flag: metrics[flag] for flag in PAGE_FLAGS if metrics.get(flag)}
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def parse_page(page: dict[str, Any]) -> AuditedPage:
    meta = page.get("meta") or {}
    checks = page.get("checks") or {}
    issues = sorted(
        name for name, bad in checks.items() if name in ISSUE_CHECKS and bad
    )
    issues += [flag for flag in PAGE_FLAGS if page.get(flag)]
    return AuditedPage(
        url=page.get("url") or "",
        status_code=page.get("status_code"),
        title=meta.get("title") or "",
        description=meta.get("description") or "",
        h1=(meta.get("htags") or {}).get("h1") or [],
        canonical=meta.get("canonical") or "",
        click_depth=page.get("click_depth"),
        onpage_score=page.get("onpage_score"),
        issues=issues,
    )


_TEST_SUMMARY = {
    "crawl_progress": "finished",
    "crawl_status": {"pages_crawled": 5},
    "page_metrics": {
        "onpage_score": 91.5,
        "duplicate_title": 0,
        "checks": {"no_h1_tag": 1, "no_image_alt": 5, "is_https": 5},
    },
}
_TEST_PAGE = {
    "url": "https://example.com/collections/soy",
    "status_code": 200,
    "meta": {"title": "Soy candles", "htags": {"h2": ["Why soy"]}},
    "click_depth": 2,
    "onpage_score": 80.0,
    "checks": {"no_h1_tag": True, "no_image_alt": True, "is_https": True},
    "duplicate_title": True,
}
_TEST_AUDITED_PAGE = AuditedPage(
    url="https://example.com/collections/soy",
    status_code=200,
    title="Soy candles",
    description="",
    h1=[],
    canonical="",
    click_depth=2,
    onpage_score=80.0,
    issues=["no_h1_tag", "no_image_alt", "duplicate_title"],
)
