"""
DataForSEO Site Audit block: crawl a site and report its technical SEO issues.
"""

from asyncio import get_running_loop, sleep, wait_for

from pydantic import Field

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
from backend.util.exceptions import BlockInputError

from ._api import DataForSeoClient, DataForSeoError, record_cost
from ._config import dataforseo
from ._onpage import AuditedPage, count_issues, parse_page
from ._targets import normalise_domain

TASK_POST_PATH = "/v3/on_page/task_post"
SUMMARY_PATH = "/v3/on_page/summary/{task_id}"
PAGES_PATH = "/v3/on_page/pages"
FORCE_STOP_PATH = "/v3/on_page/force_stop"
POLL_INTERVAL_SECONDS = 15
# Otto's direct block runs are cancelled after 300 seconds. The crawl wait plus
# the stop wait leaves a minute for the requests around them.
MAX_WAIT_SECONDS = 210
STOP_WAIT_SECONDS = 30
# The page list comes last; if DataForSEO is slow to send it, return the crawl
# summary without it rather than run past the limit with nothing to show.
PAGES_TIMEOUT_SECONDS = 30


class CrawlState(BaseModel):
    """What is known about a crawl so far, kept as it goes so that billing
    still has it when a later request fails."""

    summary: dict = Field(default_factory=dict)
    stop_requested: bool = False


class DataForSeoSiteAuditBlock(Block):
    """Crawls a website and lists each page's technical SEO issues."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = dataforseo.credentials_field(
            description="DataForSEO credentials (username and password)"
        )
        target: str = SchemaField(description="Domain to crawl, e.g. 'example.com'")
        max_crawl_pages: int = SchemaField(
            description="Most pages to crawl (billed per page crawled)",
            default=100,
            ge=1,
            le=1000,
        )
        max_wait_seconds: int = SchemaField(
            description=(
                "How long to let the crawl run. A crawl still running then is "
                "stopped, and the pages crawled so far are returned"
            ),
            default=180,
            ge=0,
            le=MAX_WAIT_SECONDS,
        )

    class Output(BlockSchemaOutput):
        complete: bool = SchemaField(
            description=(
                "False when the crawl was stopped at max_wait_seconds, or failed, "
                "before reaching every page up to max_crawl_pages"
            )
        )
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
                "click_depth: clicks from the home page. Empty if DataForSEO did "
                "not send the page list in time"
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
                ("complete", True),
                ("pages_crawled", 5),
                ("onpage_score", 91.5),
                (
                    "issue_counts",
                    {"no_image_alt": 5, "no_h1_tag": 1, "duplicate_title": 1},
                ),
                ("pages", [_TEST_AUDITED_PAGE]),
            ],
            test_mock={
                "_start_crawl": lambda *args, **kwargs: ("task-1", 0.0),
                "_summary": lambda *args, **kwargs: _TEST_SUMMARY,
                "_pages": lambda *args, **kwargs: [_TEST_PAGE],
            },
            effect=BlockEffect.READ,
        )

    async def _start_crawl(
        self, client: DataForSeoClient, target: str, max_crawl_pages: int
    ) -> tuple[str, float]:
        """Post the crawl; return its ID and the USD DataForSEO took upfront."""
        await client.post_task(
            TASK_POST_PATH, {"target": target, "max_crawl_pages": max_crawl_pages}
        )
        if not client.last_task_id:
            raise DataForSeoError("DataForSEO did not create the crawl")
        return client.last_task_id, client.last_cost_usd

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
        task_id, upfront_usd = await self._start_crawl(
            client, self._target(input_data), input_data.max_crawl_pages
        )
        state = CrawlState()
        try:
            await self._crawl(client, task_id, input_data.max_wait_seconds, state)
        finally:
            # Bill even when polling fails: DataForSEO has already been paid.
            record_cost(
                self, crawl_cost(upfront_usd, input_data.max_crawl_pages, state)
            )
        pages = await self._pages_in_time(client, task_id, input_data.max_crawl_pages)
        summary = state.summary
        metrics = summary.get("page_metrics") or {}
        yield "complete", _is_finished(summary) and not _ended_early(summary)
        yield "pages_crawled", _pages_crawled(summary)
        yield "onpage_score", metrics.get("onpage_score")
        yield "issue_counts", count_issues(metrics, pages)
        yield "pages", [parse_page(p) for p in pages]

    async def _pages_in_time(
        self, client: DataForSeoClient, task_id: str, limit: int
    ) -> list:
        try:
            return await wait_for(
                self._pages(client, task_id, limit), PAGES_TIMEOUT_SECONDS
            )
        except TimeoutError:
            return []

    def _target(self, input_data: Input) -> str:
        target = normalise_domain(input_data.target)
        if not target:
            raise BlockInputError(
                message="Give a domain to crawl, e.g. 'example.com'",
                block_name=self.name,
                block_id=self.id,
            )
        return target

    async def _crawl(
        self,
        client: DataForSeoClient,
        task_id: str,
        max_wait_seconds: int,
        state: CrawlState,
    ) -> None:
        """Wait for the crawl; stop it at the limit so it ends, and is billed, now."""
        await self._wait_for_crawl(client, task_id, max_wait_seconds, state)
        if _is_finished(state.summary):
            return
        await self._stop_crawl(client, task_id)
        state.stop_requested = True
        await self._wait_for_crawl(client, task_id, STOP_WAIT_SECONDS, state)

    async def _stop_crawl(self, client: DataForSeoClient, task_id: str) -> None:
        await client.post_task(FORCE_STOP_PATH, {"id": task_id})

    async def _wait_for_crawl(
        self,
        client: DataForSeoClient,
        task_id: str,
        max_wait_seconds: int,
        state: CrawlState,
    ) -> None:
        loop = get_running_loop()
        deadline = loop.time() + max_wait_seconds
        waited = 0.0
        state.summary = await self._summary(client, task_id)
        while not _is_finished(state.summary):
            # Bounded by the sleeps taken and by the clock, which also counts
            # the time the requests themselves took.
            remaining = min(max_wait_seconds - waited, deadline - loop.time())
            if remaining <= 0:
                break
            interval = min(POLL_INTERVAL_SECONDS, remaining)
            await sleep(interval)
            waited += interval
            state.summary = await self._summary(client, task_id)


def _is_finished(summary: dict) -> bool:
    return summary.get("crawl_progress") == "finished"


def _ended_early(summary: dict) -> bool:
    """Stopped at the wait limit, or ended by a DataForSEO crawler failure."""
    return summary.get("crawl_stop_reason") in ("force_stopped", "unexpected_exception")


def _pages_crawled(summary: dict) -> int:
    return (summary.get("crawl_status") or {}).get("pages_crawled") or 0


def crawl_cost(upfront_usd: float, max_crawl_pages: int, state: CrawlState) -> float:
    """What the crawl really costs.

    DataForSEO takes payment for max_crawl_pages when the crawl is posted and
    refunds the pages it didn't crawl once the crawl finishes, including when
    it was stopped. So once finished or stopped, bill the pages crawled (as
    last seen, for a stop that hasn't landed yet); otherwise the crawl may
    still run to max_crawl_pages, and the whole upfront amount is spent.
    """
    if not (_is_finished(state.summary) or state.stop_requested):
        return upfront_usd
    crawled = min(_pages_crawled(state.summary), max_crawl_pages)
    return upfront_usd / max_crawl_pages * crawled


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
