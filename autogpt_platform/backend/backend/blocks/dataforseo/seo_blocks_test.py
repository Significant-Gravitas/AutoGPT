"""Tests for the DataForSEO SERP, ranked keywords, backlinks, AI and audit blocks."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.util.exceptions import BlockInputError

from ._api import DataForSeoClient, DataForSeoError
from ._config import dataforseo
from ._onpage import count_issues, normalise_domain, parse_page
from .ai_mentions import DataForSeoAiMentionsBlock
from .ai_mentions import build_task as build_mentions_task
from .backlinks import DataForSeoBacklinksSummaryBlock
from .ranked_keywords import POSITION_FIELD, DataForSeoRankedKeywordsBlock
from .ranked_keywords import build_task as build_ranked_task
from .serp import parse_organic
from .site_audit import DataForSeoSiteAuditBlock, crawl_cost

CREDS = dataforseo.get_test_credentials()
CREDS_INPUT = CREDS.model_dump()


def _client() -> DataForSeoClient:
    return DataForSeoClient(CREDS)


class TestTaskResult:
    def test_returns_result_and_records_cost_and_id(self):
        client = _client()
        data = {
            "tasks": [
                {"id": "t1", "status_code": 20000, "cost": 0.02, "result": [{"a": 1}]}
            ]
        }
        assert client._task_result(200, data) == [{"a": 1}]
        assert client.last_cost_usd == 0.02
        assert client.last_task_id == "t1"

    def test_task_created_has_no_result(self):
        client = _client()
        data = {
            "tasks": [{"id": "t2", "status_code": 20100, "cost": 0.01, "result": None}]
        }
        assert client._task_result(200, data) == []
        assert client.last_task_id == "t2"

    def test_queued_task_is_not_ready_rather_than_an_error(self):
        data = {
            "tasks": [
                {"id": "t3", "status_code": 40602, "status_message": "Task In Queue."}
            ]
        }
        assert _client()._task_result(200, data) == []

    def test_failed_task_raises(self):
        data = {"tasks": [{"status_code": 40501, "status_message": "Invalid Field"}]}
        with pytest.raises(DataForSeoError, match="Invalid Field"):
            _client()._task_result(200, data)

    def test_http_error_raises(self):
        with pytest.raises(DataForSeoError, match="401"):
            _client()._task_result(401, {"status_message": "Unauthorized"})


class TestSerp:
    def test_positions_count_organic_listings_only(self):
        items = [
            {"type": "ai_overview"},
            {"type": "organic", "url": "https://a.com/x", "domain": "a.com"},
            {"type": "people_also_ask"},
            {"type": "organic", "url": "https://b.com/y", "domain": "b.com"},
        ]
        assert [(r.position, r.domain) for r in parse_organic(items)] == [
            (1, "a.com"),
            (2, "b.com"),
        ]

    def test_unresolved_google_redirect_takes_domain_from_breadcrumb(self):
        items = [
            {
                "type": "organic",
                "url": "https://google.com/goto?url=opaque",
                "domain": "google.com",
                "breadcrumb": "https://www.shop.com › blog › candles",
            },
            {
                "type": "organic",
                "url": "https://google.com/goto?url=opaque2",
                "domain": "google.com",
                "breadcrumb": "8 comments · 1 year ago",
            },
        ]
        first, second = parse_organic(items)
        assert (first.domain, first.url_resolved) == ("www.shop.com", False)
        assert (second.domain, second.url_resolved) == ("", False)


class TestRankedKeywords:
    def _input(self, **kwargs) -> DataForSeoRankedKeywordsBlock.Input:
        return DataForSeoRankedKeywordsBlock.Input(
            credentials=CREDS_INPUT, target=" example.com ", **kwargs
        )

    def test_position_range_becomes_one_and_filter(self):
        task = build_ranked_task(self._input(min_position=7, max_position=15))
        assert task["target"] == "example.com"
        assert task["filters"] == [
            [POSITION_FIELD, ">=", 7],
            "and",
            [POSITION_FIELD, "<=", 15],
        ]

    def test_single_bound_is_a_single_filter(self):
        task = build_ranked_task(self._input(max_position=3))
        assert task["filters"] == [POSITION_FIELD, "<=", 3]

    def test_no_bounds_sends_no_filter(self):
        assert "filters" not in build_ranked_task(self._input())

    async def test_inverted_range_is_refused(self):
        block = DataForSeoRankedKeywordsBlock()
        with pytest.raises(ValueError, match="min_position"):
            async for _ in block.run(
                self._input(min_position=15, max_position=7), credentials=CREDS
            ):
                pass


class TestAiMentions:
    def test_domain_and_keyword_are_both_targets(self):
        task = build_mentions_task(
            DataForSeoAiMentionsBlock.Input(
                credentials=CREDS_INPUT, domain="example.com", keyword="soy candles"
            )
        )
        assert task["target"] == [{"domain": "example.com"}, {"keyword": "soy candles"}]

    async def test_needs_a_domain_or_keyword(self):
        block = DataForSeoAiMentionsBlock()
        with pytest.raises(ValueError, match="domain, a keyword"):
            async for _ in block.run(
                DataForSeoAiMentionsBlock.Input(credentials=CREDS_INPUT, domain=" "),
                credentials=CREDS,
            ):
                pass


class TestSiteAudit:
    def test_counts_only_problem_checks(self):
        metrics = {
            "checks": {"no_h1_tag": 2, "is_https": 9, "no_title": 0},
            "duplicate_title": 3,
        }
        assert count_issues(metrics) == {"duplicate_title": 3, "no_h1_tag": 2}

    def test_page_issues_skip_healthy_checks(self):
        page = parse_page(
            {
                "url": "https://example.com/",
                "checks": {"is_4xx_code": True, "seo_friendly_url": True},
                "broken_links": True,
            }
        )
        assert page.issues == ["is_4xx_code", "broken_links"]

    async def test_polls_until_the_crawl_finishes(self):
        block = DataForSeoSiteAuditBlock()
        summaries = [
            {},
            {"crawl_progress": "in_progress"},
            {"crawl_progress": "finished"},
        ]
        with (
            patch.object(block, "_summary", AsyncMock(side_effect=summaries)),
            patch("backend.blocks.dataforseo.site_audit.sleep", AsyncMock()) as sleep,
        ):
            summary = await block._wait_for_crawl(_client(), "t1", max_wait_seconds=300)
        assert summary["crawl_progress"] == "finished"
        assert sleep.await_count == 2

    async def test_stops_waiting_at_the_limit_and_returns_partial(self):
        block = DataForSeoSiteAuditBlock()
        running = {"crawl_progress": "in_progress"}
        with (
            patch.object(block, "_summary", AsyncMock(return_value=running)),
            patch("backend.blocks.dataforseo.site_audit.sleep", AsyncMock()) as sleep,
        ):
            summary = await block._wait_for_crawl(_client(), "t1", max_wait_seconds=30)
        assert summary == running
        assert sleep.await_count == 2

    async def test_crawl_still_running_at_the_limit_is_stopped(self):
        block = DataForSeoSiteAuditBlock()
        running = {"crawl_progress": "in_progress"}
        stopped = {"crawl_progress": "finished", "crawl_stop_reason": "force_stopped"}
        stop = AsyncMock()
        with (
            patch.object(block, "_summary", AsyncMock(side_effect=[running, stopped])),
            patch.object(block, "_stop_crawl", stop),
            patch("backend.blocks.dataforseo.site_audit.sleep", AsyncMock()),
        ):
            summary = await block._crawl(_client(), "t1", max_wait_seconds=0)
        stop.assert_awaited_once()
        assert summary == stopped

    async def test_stopped_crawl_is_reported_incomplete(self):
        block = DataForSeoSiteAuditBlock()
        stopped = {
            "crawl_progress": "finished",
            "crawl_stop_reason": "force_stopped",
            "crawl_status": {"pages_crawled": 8},
        }
        with (
            patch.object(block, "_start_crawl", AsyncMock(return_value=("t1", 0.009))),
            patch.object(block, "_crawl", AsyncMock(return_value=stopped)),
            patch.object(block, "_pages", AsyncMock(return_value=[])),
        ):
            outputs = dict(
                [
                    output
                    async for output in block.run(
                        DataForSeoSiteAuditBlock.Input(
                            credentials=CREDS_INPUT,
                            target="example.com",
                            max_crawl_pages=60,
                        ),
                        credentials=CREDS,
                    )
                ]
            )
        assert outputs["complete"] is False
        assert outputs["pages_crawled"] == 8
        # 8 of 60 paid pages; DataForSEO refunds the rest once stopped.
        assert block.execution_stats.provider_cost == pytest.approx(0.0012)


class TestBilling:
    async def test_live_blocks_bill_the_reported_cost(self):
        block = DataForSeoBacklinksSummaryBlock()

        async def post_task(client, path, task):
            client.last_cost_usd = 0.024
            return [{"rank": 1}]

        with patch.object(DataForSeoClient, "post_task", post_task):
            async for _ in block.run(
                DataForSeoBacklinksSummaryBlock.Input(
                    credentials=CREDS_INPUT, target="example.com"
                ),
                credentials=CREDS,
            ):
                pass
        assert block.execution_stats.provider_cost == 0.024
        assert block.execution_stats.provider_cost_type == "cost_usd"

    def test_finished_crawl_bills_only_the_pages_crawled(self):
        finished = {"crawl_progress": "finished", "crawl_status": {"pages_crawled": 1}}
        assert crawl_cost(0.006, 40, finished) == pytest.approx(0.00015)

    def test_unfinished_crawl_bills_the_upfront_amount(self):
        running = {
            "crawl_progress": "in_progress",
            "crawl_status": {"pages_crawled": 1},
        }
        assert crawl_cost(0.006, 40, running) == 0.006
        assert crawl_cost(0.006, 40, {}) == 0.006

    async def test_crawl_is_billed_even_when_polling_fails(self):
        block = DataForSeoSiteAuditBlock()
        with (
            patch.object(block, "_start_crawl", AsyncMock(return_value=("t1", 0.006))),
            patch.object(
                block, "_summary", AsyncMock(side_effect=DataForSeoError("x"))
            ),
            pytest.raises(DataForSeoError),
        ):
            async for _ in block.run(
                DataForSeoSiteAuditBlock.Input(
                    credentials=CREDS_INPUT, target="example.com"
                ),
                credentials=CREDS,
            ):
                pass
        assert block.execution_stats.provider_cost == 0.006


@pytest.mark.parametrize(
    "raw, domain",
    [
        ("example.com", "example.com"),
        (" https://www.example.com/shop ", "example.com"),
        ("www.example.com", "example.com"),
        ("shop.example.com", "shop.example.com"),
        ("", ""),
    ],
)
def test_normalise_domain(raw, domain):
    assert normalise_domain(raw) == domain


async def test_site_audit_needs_a_target():
    block = DataForSeoSiteAuditBlock()
    with pytest.raises(BlockInputError, match="domain to crawl"):
        async for _ in block.run(
            DataForSeoSiteAuditBlock.Input(credentials=CREDS_INPUT, target=" "),
            credentials=CREDS,
        ):
            pass
