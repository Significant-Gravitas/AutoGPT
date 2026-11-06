"""Tests for the DataForSEO Site Audit block."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.util.exceptions import BlockInputError

from ._api import DataForSeoClient, DataForSeoError
from ._config import dataforseo
from ._onpage import count_issues, parse_page
from .site_audit import (
    MAX_WAIT_SECONDS,
    PAGES_TIMEOUT_SECONDS,
    STOP_WAIT_SECONDS,
    CrawlState,
    DataForSeoSiteAuditBlock,
    crawl_cost,
)

CREDS = dataforseo.get_test_credentials()
CREDS_INPUT = CREDS.model_dump()
SLEEP = "backend.blocks.dataforseo.site_audit.sleep"


def _client() -> DataForSeoClient:
    return DataForSeoClient(CREDS)


def _input(**kwargs) -> DataForSeoSiteAuditBlock.Input:
    return DataForSeoSiteAuditBlock.Input(
        credentials=CREDS_INPUT, target=kwargs.pop("target", "example.com"), **kwargs
    )


def _crawl_ending_with(summary: dict, stop_requested: bool = False):
    async def crawl(client, task_id, max_wait_seconds, state: CrawlState) -> None:
        state.summary = summary
        state.stop_requested = stop_requested

    return crawl


async def _run(block: DataForSeoSiteAuditBlock, **kwargs) -> dict:
    return dict([o async for o in block.run(_input(**kwargs), credentials=CREDS)])


class TestIssues:
    def test_counts_only_problem_checks_most_common_first(self):
        metrics = {
            "checks": {"no_h1_tag": 3, "no_image_alt": 5, "is_https": 9, "no_title": 0},
            # Summary totals count links, not pages, so they are ignored.
            "broken_links": 40,
        }
        pages = [{"broken_links": True}, {"broken_links": True}, {}]
        assert list(count_issues(metrics, pages).items()) == [
            ("no_image_alt", 5),
            ("no_h1_tag", 3),
            ("broken_links", 2),
        ]

    def test_page_issues_skip_healthy_checks(self):
        page = parse_page(
            {
                "url": "https://example.com/",
                "checks": {"is_4xx_code": True, "seo_friendly_url": True},
                "broken_links": True,
            }
        )
        assert page.issues == ["is_4xx_code", "broken_links"]


class TestPolling:
    async def test_polls_until_the_crawl_finishes(self):
        block = DataForSeoSiteAuditBlock()
        summaries = [
            {},
            {"crawl_progress": "in_progress"},
            {"crawl_progress": "finished"},
        ]
        state = CrawlState()
        with (
            patch.object(block, "_summary", AsyncMock(side_effect=summaries)),
            patch(SLEEP, AsyncMock()) as sleep,
        ):
            await block._wait_for_crawl(_client(), "t1", 300, state)
        assert state.summary["crawl_progress"] == "finished"
        assert sleep.await_count == 2

    async def test_stops_waiting_at_the_limit(self):
        block = DataForSeoSiteAuditBlock()
        running = {"crawl_progress": "in_progress"}
        state = CrawlState()
        with (
            patch.object(block, "_summary", AsyncMock(return_value=running)),
            patch(SLEEP, AsyncMock()) as sleep,
        ):
            await block._wait_for_crawl(_client(), "t1", 30, state)
        assert state.summary == running
        assert sleep.await_count == 2

    async def test_crawl_still_running_at_the_limit_is_stopped(self):
        block = DataForSeoSiteAuditBlock()
        running = {"crawl_progress": "in_progress"}
        stopped = {"crawl_progress": "finished", "crawl_stop_reason": "force_stopped"}
        stop = AsyncMock()
        state = CrawlState()
        with (
            patch.object(block, "_summary", AsyncMock(side_effect=[running, stopped])),
            patch.object(block, "_stop_crawl", stop),
            patch(SLEEP, AsyncMock()),
        ):
            await block._crawl(_client(), "t1", 0, state)
        stop.assert_awaited_once()
        assert state.summary == stopped
        assert state.stop_requested

    def test_longest_run_leaves_room_inside_ottos_limit(self):
        # Otto cancels a direct block run after 300 seconds.
        assert MAX_WAIT_SECONDS + STOP_WAIT_SECONDS + PAGES_TIMEOUT_SECONDS <= 270


class TestOutputs:
    @pytest.mark.parametrize("reason", ["force_stopped", "unexpected_exception"])
    async def test_crawl_ended_early_is_reported_incomplete(self, reason):
        block = DataForSeoSiteAuditBlock()
        ended = {"crawl_progress": "finished", "crawl_stop_reason": reason}
        with (
            patch.object(block, "_start_crawl", AsyncMock(return_value=("t1", 0.0))),
            patch.object(block, "_crawl", _crawl_ending_with(ended)),
            patch.object(block, "_pages", AsyncMock(return_value=[])),
        ):
            outputs = await _run(block)
        assert outputs["complete"] is False

    async def test_slow_page_list_still_returns_the_summary(self):
        block = DataForSeoSiteAuditBlock()
        done = {
            "crawl_progress": "finished",
            "crawl_status": {"pages_crawled": 4},
            "page_metrics": {"onpage_score": 90.0, "checks": {"no_h1_tag": 1}},
        }

        async def never(*args, **kwargs):
            await asyncio.Event().wait()

        with (
            patch.object(block, "_start_crawl", AsyncMock(return_value=("t1", 0.0))),
            patch.object(block, "_crawl", _crawl_ending_with(done)),
            patch.object(block, "_pages", never),
            patch("backend.blocks.dataforseo.site_audit.PAGES_TIMEOUT_SECONDS", 0.01),
        ):
            outputs = await _run(block)
        assert outputs["pages"] == []
        assert outputs["pages_crawled"] == 4
        assert outputs["issue_counts"] == {"no_h1_tag": 1}

    async def test_needs_a_target(self):
        block = DataForSeoSiteAuditBlock()
        with pytest.raises(BlockInputError, match="domain to crawl"):
            await _run(block, target=" ")


class TestCrawlBilling:
    def test_finished_crawl_bills_only_the_pages_crawled(self):
        state = CrawlState(
            summary={"crawl_progress": "finished", "crawl_status": {"pages_crawled": 1}}
        )
        assert crawl_cost(0.006, 40, state) == pytest.approx(0.00015)

    def test_stop_not_yet_landed_bills_the_pages_crawled_so_far(self):
        state = CrawlState(
            summary={
                "crawl_progress": "in_progress",
                "crawl_status": {"pages_crawled": 8},
            },
            stop_requested=True,
        )
        assert crawl_cost(0.009, 60, state) == pytest.approx(0.0012)

    def test_unstopped_unfinished_crawl_bills_the_upfront_amount(self):
        running = {
            "crawl_progress": "in_progress",
            "crawl_status": {"pages_crawled": 1},
        }
        assert crawl_cost(0.006, 40, CrawlState(summary=running)) == 0.006
        assert crawl_cost(0.006, 40, CrawlState()) == 0.006

    def test_never_bills_more_pages_than_were_paid_for(self):
        over = {"crawl_progress": "finished", "crawl_status": {"pages_crawled": 50}}
        assert crawl_cost(0.006, 40, CrawlState(summary=over)) == 0.006

    async def test_stopped_crawl_bills_pages_crawled(self):
        block = DataForSeoSiteAuditBlock()
        stopped = {
            "crawl_progress": "finished",
            "crawl_stop_reason": "force_stopped",
            "crawl_status": {"pages_crawled": 8},
        }
        with (
            patch.object(block, "_start_crawl", AsyncMock(return_value=("t1", 0.009))),
            patch.object(block, "_crawl", _crawl_ending_with(stopped, True)),
            patch.object(block, "_pages", AsyncMock(return_value=[])),
        ):
            outputs = await _run(block, max_crawl_pages=60)
        assert outputs["pages_crawled"] == 8
        # 8 of 60 paid pages; DataForSEO refunds the rest once stopped.
        assert block.execution_stats.provider_cost == pytest.approx(0.0012)

    async def test_crawl_is_billed_even_when_polling_fails(self):
        block = DataForSeoSiteAuditBlock()
        with (
            patch.object(block, "_start_crawl", AsyncMock(return_value=("t1", 0.006))),
            patch.object(
                block, "_summary", AsyncMock(side_effect=DataForSeoError("x"))
            ),
            pytest.raises(DataForSeoError),
        ):
            await _run(block)
        assert block.execution_stats.provider_cost == 0.006
