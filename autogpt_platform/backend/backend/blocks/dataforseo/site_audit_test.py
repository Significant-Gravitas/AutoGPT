"""Tests for the DataForSEO Site Audit block."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.util.exceptions import BlockInputError

from ._api import DataForSeoClient, DataForSeoError
from ._config import dataforseo
from ._onpage import count_issues, normalise_domain, parse_page
from .site_audit import (
    MAX_WAIT_SECONDS,
    STOP_WAIT_SECONDS,
    DataForSeoSiteAuditBlock,
    crawl_cost,
)

CREDS = dataforseo.get_test_credentials()
CREDS_INPUT = CREDS.model_dump()


def _client() -> DataForSeoClient:
    return DataForSeoClient(CREDS)


class TestSiteAudit:
    def test_counts_only_problem_checks(self):
        metrics = {
            "checks": {"no_h1_tag": 2, "is_https": 9, "no_title": 0},
            # Summary totals count links, not pages, so they are ignored.
            "broken_links": 40,
        }
        pages = [{"broken_links": True}, {"broken_links": True}, {}]
        assert count_issues(metrics, pages) == {"broken_links": 2, "no_h1_tag": 2}

    async def test_crawl_that_failed_is_reported_incomplete(self):
        block = DataForSeoSiteAuditBlock()
        failed = {
            "crawl_progress": "finished",
            "crawl_stop_reason": "unexpected_exception",
        }
        with (
            patch.object(block, "_start_crawl", AsyncMock(return_value=("t1", 0.0))),
            patch.object(block, "_crawl", AsyncMock(return_value=failed)),
            patch.object(block, "_pages", AsyncMock(return_value=[])),
        ):
            outputs = dict(
                [
                    output
                    async for output in block.run(
                        DataForSeoSiteAuditBlock.Input(
                            credentials=CREDS_INPUT, target="example.com"
                        ),
                        credentials=CREDS,
                    )
                ]
            )
        assert outputs["complete"] is False

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


class TestCrawlBilling:
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


def test_longest_wait_leaves_room_inside_ottos_limit():
    # Otto cancels a direct block run after 300 seconds.
    assert MAX_WAIT_SECONDS + STOP_WAIT_SECONDS <= 240


def test_crawl_cost_never_bills_more_pages_than_were_paid_for():
    over = {"crawl_progress": "finished", "crawl_status": {"pages_crawled": 50}}
    assert crawl_cost(0.006, 40, over) == 0.006
