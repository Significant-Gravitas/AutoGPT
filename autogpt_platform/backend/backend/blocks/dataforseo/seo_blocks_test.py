"""Tests for the DataForSEO SERP, ranked keywords, backlinks, AI and audit blocks."""

from unittest.mock import patch

import pytest

from ._api import DataForSeoClient, DataForSeoError
from ._config import dataforseo
from .ai_mentions import DataForSeoAiMentionsBlock
from .ai_mentions import build_task as build_mentions_task
from .backlinks import DataForSeoBacklinksSummaryBlock
from .ranked_keywords import POSITION_FIELD, DataForSeoRankedKeywordsBlock
from .ranked_keywords import build_task as build_ranked_task
from .serp import parse_organic

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
