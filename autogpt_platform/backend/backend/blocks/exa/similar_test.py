"""Tests for the ExaFindSimilarBlock request body."""

from datetime import UTC, datetime

from backend.blocks.exa._test import TEST_CREDENTIALS_INPUT
from backend.blocks.exa.helpers import ContentSettings, HighlightSettings
from backend.blocks.exa.search_options import ExaSearchCategories
from backend.blocks.exa.similar import ExaFindSimilarBlock, build_find_similar_payload


def _input(**kwargs) -> ExaFindSimilarBlock.Input:
    return ExaFindSimilarBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT, url="https://example.com", **kwargs
    )


def test_default_payload_keeps_legacy_text_default():
    assert build_find_similar_payload(_input()) == {
        "url": "https://example.com",
        "numResults": 10,
        "contents": {"text": {"maxCharacters": 10000}},
    }


def test_new_options_and_ignored_crawl_dates():
    payload = build_find_similar_payload(
        _input(
            exclude_source_domain=True,
            category=ExaSearchCategories.RESEARCH_PAPER,
            start_crawl_date=datetime(2024, 1, 1, tzinfo=UTC),
            contents=ContentSettings(highlights=HighlightSettings(query="pricing")),
        )
    )
    assert payload["excludeSourceDomain"] is True
    assert payload["category"] == "publication"
    assert "startCrawlDate" not in payload
    assert payload["contents"] == {"highlights": {"query": "pricing"}}
