"""Tests for the ExaSearchBlock request body, per-type rules and response parsing."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa.helpers import (
    ContentSettings,
    HighlightSettings,
    SummarySettings,
    TextAdvanced,
)
from backend.blocks.exa.search import ExaSearchBlock, build_search_payload
from backend.blocks.exa.search_options import ExaSearchCategories, ExaSearchTypes


def _input(**kwargs) -> ExaSearchBlock.Input:
    return ExaSearchBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT, query=kwargs.pop("query", "q"), **kwargs
    )


def test_default_payload_keeps_legacy_text_default():
    assert build_search_payload(_input()) == {
        "query": "q",
        "type": "auto",
        "numResults": 10,
        "contents": {"text": {"maxCharacters": 10000}},
    }


def test_crawl_dates_are_not_sent():
    payload = build_search_payload(
        _input(
            start_crawl_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_crawl_date=datetime(2024, 2, 1, tzinfo=UTC),
            start_published_date=datetime(2024, 3, 1, tzinfo=UTC),
        )
    )
    assert "startCrawlDate" not in payload and "endCrawlDate" not in payload
    assert payload["startPublishedDate"] == "2024-03-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "category, expected",
    [
        (ExaSearchCategories.RESEARCH_PAPER, "publication"),
        (ExaSearchCategories.LINKEDIN_PROFILE, "people"),
        (ExaSearchCategories.PEOPLE, "people"),
        (ExaSearchCategories.TWEET, "tweet"),
    ],
)
def test_legacy_categories_are_translated(category, expected):
    assert build_search_payload(_input(category=category))["category"] == expected


def test_new_top_level_options_are_sent():
    payload = build_search_payload(
        _input(
            type=ExaSearchTypes.DEEP,
            additional_queries=["variant one"],
            output_schema={"type": "text", "description": "Three bullets"},
            system_prompt="Prefer primary sources",
            objective="Find funding figures",
            user_identifier="user_123-abc",
            user_location="us",
            moderation=True,
        )
    )
    assert payload["additionalQueries"] == ["variant one"]
    assert payload["outputSchema"] == {"type": "text", "description": "Three bullets"}
    assert payload["systemPrompt"] == "Prefer primary sources"
    assert payload["objective"] == "Find funding figures"
    assert payload["userIdentifier"] == "user_123-abc"
    assert payload["userLocation"] == "US"
    assert payload["moderation"] is True


def test_content_settings_are_all_forwarded_under_contents():
    payload = build_search_payload(
        _input(
            contents=ContentSettings(
                text=TextAdvanced(max_characters=2000),
                highlights=HighlightSettings(),
                summary=SummarySettings(query="gist"),
                max_age_hours=24,
                subpages=2,
            )
        )
    )
    assert payload["contents"] == {
        "text": {"maxCharacters": 2000},
        "highlights": True,
        "summary": {"query": "gist"},
        "maxAgeHours": 24,
        "subpages": 2,
    }


def test_additional_queries_require_deep_type():
    with pytest.raises(ValueError, match="deep"):
        build_search_payload(_input(additional_queries=["x"]))


def test_lite_accepts_its_parameter_set():
    payload = build_search_payload(
        _input(
            type=ExaSearchTypes.LITE,
            include_domains=["example.com"],
            contents=ContentSettings(highlights=HighlightSettings(max_characters=500)),
            objective="goal",
        )
    )
    assert payload["type"] == "lite"


def test_lite_default_contents_are_allowed():
    assert build_search_payload(_input(type=ExaSearchTypes.LITE))["contents"]


def test_lite_rejects_unsupported_options_naming_them():
    with pytest.raises(ValueError) as error:
        build_search_payload(
            _input(
                type=ExaSearchTypes.LITE,
                system_prompt="x",
                include_text=["word"],
                contents=ContentSettings(
                    summary=SummarySettings(query="s"), max_age_hours=1
                ),
            )
        )
    message = str(error.value)
    for label in (
        "System Prompt",
        "Include Text",
        "Contents > Summary",
        "Contents > Max Age Hours",
    ):
        assert label in message
    assert "'auto'" in message


def test_lite_ignores_crawl_dates_since_they_are_not_sent():
    payload = build_search_payload(
        _input(
            type=ExaSearchTypes.LITE,
            start_crawl_date=datetime(2024, 1, 1, tzinfo=UTC),
        )
    )
    assert payload["type"] == "lite"


@pytest.mark.parametrize("location", ["USA", "U1", "United States"])
def test_invalid_country_code_is_rejected(location):
    with pytest.raises(ValueError, match="country code"):
        build_search_payload(_input(user_location=location))


def test_invalid_user_identifier_is_rejected():
    with pytest.raises(ValueError, match="User Identifier"):
        build_search_payload(_input(user_identifier="jane doe"))


@pytest.mark.asyncio
async def test_run_yields_synthesis_warnings_notices_and_entities():
    block = ExaSearchBlock()
    response = {
        "requestId": "req-5",
        "results": [
            {
                "id": "https://acme.example",
                "url": "https://acme.example",
                "title": "Acme",
                "entities": [
                    {
                        "id": "co-1",
                        "type": "company",
                        "version": 1,
                        "properties": {"name": "Acme"},
                    }
                ],
                "extras": {
                    "richImageLinks": [{"url": "https://acme.example/logo.png"}]
                },
            }
        ],
        "output": {
            "content": {"companies": [{"name": "Acme"}]},
            "grounding": [
                {
                    "field": "companies[0].name",
                    "citations": [{"url": "https://acme.example", "title": "Acme"}],
                    "confidence": "high",
                }
            ],
        },
        "warnings": [
            {
                "code": "IGNORED_FIELD",
                "path": "/startCrawlDate",
                "message": "Remove it",
            }
        ],
        "notices": [
            {"code": "RESTRICTIONS_RELAXED", "path": "/query", "message": "Relaxed"}
        ],
        "resolvedSearchType": "",
        "costDollars": {"total": 0.012, "search": {"neural": 0.012}},
    }
    with patch.object(block, "call_exa", AsyncMock(return_value=response)) as call:
        outputs = dict(
            [item async for item in block.run(_input(), credentials=TEST_CREDENTIALS)]
        )
    assert call.call_args.kwargs["body"]["query"] == "q"
    result = outputs["result"]
    assert result.entities[0].properties == {"name": "Acme"}
    assert result.extras.richImageLinks[0].url == "https://acme.example/logo.png"
    assert outputs["output_content"] == {"companies": [{"name": "Acme"}]}
    assert outputs["output_grounding"][0].confidence == "high"
    assert outputs["warnings"][0].code == "IGNORED_FIELD"
    assert outputs["notices"][0].code == "RESTRICTIONS_RELAXED"
    assert "resolved_search_type" not in outputs
    assert outputs["cost_dollars"].search.neural == pytest.approx(0.012)
    assert outputs["request_id"] == "req-5"


@pytest.mark.asyncio
async def test_run_without_output_skips_synthesis_outputs():
    block = ExaSearchBlock()
    with patch.object(
        block, "call_exa", AsyncMock(return_value={"results": [{"url": "https://a"}]})
    ):
        outputs = dict(
            [item async for item in block.run(_input(), credentials=TEST_CREDENTIALS)]
        )
    assert "output_content" not in outputs
    assert outputs["result"].id == "https://a"
