"""Tests for the ExaContentsBlock request body and response handling."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa.contents import ExaContentsBlock, build_contents_payload
from backend.blocks.exa.helpers import (
    ExtrasSettings,
    HighlightSettings,
    LivecrawlTypes,
    PageSection,
    SummarySettings,
    TextVerbosity,
)


def _input(**kwargs) -> ExaContentsBlock.Input:
    return ExaContentsBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        urls=kwargs.pop("urls", ["https://example.com"]),
        **kwargs,
    )


def test_default_payload_matches_previous_wire_behaviour():
    """Defaults send text with HTML tags, the 10 s timeout and context, and no
    deprecated livecrawl/highlight fields."""
    assert build_contents_payload(_input()) == {
        "urls": ["https://example.com"],
        "text": {"includeHtmlTags": True},
        "livecrawlTimeout": 10000,
        "context": True,
    }


def test_requires_urls_or_ids():
    with pytest.raises(ValueError, match="urls"):
        build_contents_payload(_input(urls=[]))


def test_ids_used_when_no_urls():
    payload = build_contents_payload(_input(urls=[], ids=["doc-1"]))
    assert payload["ids"] == ["doc-1"]
    assert "urls" not in payload


def test_summary_output_schema_maps_to_schema_key():
    payload = build_contents_payload(
        _input(summary=SummarySettings(query="overview", output_schema={"a": 1}))
    )
    assert payload["summary"] == {"query": "overview", "schema": {"a": 1}}


def test_extras_omit_zero_counts_and_use_camel_case():
    payload = build_contents_payload(
        _input(extras=ExtrasSettings(links=0, image_links=3, rich_image_links=2))
    )
    assert payload["extras"] == {"imageLinks": 3, "richImageLinks": 2}


def test_text_options_are_sent_top_level():
    payload = build_contents_payload(
        _input(
            text_max_characters=500,
            text_include_html_tags=False,
            text_verbosity=TextVerbosity.FULL,
            text_exclude_sections=[PageSection.FOOTER, PageSection.NAVIGATION],
        )
    )
    assert payload["text"] == {
        "maxCharacters": 500,
        "verbosity": "full",
        "excludeSections": ["footer", "navigation"],
    }


def test_text_disabled_sends_no_text():
    assert "text" not in build_contents_payload(_input(text=False))


@pytest.mark.parametrize(
    "highlights, expected",
    [
        (HighlightSettings(), None),
        # Legacy saved defaults count as "off", as before.
        (HighlightSettings(num_sentences=1, highlights_per_url=1), None),
        (HighlightSettings(query="pricing"), {"query": "pricing"}),
        (HighlightSettings(max_characters=800), {"maxCharacters": 800}),
        # Deprecated numSentences is translated, highlightsPerUrl dropped.
        (
            HighlightSettings(num_sentences=3, highlights_per_url=2),
            {"maxCharacters": 3999},
        ),
        (
            HighlightSettings(num_sentences=1, highlights_per_url=4),
            {"maxCharacters": 1333},
        ),
    ],
)
def test_highlights(highlights, expected):
    payload = build_contents_payload(_input(highlights=highlights))
    assert payload.get("highlights") == expected


def test_max_age_hours_replaces_livecrawl():
    payload = build_contents_payload(
        _input(max_age_hours=24, livecrawl=LivecrawlTypes.ALWAYS)
    )
    assert payload["maxAgeHours"] == 24
    assert "livecrawl" not in payload


@pytest.mark.parametrize(
    "livecrawl, expected",
    [
        (LivecrawlTypes.ALWAYS, {"maxAgeHours": 0}),
        (LivecrawlTypes.NEVER, {"maxAgeHours": -1}),
        (LivecrawlTypes.FALLBACK, {}),
        (LivecrawlTypes.PREFERRED, {"livecrawl": "preferred"}),
    ],
)
def test_legacy_livecrawl_translation(livecrawl, expected):
    payload = build_contents_payload(_input(livecrawl=livecrawl))
    freshness = {k: v for k, v in payload.items() if k in ("maxAgeHours", "livecrawl")}
    assert freshness == expected


def test_snapshot_drops_live_fetch_timeout():
    payload = build_contents_payload(_input(snapshot_as_of="2026-01-15"))
    assert payload["snapshotAsOf"] == "2026-01-15"
    assert "livecrawlTimeout" not in payload


def test_snapshot_with_max_age_is_rejected():
    with pytest.raises(ValueError, match="Snapshot"):
        build_contents_payload(_input(snapshot_as_of="2026-01-15", max_age_hours=1))


def test_invalid_user_identifier_is_rejected():
    with pytest.raises(ValueError, match="User Identifier"):
        build_contents_payload(_input(user_identifier="me@example.com"))


@pytest.mark.asyncio
async def test_run_parses_statuses_with_new_fields():
    block = ExaContentsBlock()
    response = {
        "requestId": "req-9",
        "results": [{"id": "https://a.com", "url": "https://a.com", "text": "A"}],
        "statuses": [
            {"id": "https://a.com", "status": "success", "source": "cached"},
            {
                "id": "https://b.com",
                "status": "error",
                "error": {"tag": "SOME_NEW_TAG", "httpStatusCode": 451},
            },
        ],
    }
    with patch.object(block, "call_exa", AsyncMock(return_value=response)):
        outputs = [
            item async for item in block.run(_input(), credentials=TEST_CREDENTIALS)
        ]
    statuses = dict(outputs)["statuses"]
    assert statuses[0].source == "cached"
    assert statuses[1].error.tag == "SOME_NEW_TAG"
    assert dict(outputs)["request_id"] == "req-9"
