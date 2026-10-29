"""Unit tests for exa/helpers cost-extraction + merge helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.blocks.exa.helpers import (
    ContentSettings,
    ExtrasSettings,
    HighlightSettings,
    LivecrawlTypes,
    PageSection,
    SummarySettings,
    TextAdvanced,
    TextVerbosity,
    extract_exa_cost_usd,
    merge_exa_cost,
    parse_cost_dollars,
    process_contents_settings,
)
from backend.data.model import NodeExecutionStats


@pytest.mark.parametrize(
    "response, expected",
    [
        # Dataclass / SimpleNamespace with cost_dollars.total
        (SimpleNamespace(cost_dollars=SimpleNamespace(total=0.05)), 0.05),
        # Dict camelCase
        ({"costDollars": {"total": 0.10}}, 0.10),
        # Dict snake_case
        ({"cost_dollars": {"total": 0.07}}, 0.07),
        # code_context endpoint shape: plain numeric string
        (SimpleNamespace(cost_dollars="0.005"), 0.005),
        # Scalar float on cost_dollars directly
        (SimpleNamespace(cost_dollars=0.02), 0.02),
        # Scalar int on cost_dollars
        (SimpleNamespace(cost_dollars=3), 3.0),
        # Missing cost info — returns None
        ({}, None),
        (SimpleNamespace(other="foo"), None),
        (None, None),
        # Nested total=None
        (SimpleNamespace(cost_dollars=SimpleNamespace(total=None)), None),
        # Invalid numeric string
        (SimpleNamespace(cost_dollars="not-a-number"), None),
        # Negative values clamp to 0
        (SimpleNamespace(cost_dollars=SimpleNamespace(total=-1.0)), 0.0),
    ],
)
def test_extract_exa_cost_usd_handles_all_shapes(response, expected):
    assert extract_exa_cost_usd(response) == expected


def test_merge_exa_cost_emits_stats_when_cost_present():
    block = MagicMock()
    response = SimpleNamespace(cost_dollars=SimpleNamespace(total=0.0421))
    merge_exa_cost(block, response)

    block.merge_stats.assert_called_once()
    stats: NodeExecutionStats = block.merge_stats.call_args.args[0]
    assert stats.provider_cost == pytest.approx(0.0421)
    assert stats.provider_cost_type == "cost_usd"


def test_merge_exa_cost_noops_when_no_cost():
    """Webset CRUD endpoints don't surface cost_dollars today — the helper
    must silently skip instead of emitting a 0-cost telemetry record."""
    block = MagicMock()
    merge_exa_cost(block, SimpleNamespace(other_field="nothing"))
    block.merge_stats.assert_not_called()


def test_merge_exa_cost_noops_when_response_is_none():
    block = MagicMock()
    merge_exa_cost(block, None)
    block.merge_stats.assert_not_called()


# Tests for process_contents_settings and output_schema mapping


def test_process_contents_settings_with_output_schema():
    """Test that output_schema is correctly mapped to 'schema' in the result."""
    contents = ContentSettings(
        summary=SummarySettings(
            query="test query",
            output_schema={"type": "object"},
        )
    )
    result = process_contents_settings(contents)

    assert result == {
        "summary": {
            "query": "test query",
            "schema": {"type": "object"},
        }
    }


def test_process_contents_settings_with_empty_output_schema():
    """Test that empty dict {} for output_schema is preserved."""
    contents = ContentSettings(
        summary=SummarySettings(
            query=None,
            output_schema={},
        )
    )
    result = process_contents_settings(contents)

    assert result == {
        "summary": {
            "schema": {},
        }
    }


def test_process_contents_settings_with_none_output_schema():
    """Test that None for output_schema is handled correctly."""
    contents = ContentSettings(
        summary=SummarySettings(
            query="test query",
            output_schema=None,
        )
    )
    result = process_contents_settings(contents)

    assert result == {
        "summary": {
            "query": "test query",
        }
    }


def test_process_contents_settings_without_summary():
    """Test that contents without summary returns empty dict."""
    contents = ContentSettings(summary=None)
    result = process_contents_settings(contents)

    assert result == {}


def test_summary_settings_backward_compatibility_with_schema_alias():
    """Test that SummarySettings accepts 'schema' as an alias for 'output_schema'."""
    # Test that we can create SummarySettings with the old 'schema' key
    summary = SummarySettings(schema={"type": "object"})
    assert summary.output_schema == {"type": "object"}

    # Test that we can also create with the new 'output_schema' key
    summary2 = SummarySettings(output_schema={"type": "object"})
    assert summary2.output_schema == {"type": "object"}

    # Test that serialization uses 'output_schema'
    assert summary.model_dump() == {"query": None, "output_schema": {"type": "object"}}

    # Test that model_validate also works with the old key
    summary3 = SummarySettings.model_validate({"schema": {"type": "object"}})
    assert summary3.output_schema == {"type": "object"}


def test_process_contents_settings_omits_zero_extras():
    """Zero int counts must be omitted, not emitted as ``0`` to the Exa API."""
    contents = ContentSettings(extras=ExtrasSettings(links=0, image_links=0))
    result = process_contents_settings(contents)

    assert result["extras"] == {}


def test_process_contents_settings_includes_positive_extras():
    """Positive counts are included and mapped to the API's camelCase keys."""
    contents = ContentSettings(extras=ExtrasSettings(links=0, image_links=2))
    result = process_contents_settings(contents)

    assert result["extras"] == {"imageLinks": 2}


def test_default_content_settings_payload_is_empty():
    """Nothing deprecated (livecrawl, numSentences, ...) is sent by default."""
    assert process_contents_settings(ContentSettings()) == {}


def test_highlights_default_to_adaptive_true():
    contents = ContentSettings(highlights=HighlightSettings())
    assert process_contents_settings(contents) == {"highlights": True}


def test_legacy_num_sentences_maps_to_max_characters():
    contents = ContentSettings(
        highlights=HighlightSettings(num_sentences=2, highlights_per_url=5)
    )
    assert process_contents_settings(contents) == {
        "highlights": {"maxCharacters": 2666}
    }


def test_max_characters_wins_over_num_sentences():
    contents = ContentSettings(
        highlights=HighlightSettings(max_characters=700, num_sentences=3)
    )
    assert process_contents_settings(contents)["highlights"] == {"maxCharacters": 700}


@pytest.mark.parametrize(
    "livecrawl, max_age, expected",
    [
        (LivecrawlTypes.ALWAYS, None, {"maxAgeHours": 0}),
        (LivecrawlTypes.NEVER, None, {"maxAgeHours": -1}),
        (LivecrawlTypes.FALLBACK, None, {}),
        (LivecrawlTypes.PREFERRED, None, {"livecrawl": "preferred"}),
        (LivecrawlTypes.ALWAYS, 12, {"maxAgeHours": 12}),
        (None, 0, {"maxAgeHours": 0}),
    ],
)
def test_freshness_never_sends_livecrawl_and_max_age_together(
    livecrawl, max_age, expected
):
    contents = ContentSettings(livecrawl=livecrawl, max_age_hours=max_age)
    assert process_contents_settings(contents) == expected


def test_text_advanced_new_options():
    contents = ContentSettings(
        text=TextAdvanced(
            verbosity=TextVerbosity.STANDARD,
            include_sections=[PageSection.BODY],
        )
    )
    assert process_contents_settings(contents) == {
        "text": {"verbosity": "standard", "includeSections": ["body"]}
    }


def test_snapshot_drops_timeout_and_rejects_live_fetch():
    contents = ContentSettings(snapshot_as_of="2026-01-15", livecrawl_timeout=5000)
    assert process_contents_settings(contents) == {"snapshotAsOf": "2026-01-15"}
    with pytest.raises(ValueError, match="Snapshot"):
        process_contents_settings(
            ContentSettings(snapshot_as_of="2026-01-15", max_age_hours=0)
        )


@pytest.mark.parametrize(
    "data, expected_total",
    [
        ({"costDollars": {"total": 0.007, "search": {"neural": 0.007}}}, 0.007),
        (
            {
                "costDollars": {
                    "total": 0.003,
                    "contents": {"text": 0.001, "summary": 0.002},
                }
            },
            0.003,
        ),
        ({"costDollars": {"total": 0.01, "summary": 0.002}}, 0.01),
        ({}, None),
        ({"costDollars": "0.005"}, None),
    ],
)
def test_parse_cost_dollars_current_shapes(data, expected_total):
    cost = parse_cost_dollars(data)
    if expected_total is None:
        assert cost is None
    else:
        assert cost is not None
        assert cost.total == pytest.approx(expected_total)
