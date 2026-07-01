"""Unit tests for builder block listing/search db helpers.

Covers two areas the /builder endpoints depend on:

- Block-menu categorization (``_block_menu_type``) and the type-filtered
  ``get_blocks`` listing, including the input/output/action partition and the
  count/filter lockstep invariant.
- The block scoring/index logic behind /builder/search. The behavioural search
  tests patch the in-memory block index with a small set of fake entries so
  they're deterministic and don't depend on the real ~400-block registry; one
  test exercises the real index builder.
"""

from types import SimpleNamespace
from typing import Any

import pytest

import backend.api.features.builder.db as db
from backend.api.features.builder.model import BlockTypeFilter
from backend.blocks import load_all_blocks
from backend.blocks._base import AnyBlockSchema, BlockInfo, BlockType
from backend.util.text import split_camelcase

# ============================================================================
# Block-menu categorization — triggers belong with input blocks
# ============================================================================


def _menu_block_ids(*block_types: BlockType) -> set[str]:
    ids: set[str] = set()
    for block_cls in load_all_blocks().values():
        block: AnyBlockSchema = block_cls()
        if block.disabled or block.id in db.EXCLUDED_BLOCK_IDS:
            continue
        if block.block_type in block_types:
            ids.add(block.id)
    return ids


def _filtered_block_ids(type_filter: BlockTypeFilter) -> set[str]:
    response = db.get_blocks(type=type_filter, page_size=1_000_000)
    assert response.pagination.total_items == len(
        response.blocks
    ), "page_size too small — results were truncated"
    return {b.id for b in response.blocks}


def test_block_menu_type_classifies_triggers_as_input():
    assert db._block_menu_type(BlockType.INPUT) == "input"
    assert db._block_menu_type(BlockType.WEBHOOK) == "input"
    assert db._block_menu_type(BlockType.WEBHOOK_MANUAL) == "input"


def test_block_menu_type_classifies_output_and_action():
    assert db._block_menu_type(BlockType.OUTPUT) == "output"
    assert db._block_menu_type(BlockType.STANDARD) == "action"
    assert db._block_menu_type(BlockType.AI) == "action"
    assert db._block_menu_type(BlockType.AGENT) == "action"


def test_every_block_type_has_a_menu_category():
    # Guards against a future BlockType being classified as something other than
    # one of the three menu categories the frontend knows how to render.
    for block_type in BlockType:
        assert db._block_menu_type(block_type) in ("input", "output", "action")


def test_trigger_blocks_appear_under_input_blocks():
    trigger_ids = _menu_block_ids(BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
    if not trigger_ids:
        pytest.skip("no trigger blocks loaded in this environment")
    assert trigger_ids <= _filtered_block_ids("input")


def test_trigger_blocks_do_not_appear_under_action_blocks():
    trigger_ids = _menu_block_ids(BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
    if not trigger_ids:
        pytest.skip("no trigger blocks loaded in this environment")
    assert not trigger_ids & _filtered_block_ids("action")


def test_input_output_action_partition_all_blocks():
    all_ids = _filtered_block_ids("all")
    input_ids = _filtered_block_ids("input")
    output_ids = _filtered_block_ids("output")
    action_ids = _filtered_block_ids("action")

    assert input_ids | output_ids | action_ids == all_ids
    assert not input_ids & output_ids
    assert not input_ids & action_ids
    assert not output_ids & action_ids


def test_filtered_counts_match_block_menu_type_classification():
    # The get_blocks filter and the _get_static_counts badges both classify via
    # _block_menu_type, so the filtered set sizes must match a direct count.
    expected = {"input": 0, "output": 0, "action": 0}
    for block_cls in load_all_blocks().values():
        block: AnyBlockSchema = block_cls()
        if block.disabled or block.id in db.EXCLUDED_BLOCK_IDS:
            continue
        expected[db._block_menu_type(block.block_type)] += 1

    assert len(_filtered_block_ids("input")) == expected["input"]
    assert len(_filtered_block_ids("output")) == expected["output"]
    assert len(_filtered_block_ids("action")) == expected["action"]


# ============================================================================
# Search index helpers
# ============================================================================


def _entry(
    name: str,
    normalized_name: str,
    searchable_text: str,
    *,
    is_integration: bool = False,
    has_llm_model: bool = False,
) -> db._BlockIndexEntry:
    return db._BlockIndexEntry(
        block_info=BlockInfo.model_construct(id=name, name=name),
        normalized_name=normalized_name,
        name_sort_key=name.lower(),
        searchable_text=searchable_text,
        is_integration=is_integration,
        has_llm_model=has_llm_model,
    )


def _patch_index(mocker, entries: list[db._BlockIndexEntry]) -> None:
    mocker.patch.object(
        db,
        "_get_block_search_index",
        return_value=tuple(entries),
    )


# ============================================================================
# Scoring formula — locked so "ranking identical" can't silently regress
# ============================================================================


def test_score_primary_fields_regression():
    # name.startswith(query) -> +90, plus SequenceMatcher(name, query) contribution
    assert (
        db._score_primary_fields("youtube transcript", "get the transcript", "youtube")
        == 120.0
    )
    # unrelated name/description -> small fuzzy-only score, below threshold
    assert db._score_primary_fields("foo bar", "baz", "xyzzy") == 6.25
    # exact name match is the top of the scale
    assert db._score_primary_fields("exact", "some desc", "exact") == pytest.approx(
        177.14285714285714
    )


# ============================================================================
# Block text search — matching, filtering, thresholds
# ============================================================================


def test_text_search_matches_expected_block(mocker):
    _patch_index(
        mocker,
        [
            _entry("YoutubeTranscript", "youtube transcript", "get the transcript"),
            _entry("WeatherForecast", "weather forecast", "current weather conditions"),
            _entry("SendEmail", "send email", "send an email message"),
        ],
    )

    results, blocks, integrations = db._text_search_blocks(
        query="youtube", include_blocks=True, include_integrations=True
    )

    assert [r.item.name for r in results] == ["YoutubeTranscript"]
    assert (blocks, integrations) == (1, 0)


def test_integration_filter(mocker):
    _patch_index(
        mocker,
        [
            _entry("PlainBlock", "plain block", "does things", is_integration=False),
            _entry("ApiBlock", "api block", "calls an api", is_integration=True),
        ],
    )

    blocks_only, b_count, i_count = db._collect_block_results(
        include_blocks=True, include_integrations=False
    )
    assert [r.item.name for r in blocks_only] == ["PlainBlock"]
    assert (b_count, i_count) == (1, 0)

    integrations_only, b_count, i_count = db._collect_block_results(
        include_blocks=False, include_integrations=True
    )
    assert [r.item.name for r in integrations_only] == ["ApiBlock"]
    assert (b_count, i_count) == (0, 1)


def test_threshold_filtering(mocker):
    # A query that only weakly fuzzy-matches scores below
    # MIN_SCORE_FOR_FILTERED_RESULTS and is dropped.
    _patch_index(
        mocker,
        [
            _entry("Calculator", "calculator", "add two numbers together"),
            _entry("Note", "note", "a sticky note"),
        ],
    )
    below, _, _ = db._text_search_blocks(
        query="xyzzy", include_blocks=True, include_integrations=True
    )
    assert below == []

    # An exact-name query clears the threshold.
    _patch_index(mocker, [_entry("ExactName", "exact", "some desc")])
    above, _, _ = db._text_search_blocks(
        query="exact", include_blocks=True, include_integrations=True
    )
    assert [r.item.name for r in above] == ["ExactName"]


def test_query_results_are_subset_of_full_listing(mocker):
    entries = [
        _entry("YoutubeTranscript", "youtube transcript", "get the transcript"),
        _entry("WeatherForecast", "weather forecast", "current weather conditions"),
    ]
    _patch_index(mocker, entries)

    listed, _, _ = db._collect_block_results(
        include_blocks=True, include_integrations=True
    )
    matched, _, _ = db._text_search_blocks(
        query="youtube", include_blocks=True, include_integrations=True
    )

    assert {r.item.name for r in listed} == {"YoutubeTranscript", "WeatherForecast"}
    assert {r.item.name for r in matched} == {"YoutubeTranscript"}
    assert len(matched) < len(listed)


# ============================================================================
# LLM-model bonus — the build-time/query-time split must preserve the +20
# ============================================================================


def test_query_matches_llm_model():
    a_model_name = db.llm_models[0]
    assert db._query_matches_llm_model(a_model_name) is True
    assert db._query_matches_llm_model("definitely-not-a-model-name") is False


def test_llm_model_bonus_adds_twenty():
    # Use a real model name as both the block name and the query so the base
    # score clears the threshold and the only difference is the LLM bonus.
    model_query = db.llm_models[0]
    with_llm = _entry("M", model_query, "", has_llm_model=True)
    without_llm = _entry("M", model_query, "", has_llm_model=False)

    scored_with = db._score_block_entry(with_llm, model_query)
    scored_without = db._score_block_entry(without_llm, model_query)

    assert scored_with is not None and scored_without is not None
    assert scored_with.score - scored_without.score == 20


# ============================================================================
# Index builder — exercised against the real block registry
# ============================================================================


def test_index_excludes_disabled_and_excluded_blocks():
    db._get_block_search_index.cache_clear()
    entries = db._get_block_search_index()
    db._get_block_search_index.cache_clear()

    assert len(entries) > 0
    entry_ids = {entry.block_info.id for entry in entries}
    assert entry_ids.isdisjoint(db.EXCLUDED_BLOCK_IDS)

    sample = entries[0]
    assert sample.normalized_name
    assert sample.block_info.name
    assert sample.normalized_name == split_camelcase(sample.block_info.name).lower()


# ============================================================================
# Orchestration — concurrent branches are all merged (gather regression guard)
# ============================================================================


async def test_build_cached_search_results_merges_all_branches(mocker):
    _patch_index(
        mocker,
        [_entry("YoutubeTranscript", "youtube transcript", "get the transcript")],
    )
    library_item = db._ScoredItem(
        item=BlockInfo.model_construct(id="lib", name="LibAgent"),
        filter_type="my_agents",
        score=100.0,
        sort_key="libagent",
    )
    marketplace_item = db._ScoredItem(
        item=BlockInfo.model_construct(id="mkt", name="StoreAgent"),
        filter_type="marketplace_agents",
        score=80.0,
        sort_key="storeagent",
    )
    mock_library = mocker.patch.object(
        db, "_search_library", return_value=([library_item], {"my_agents": 3})
    )
    mock_marketplace = mocker.patch.object(
        db,
        "_search_marketplace",
        return_value=([marketplace_item], {"marketplace_agents": 9}),
    )

    # Call the undecorated function to bypass the Redis-backed @cached wrapper.
    result = await db._build_cached_search_results.__wrapped__(
        "user-1",
        "youtube",
        ("blocks", "integrations", "marketplace_agents", "my_agents"),
        (),
    )

    # Items are merged and ordered by descending score (170 > 100 > 80).
    assert [item.name for item in result.items] == [
        "YoutubeTranscript",
        "LibAgent",
        "StoreAgent",
    ]
    assert result.total_items == {
        "blocks": 1,
        "integrations": 0,
        "marketplace_agents": 9,
        "my_agents": 3,
    }
    # Each branch is awaited exactly once with the expected arguments.
    mock_library.assert_awaited_once_with("user-1", "youtube", "youtube")
    mock_marketplace.assert_awaited_once_with([], "youtube", "youtube")


def test_search_blocks_routes_query_vs_listing(mocker):
    _patch_index(
        mocker,
        [
            _entry("YoutubeTranscript", "youtube transcript", "get the transcript"),
            _entry("WeatherForecast", "weather forecast", "current weather"),
        ],
    )

    # Query present -> text search returns only matching blocks.
    items, totals = db._search_blocks("youtube", "youtube", True, True)
    assert [i.item.name for i in items] == ["YoutubeTranscript"]
    assert totals == {"blocks": 1, "integrations": 0}

    # No query -> full block listing.
    items, totals = db._search_blocks("", "", True, True)
    assert {i.item.name for i in items} == {"YoutubeTranscript", "WeatherForecast"}
    assert totals == {"blocks": 2, "integrations": 0}


def test_block_searchable_text_includes_field_descriptions():
    block: Any = SimpleNamespace(
        description="My Block",
        input_schema=SimpleNamespace(
            model_fields={
                "keyword": SimpleNamespace(description="A searchable keyword"),
                "plain": SimpleNamespace(description=""),  # no description -> skipped
            }
        ),
    )
    assert db._block_searchable_text(block) == "my block keyword: a searchable keyword"
