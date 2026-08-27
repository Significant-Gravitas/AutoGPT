"""Tests for the backfill migration script — Cypher boundary mocked.

The migration selects NULL-metadata edges, recovers ``source_kind`` /
``scope`` from the surviving ``MemoryEnvelope`` JSON on the source
episodes, and applies one coalesce-guarded SET per bucket. These tests
pin the recovery rules (the #13389 review finding: a blanket
``user_asserted`` default would grant dream-derived facts user-level
trust) and the driver-flow branches: success, no-graph, invalid-group.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.memory_model import MemoryEnvelope, SourceKind

from . import backfill_edge_props as mig


@pytest.fixture(autouse=True)
def _stub_driver(mocker):
    """Replace AutoGPTFalkorDriver with a MagicMock that returns canned
    execute_query results."""
    driver = mocker.MagicMock()
    driver.close = AsyncMock(return_value=None)
    mocker.patch.object(
        mig, "AutoGPTFalkorDriver", mocker.MagicMock(return_value=driver)
    )
    return driver


def _envelope(source_kind: SourceKind, scope: str = "real:global") -> MemoryEnvelope:
    return MemoryEnvelope(content="fact", source_kind=source_kind, scope=scope)


class TestRecoverEdgeMetadata:
    """Pure recovery rules — the heart of the mislabeling fix."""

    def test_dream_envelope_keeps_assistant_derived_and_scope(self):
        """A pre-fix dream edge must NOT be branded user_asserted."""
        envelopes = {
            "ep1": _envelope(SourceKind.assistant_derived, "project:atlas"),
        }
        assert mig._recover_edge_metadata(["ep1"], envelopes) == (
            "assistant_derived",
            "project:atlas",
        )

    def test_user_asserted_envelope_recovers_custom_scope(self):
        envelopes = {"ep1": _envelope(SourceKind.user_asserted, "book:dune")}
        assert mig._recover_edge_metadata(["ep1"], envelopes) == (
            "user_asserted",
            "book:dune",
        )

    def test_merged_user_and_dream_episodes_favor_user_asserted(self):
        """Dedup-merged edge: the user actually asserted the fact, so the
        most-trusted source wins."""
        envelopes = {
            "ep-user": _envelope(SourceKind.user_asserted),
            "ep-dream": _envelope(SourceKind.assistant_derived),
        }
        source_kind, _ = mig._recover_edge_metadata(["ep-user", "ep-dream"], envelopes)
        assert source_kind == "user_asserted"

    def test_tool_observed_outranks_assistant_derived(self):
        envelopes = {
            "ep-tool": _envelope(SourceKind.tool_observed),
            "ep-dream": _envelope(SourceKind.assistant_derived),
        }
        source_kind, _ = mig._recover_edge_metadata(["ep-tool", "ep-dream"], envelopes)
        assert source_kind == "tool_observed"

    def test_non_envelope_episode_votes_user_asserted_global(self):
        """A conversation-turn episode (no envelope) merged with a dream
        envelope makes the edge user_asserted / real:global."""
        envelopes = {
            "ep-dream": _envelope(SourceKind.assistant_derived, "project:atlas"),
        }
        assert mig._recover_edge_metadata(["ep-conv", "ep-dream"], envelopes) == (
            "user_asserted",
            "real:global",
        )

    def test_mixed_scopes_fall_back_to_global(self):
        envelopes = {
            "ep1": _envelope(SourceKind.assistant_derived, "project:atlas"),
            "ep2": _envelope(SourceKind.assistant_derived, "project:zeus"),
        }
        assert mig._recover_edge_metadata(["ep1", "ep2"], envelopes) == (
            "assistant_derived",
            "real:global",
        )

    def test_no_episodes_defaults(self):
        assert mig._recover_edge_metadata([], {}) == (
            "user_asserted",
            "real:global",
        )


class TestParseEnvelope:
    def test_valid_envelope_json_parses(self):
        body = _envelope(SourceKind.assistant_derived, "project:x").model_dump_json()
        parsed = mig._parse_envelope(body)
        assert parsed is not None
        assert parsed.source_kind is SourceKind.assistant_derived

    @pytest.mark.parametrize(
        "content",
        [
            "User: hello\nAssistant: hi",  # conversation turn, not JSON
            '["not", "a", "dict"]',
            '{"no_content_key": true}',
            '{"content": "x", "source_kind": "not-a-kind"}',
            None,
            12,
        ],
    )
    def test_non_envelope_content_returns_none(self, content):
        assert mig._parse_envelope(content) is None


def test_apply_query_sets_all_defaultable_fields_idempotently():
    """#13389: the SET must cover the three Cypher-filterable fields
    (status/source_kind/scope), idempotently (coalesce so re-runs and
    real values are never clobbered), with temporal-aware status so an
    already-expired edge isn't mislabeled 'active'. ``source_kind`` and
    ``scope`` are parameterized (recovered per bucket), never literal
    defaults baked into the query."""
    q = mig.APPLY_DEFAULTS_QUERY
    assert "e.source_kind = coalesce(e.source_kind, $source_kind)" in q
    assert "e.scope = coalesce(e.scope, $scope)" in q
    assert "'user_asserted'" not in q and "'real:global'" not in q
    assert "coalesce(" in q and "e.status" in q
    # status is gated on expired_at ONLY (the reliable already-retired
    # signal); invalid_at is excluded because it can be future-dated.
    assert "WHEN e.expired_at IS NOT NULL THEN 'superseded'" in q
    assert "e.invalid_at" not in q
    # confidence/provenance legitimately stay NULL — not forced.
    assert "e.confidence =" not in q
    assert "e.provenance =" not in q


@pytest.mark.asyncio
async def test_backfill_one_user_counts_updated(_stub_driver):
    """Happy path — edges with no episodes bucket to the defaults and the
    apply query's count comes back."""
    _stub_driver.execute_query = AsyncMock(
        side_effect=[
            (
                [{"uuid": "e1", "episodes": None}, {"uuid": "e2", "episodes": []}],
                None,
                None,
            ),
            ([{"updated": 2}], None, None),
        ]
    )
    updated = await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264")
    assert updated == 2
    apply_call = _stub_driver.execute_query.await_args_list[1]
    assert apply_call.args[0] == mig.APPLY_DEFAULTS_QUERY
    assert sorted(apply_call.kwargs["uuids"]) == ["e1", "e2"]
    assert apply_call.kwargs["source_kind"] == "user_asserted"
    assert apply_call.kwargs["scope"] == "real:global"
    _stub_driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_one_user_recovers_envelope_bucket(_stub_driver):
    """An edge sourced from a dream envelope is stamped with the
    envelope's own source_kind/scope, not the defaults."""
    body = _envelope(SourceKind.assistant_derived, "project:atlas").model_dump_json()
    _stub_driver.execute_query = AsyncMock(
        side_effect=[
            ([{"uuid": "e1", "episodes": ["ep1"]}], None, None),
            ([{"uuid": "ep1", "content": body}], None, None),
            ([{"updated": 1}], None, None),
        ]
    )
    updated = await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264")
    assert updated == 1
    episode_call = _stub_driver.execute_query.await_args_list[1]
    assert episode_call.args[0] == mig.EPISODE_CONTENT_QUERY
    apply_call = _stub_driver.execute_query.await_args_list[2]
    assert apply_call.kwargs["source_kind"] == "assistant_derived"
    assert apply_call.kwargs["scope"] == "project:atlas"


@pytest.mark.asyncio
async def test_backfill_one_user_no_records_returns_zero(_stub_driver):
    """Empty result set (no edges needed backfilling)."""
    _stub_driver.execute_query = AsyncMock(return_value=([], None, None))
    assert await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264") == 0
    _stub_driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_one_user_none_result_returns_zero(_stub_driver):
    """Driver returns None when the database doesn't exist yet.

    Pyright fix path — the migration handles this gracefully so it
    doesn't crash on freshly-signed-up users with no Graphiti graph.
    """
    _stub_driver.execute_query = AsyncMock(return_value=None)
    assert await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264") == 0


@pytest.mark.asyncio
async def test_backfill_one_user_swallows_driver_exception(_stub_driver, caplog):
    """Missing-graph errors are logged at debug and treated as no-op."""
    _stub_driver.execute_query = AsyncMock(side_effect=Exception("no such db"))
    assert await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264") == 0


@pytest.mark.asyncio
async def test_backfill_one_user_invalid_user_id_short_circuits(_stub_driver):
    """``derive_group_id`` raises ValueError on garbage input — migration
    must log and return 0 rather than attempting a query."""
    assert await mig.backfill_one_user("") == 0
    _stub_driver.execute_query.assert_not_called()
