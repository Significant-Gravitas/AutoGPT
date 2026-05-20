"""Tests for the per-user community rebuild helper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .communities import (
    _activity_since_last_rebuild,
    _summarize_communities,
    rebuild_communities_for_user,
)


class TestSummarizeCommunities:
    def test_none_passthrough(self) -> None:
        assert _summarize_communities(None) is None

    def test_list_returns_count(self) -> None:
        assert _summarize_communities([1, 2, 3]) == {"count": 3}

    def test_dict_passthrough(self) -> None:
        assert _summarize_communities({"a": 1}) == {"a": 1}

    def test_unknown_shape_coerces_to_str(self) -> None:
        result = _summarize_communities(object())
        assert isinstance(result, dict)
        assert "raw" in result


class TestRebuildCommunitiesForUser:
    @pytest.mark.asyncio
    async def test_invalid_user_id_returns_error_dict(self) -> None:
        result = await rebuild_communities_for_user("")
        assert result["error"] is not None
        assert result["communities_built"] is None

    @pytest.mark.asyncio
    async def test_success_path_calls_detach_delete_then_build(self) -> None:
        # Mock the Graphiti client + its driver
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)

        client = MagicMock()
        client.graph_driver = driver
        client.build_communities = AsyncMock(return_value=[{"name": "c1"}])

        with patch(
            "backend.copilot.graphiti.communities.get_graphiti_client",
            new=AsyncMock(return_value=client),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["error"] is None
        assert result["communities_built"] == {"count": 1}
        assert result["elapsed_seconds"] is not None

        # Defensive DETACH DELETE must run before build_communities
        cleanup_query = driver.execute_query.call_args.args[0]
        assert "MATCH (c:Community" in cleanup_query
        assert "DETACH DELETE c" in cleanup_query
        client.build_communities.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failure_path_returns_error_in_result(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)

        client = MagicMock()
        client.graph_driver = driver
        # Stub activity gate as "rebuild needed" so we reach build_communities
        client.build_communities = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch(
                "backend.copilot.graphiti.communities.get_graphiti_client",
                new=AsyncMock(return_value=client),
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(return_value=(True, "first_rebuild", {})),
            ),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["error"] is not None
        assert "RuntimeError" in result["error"]
        # Always returns elapsed_seconds even on failure
        assert result["elapsed_seconds"] is not None


def _ep_query_result(latest: str | None, total: int) -> tuple:
    """Build an execute_query return tuple shaped like an episode/community count."""
    rows = [{"latest": latest, "total": total}] if latest is not None or total else []
    if not rows:
        rows = [{"latest": None, "total": 0}]
    return (rows, None, None)


class TestActivitySinceLastRebuild:
    """Gate that decides whether ``rebuild_communities_for_user`` actually
    pays the LLM-summarization cost or short-circuits as a no-op."""

    @pytest.mark.asyncio
    async def test_skips_when_no_episodes(self) -> None:
        driver = AsyncMock()
        # First query: episodes — empty. Gate should return early.
        driver.execute_query.side_effect = [
            _ep_query_result(None, 0),  # episodes
            _ep_query_result(None, 0),  # communities
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is False
        assert reason == "no_episodes"
        assert stats["episodes_total"] == 0

    @pytest.mark.asyncio
    async def test_rebuilds_on_first_run(self) -> None:
        """Episodes exist but no communities yet — first rebuild always runs."""
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _ep_query_result("2026-05-20T01:00:00Z", 10),  # episodes
            _ep_query_result(None, 0),  # communities
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is True
        assert reason == "first_rebuild"
        assert stats["new_episodes_since_last_rebuild"] == 10

    @pytest.mark.asyncio
    async def test_skips_when_no_new_episodes_since_last_rebuild(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _ep_query_result("2026-05-20T01:00:00Z", 10),  # episodes
            _ep_query_result("2026-05-20T02:00:00Z", 3),  # communities — newer
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is False
        assert reason == "no_new_episodes_since_last_rebuild"
        assert stats["new_episodes_since_last_rebuild"] == 0

    @pytest.mark.asyncio
    async def test_skips_when_new_episodes_below_threshold(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _ep_query_result("2026-05-20T03:00:00Z", 13),  # episodes
            _ep_query_result("2026-05-20T02:00:00Z", 3),  # communities older
            ([{"c": 3}], None, None),  # count of new episodes — below threshold 5
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is False
        assert reason == "below_activity_threshold"
        assert stats["new_episodes_since_last_rebuild"] == 3
        assert stats["min_new_episodes_threshold"] == 5

    @pytest.mark.asyncio
    async def test_rebuilds_when_above_threshold(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _ep_query_result("2026-05-20T03:00:00Z", 20),  # episodes
            _ep_query_result("2026-05-20T02:00:00Z", 3),  # communities older
            ([{"c": 8}], None, None),  # 8 new episodes >= threshold 5
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is True
        assert reason == "activity_above_threshold"
        assert stats["new_episodes_since_last_rebuild"] == 8


class TestRebuildForUserActivityGate:
    """End-to-end behavior of ``rebuild_communities_for_user`` with the gate."""

    @pytest.mark.asyncio
    async def test_skip_path_records_reason_and_does_not_call_build(self) -> None:
        driver = AsyncMock()
        client = MagicMock()
        client.graph_driver = driver
        client.build_communities = AsyncMock()

        with (
            patch(
                "backend.copilot.graphiti.communities.get_graphiti_client",
                new=AsyncMock(return_value=client),
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(
                    return_value=(False, "below_activity_threshold", {"x": 1})
                ),
            ),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["skipped"] is True
        assert result["skip_reason"] == "below_activity_threshold"
        assert result["activity"] == {"x": 1}
        assert result["communities_built"] is None
        assert result["error"] is None
        # Critical: build_communities NOT called, no DETACH DELETE issued
        client.build_communities.assert_not_awaited()
        driver.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_bypasses_gate(self) -> None:
        """``force=True`` skips activity check; always rebuilds."""
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)
        client = MagicMock()
        client.graph_driver = driver
        client.build_communities = AsyncMock(return_value=[])

        with (
            patch(
                "backend.copilot.graphiti.communities.get_graphiti_client",
                new=AsyncMock(return_value=client),
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(),
            ) as mock_gate,
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef", force=True
            )

        assert result["forced"] is True
        assert result["skipped"] is False
        # The activity gate must not be consulted when force=True
        mock_gate.assert_not_called()
        # build_communities WAS called
        client.build_communities.assert_awaited_once()
