"""Tests for the per-user community rebuild helper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .communities import _summarize_communities, rebuild_communities_for_user


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
        # Contract: result always carries elapsed_seconds, even on the
        # early-return invalid-user-id branch.
        assert result["elapsed_seconds"] is not None

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
        client.build_communities = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(
            "backend.copilot.graphiti.communities.get_graphiti_client",
            new=AsyncMock(return_value=client),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["error"] is not None
        assert "RuntimeError" in result["error"]
        # Always returns elapsed_seconds even on failure
        assert result["elapsed_seconds"] is not None
