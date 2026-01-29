"""
Tests for library agent fetching functionality in agent generator.

This test suite verifies the search-based library agent fetching,
including the combination of library and marketplace agents.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.chat.tools.agent_generator import core


class TestGetLibraryAgentsForGeneration:
    """Test get_library_agents_for_generation function."""

    @pytest.mark.asyncio
    async def test_fetches_agents_with_search_term(self):
        """Test that search_term is passed to the library db."""
        # Create a mock agent with proper attribute values
        mock_agent = MagicMock()
        mock_agent.graph_id = "agent-123"
        mock_agent.graph_version = 1
        mock_agent.name = "Email Agent"
        mock_agent.description = "Sends emails"
        mock_agent.input_schema = {"properties": {}}
        mock_agent.output_schema = {"properties": {}}

        mock_response = MagicMock()
        mock_response.agents = [mock_agent]

        with patch.object(
            core.library_db,
            "list_library_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_list:
            result = await core.get_library_agents_for_generation(
                user_id="user-123",
                search_query="send email",
            )

            # Verify search_term was passed
            mock_list.assert_called_once_with(
                user_id="user-123",
                search_term="send email",
                page=1,
                page_size=15,
            )

        # Verify result format
        assert len(result) == 1
        assert result[0]["graph_id"] == "agent-123"
        assert result[0]["name"] == "Email Agent"

    @pytest.mark.asyncio
    async def test_excludes_specified_graph_id(self):
        """Test that agents with excluded graph_id are filtered out."""
        mock_response = MagicMock()
        mock_response.agents = [
            MagicMock(
                graph_id="agent-123",
                graph_version=1,
                name="Agent 1",
                description="First agent",
                input_schema={},
                output_schema={},
            ),
            MagicMock(
                graph_id="agent-456",
                graph_version=1,
                name="Agent 2",
                description="Second agent",
                input_schema={},
                output_schema={},
            ),
        ]

        with patch.object(
            core.library_db,
            "list_library_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await core.get_library_agents_for_generation(
                user_id="user-123",
                exclude_graph_id="agent-123",
            )

        # Verify the excluded agent is not in results
        assert len(result) == 1
        assert result[0]["graph_id"] == "agent-456"

    @pytest.mark.asyncio
    async def test_respects_max_results(self):
        """Test that max_results parameter limits the page_size."""
        mock_response = MagicMock()
        mock_response.agents = []

        with patch.object(
            core.library_db,
            "list_library_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_list:
            await core.get_library_agents_for_generation(
                user_id="user-123",
                max_results=5,
            )

            # Verify page_size was set to max_results
            mock_list.assert_called_once_with(
                user_id="user-123",
                search_term=None,
                page=1,
                page_size=5,
            )


class TestSearchMarketplaceAgentsForGeneration:
    """Test search_marketplace_agents_for_generation function."""

    @pytest.mark.asyncio
    async def test_searches_marketplace_with_query(self):
        """Test that marketplace is searched with the query."""
        mock_response = MagicMock()
        mock_response.agents = [
            MagicMock(
                agent_name="Public Agent",
                description="A public agent",
                sub_heading="Does something useful",
                creator="creator-1",
            )
        ]

        # The store_db is dynamically imported, so patch the import path
        with patch(
            "backend.api.features.store.db.get_store_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_search:
            result = await core.search_marketplace_agents_for_generation(
                search_query="automation",
                max_results=10,
            )

            mock_search.assert_called_once_with(
                search_query="automation",
                page=1,
                page_size=10,
            )

        assert len(result) == 1
        assert result[0]["name"] == "Public Agent"
        assert result[0]["is_marketplace_agent"] is True

    @pytest.mark.asyncio
    async def test_handles_marketplace_error_gracefully(self):
        """Test that marketplace errors don't crash the function."""
        with patch(
            "backend.api.features.store.db.get_store_agents",
            new_callable=AsyncMock,
            side_effect=Exception("Marketplace unavailable"),
        ):
            result = await core.search_marketplace_agents_for_generation(
                search_query="test"
            )

        # Should return empty list, not raise exception
        assert result == []


class TestGetAllRelevantAgentsForGeneration:
    """Test get_all_relevant_agents_for_generation function."""

    @pytest.mark.asyncio
    async def test_combines_library_and_marketplace_agents(self):
        """Test that agents from both sources are combined."""
        library_agents = [
            {
                "graph_id": "lib-123",
                "graph_version": 1,
                "name": "Library Agent",
                "description": "From library",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        marketplace_agents = [
            {
                "name": "Market Agent",
                "description": "From marketplace",
                "sub_heading": "Sub heading",
                "creator": "creator-1",
                "is_marketplace_agent": True,
            }
        ]

        with patch.object(
            core,
            "get_library_agents_for_generation",
            new_callable=AsyncMock,
            return_value=library_agents,
        ):
            with patch.object(
                core,
                "search_marketplace_agents_for_generation",
                new_callable=AsyncMock,
                return_value=marketplace_agents,
            ):
                result = await core.get_all_relevant_agents_for_generation(
                    user_id="user-123",
                    search_query="test query",
                    include_marketplace=True,
                )

        # Library agents should come first
        assert len(result) == 2
        assert result[0]["name"] == "Library Agent"
        assert result[1]["name"] == "Market Agent"

    @pytest.mark.asyncio
    async def test_deduplicates_by_name(self):
        """Test that marketplace agents with same name as library are excluded."""
        library_agents = [
            {
                "graph_id": "lib-123",
                "graph_version": 1,
                "name": "Shared Agent",
                "description": "From library",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        marketplace_agents = [
            {
                "name": "Shared Agent",  # Same name, should be deduplicated
                "description": "From marketplace",
                "sub_heading": "Sub heading",
                "creator": "creator-1",
                "is_marketplace_agent": True,
            },
            {
                "name": "Unique Agent",
                "description": "Only in marketplace",
                "sub_heading": "Sub heading",
                "creator": "creator-2",
                "is_marketplace_agent": True,
            },
        ]

        with patch.object(
            core,
            "get_library_agents_for_generation",
            new_callable=AsyncMock,
            return_value=library_agents,
        ):
            with patch.object(
                core,
                "search_marketplace_agents_for_generation",
                new_callable=AsyncMock,
                return_value=marketplace_agents,
            ):
                result = await core.get_all_relevant_agents_for_generation(
                    user_id="user-123",
                    search_query="test",
                    include_marketplace=True,
                )

        # Shared Agent from marketplace should be excluded
        assert len(result) == 2
        names = [a["name"] for a in result]
        assert "Shared Agent" in names
        assert "Unique Agent" in names

    @pytest.mark.asyncio
    async def test_skips_marketplace_when_disabled(self):
        """Test that marketplace is not searched when include_marketplace=False."""
        library_agents = [
            {
                "graph_id": "lib-123",
                "graph_version": 1,
                "name": "Library Agent",
                "description": "From library",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        with patch.object(
            core,
            "get_library_agents_for_generation",
            new_callable=AsyncMock,
            return_value=library_agents,
        ):
            with patch.object(
                core,
                "search_marketplace_agents_for_generation",
                new_callable=AsyncMock,
            ) as mock_marketplace:
                result = await core.get_all_relevant_agents_for_generation(
                    user_id="user-123",
                    search_query="test",
                    include_marketplace=False,
                )

        # Marketplace should not be called
        mock_marketplace.assert_not_called()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_skips_marketplace_when_no_search_query(self):
        """Test that marketplace is not searched without a search query."""
        library_agents = [
            {
                "graph_id": "lib-123",
                "graph_version": 1,
                "name": "Library Agent",
                "description": "From library",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        with patch.object(
            core,
            "get_library_agents_for_generation",
            new_callable=AsyncMock,
            return_value=library_agents,
        ):
            with patch.object(
                core,
                "search_marketplace_agents_for_generation",
                new_callable=AsyncMock,
            ) as mock_marketplace:
                result = await core.get_all_relevant_agents_for_generation(
                    user_id="user-123",
                    search_query=None,  # No search query
                    include_marketplace=True,
                )

        # Marketplace should not be called without search query
        mock_marketplace.assert_not_called()
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
