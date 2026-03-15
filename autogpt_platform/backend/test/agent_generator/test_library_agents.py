"""
Tests for library agent fetching functionality in agent generator.

This test suite verifies the search-based library agent fetching,
including the combination of library and marketplace agents.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.agent_generator import core


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
        mock_agent.recent_executions = []

        mock_response = MagicMock()
        mock_response.agents = [mock_agent]

        mock_db = MagicMock()
        mock_db.list_library_agents = AsyncMock(return_value=mock_response)

        with patch.object(
            core,
            "library_db",
            return_value=mock_db,
        ):
            result = await core.get_library_agents_for_generation(
                user_id="user-123",
                search_query="send email",
            )

            mock_db.list_library_agents.assert_called_once_with(
                user_id="user-123",
                search_term="send email",
                page=1,
                page_size=15,
                include_executions=True,
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
                recent_executions=[],
            ),
            MagicMock(
                graph_id="agent-456",
                graph_version=1,
                name="Agent 2",
                description="Second agent",
                input_schema={},
                output_schema={},
                recent_executions=[],
            ),
        ]

        mock_db = MagicMock()
        mock_db.list_library_agents = AsyncMock(return_value=mock_response)

        with patch.object(
            core,
            "library_db",
            return_value=mock_db,
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

        mock_db = MagicMock()
        mock_db.list_library_agents = AsyncMock(return_value=mock_response)

        with patch.object(
            core,
            "library_db",
            return_value=mock_db,
        ):
            await core.get_library_agents_for_generation(
                user_id="user-123",
                max_results=5,
            )

            mock_db.list_library_agents.assert_called_once_with(
                user_id="user-123",
                search_term=None,
                page=1,
                page_size=5,
                include_executions=True,
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
                agent_graph_id="graph-123",
            )
        ]

        mock_graph = MagicMock()
        mock_graph.id = "graph-123"
        mock_graph.version = 1
        mock_graph.input_schema = {"type": "object"}
        mock_graph.output_schema = {"type": "object"}

        mock_store_db = MagicMock()
        mock_store_db.get_store_agents = AsyncMock(return_value=mock_response)

        mock_graph_db = MagicMock()
        mock_graph_db.get_store_listed_graphs = AsyncMock(
            return_value={"graph-123": mock_graph}
        )

        with (
            patch.object(core, "store_db", return_value=mock_store_db),
            patch.object(core, "graph_db", return_value=mock_graph_db),
        ):
            result = await core.search_marketplace_agents_for_generation(
                search_query="automation",
                max_results=10,
            )

            mock_store_db.get_store_agents.assert_called_once_with(
                search_query="automation",
                page=1,
                page_size=10,
            )

        assert len(result) == 1
        assert result[0]["name"] == "Public Agent"
        assert result[0]["graph_id"] == "graph-123"

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
                "graph_id": "market-456",
                "graph_version": 1,
                "name": "Market Agent",
                "description": "From marketplace",
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
    async def test_deduplicates_by_graph_id(self):
        """Test that marketplace agents with same graph_id as library are excluded."""
        library_agents = [
            {
                "graph_id": "shared-123",
                "graph_version": 1,
                "name": "Shared Agent",
                "description": "From library",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        marketplace_agents = [
            {
                "graph_id": "shared-123",  # Same graph_id, should be deduplicated
                "graph_version": 1,
                "name": "Shared Agent",
                "description": "From marketplace",
                "input_schema": {},
                "output_schema": {},
            },
            {
                "graph_id": "unique-456",
                "graph_version": 1,
                "name": "Unique Agent",
                "description": "Only in marketplace",
                "input_schema": {},
                "output_schema": {},
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

        # Shared Agent from marketplace should be excluded by graph_id
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


class TestExtractSearchTermsFromSteps:
    """Test extract_search_terms_from_steps function."""

    def test_extracts_terms_from_instructions_type(self):
        """Test extraction from valid instructions decomposition result."""
        decomposition_result = {
            "type": "instructions",
            "steps": [
                {
                    "description": "Send an email notification",
                    "block_name": "GmailSendBlock",
                },
                {"description": "Fetch weather data", "action": "Get weather API"},
            ],
        }

        result = core.extract_search_terms_from_steps(decomposition_result)

        assert "Send an email notification" in result
        assert "GmailSendBlock" in result
        assert "Fetch weather data" in result
        assert "Get weather API" in result

    def test_returns_empty_for_non_instructions_type(self):
        """Test that non-instructions types return empty list."""
        decomposition_result = {
            "type": "clarifying_questions",
            "questions": [{"question": "What email?"}],
        }

        result = core.extract_search_terms_from_steps(decomposition_result)

        assert result == []

    def test_deduplicates_terms_case_insensitively(self):
        """Test that duplicate terms are removed (case-insensitive)."""
        decomposition_result = {
            "type": "instructions",
            "steps": [
                {"description": "Send Email", "name": "send email"},
                {"description": "Other task"},
            ],
        }

        result = core.extract_search_terms_from_steps(decomposition_result)

        # Should only have one "send email" variant
        email_terms = [t for t in result if "email" in t.lower()]
        assert len(email_terms) == 1

    def test_filters_short_terms(self):
        """Test that terms with 3 or fewer characters are filtered out."""
        decomposition_result = {
            "type": "instructions",
            "steps": [
                {"description": "ab", "action": "xyz"},  # Both too short
                {"description": "Valid term here"},
            ],
        }

        result = core.extract_search_terms_from_steps(decomposition_result)

        assert "ab" not in result
        assert "xyz" not in result
        assert "Valid term here" in result

    def test_handles_empty_steps(self):
        """Test handling of empty steps list."""
        decomposition_result = {
            "type": "instructions",
            "steps": [],
        }

        result = core.extract_search_terms_from_steps(decomposition_result)

        assert result == []


class TestEnrichLibraryAgentsFromSteps:
    """Test enrich_library_agents_from_steps function."""

    @pytest.mark.asyncio
    async def test_enriches_with_additional_agents(self):
        """Test that additional agents are found based on steps."""
        existing_agents = [
            {
                "graph_id": "existing-123",
                "graph_version": 1,
                "name": "Existing Agent",
                "description": "Already fetched",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        additional_agents = [
            {
                "graph_id": "new-456",
                "graph_version": 1,
                "name": "Email Agent",
                "description": "For sending emails",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        decomposition_result = {
            "type": "instructions",
            "steps": [
                {"description": "Send email notification"},
            ],
        }

        with patch.object(
            core,
            "get_all_relevant_agents_for_generation",
            new_callable=AsyncMock,
            return_value=additional_agents,
        ):
            result = await core.enrich_library_agents_from_steps(
                user_id="user-123",
                decomposition_result=decomposition_result,
                existing_agents=existing_agents,
            )

        # Should have both existing and new agents
        assert len(result) == 2
        names = [a["name"] for a in result]
        assert "Existing Agent" in names
        assert "Email Agent" in names

    @pytest.mark.asyncio
    async def test_deduplicates_by_graph_id(self):
        """Test that agents with same graph_id are not duplicated."""
        existing_agents = [
            {
                "graph_id": "agent-123",
                "graph_version": 1,
                "name": "Existing Agent",
                "description": "Already fetched",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        # Additional search returns same agent
        additional_agents = [
            {
                "graph_id": "agent-123",  # Same ID
                "graph_version": 1,
                "name": "Existing Agent Copy",
                "description": "Same agent different name",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        decomposition_result = {
            "type": "instructions",
            "steps": [{"description": "Some action"}],
        }

        with patch.object(
            core,
            "get_all_relevant_agents_for_generation",
            new_callable=AsyncMock,
            return_value=additional_agents,
        ):
            result = await core.enrich_library_agents_from_steps(
                user_id="user-123",
                decomposition_result=decomposition_result,
                existing_agents=existing_agents,
            )

        # Should not duplicate
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_deduplicates_by_name(self):
        """Test that agents with same name are not duplicated."""
        existing_agents = [
            {
                "graph_id": "agent-123",
                "graph_version": 1,
                "name": "Email Agent",
                "description": "Already fetched",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        # Additional search returns agent with same name but different ID
        additional_agents = [
            {
                "graph_id": "agent-456",  # Different ID
                "graph_version": 1,
                "name": "Email Agent",  # Same name
                "description": "Different agent same name",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        decomposition_result = {
            "type": "instructions",
            "steps": [{"description": "Send email"}],
        }

        with patch.object(
            core,
            "get_all_relevant_agents_for_generation",
            new_callable=AsyncMock,
            return_value=additional_agents,
        ):
            result = await core.enrich_library_agents_from_steps(
                user_id="user-123",
                decomposition_result=decomposition_result,
                existing_agents=existing_agents,
            )

        # Should not duplicate by name
        assert len(result) == 1
        assert result[0].get("graph_id") == "agent-123"  # Original kept

    @pytest.mark.asyncio
    async def test_returns_existing_when_no_steps(self):
        """Test that existing agents are returned when no search terms extracted."""
        existing_agents = [
            {
                "graph_id": "existing-123",
                "graph_version": 1,
                "name": "Existing Agent",
                "description": "Already fetched",
                "input_schema": {},
                "output_schema": {},
            }
        ]

        decomposition_result = {
            "type": "clarifying_questions",  # Not instructions type
            "questions": [],
        }

        result = await core.enrich_library_agents_from_steps(
            user_id="user-123",
            decomposition_result=decomposition_result,
            existing_agents=existing_agents,
        )

        # Should return existing unchanged
        assert result == existing_agents

    @pytest.mark.asyncio
    async def test_limits_search_terms_to_three(self):
        """Test that only first 3 search terms are used."""
        existing_agents = []

        decomposition_result = {
            "type": "instructions",
            "steps": [
                {"description": "First action"},
                {"description": "Second action"},
                {"description": "Third action"},
                {"description": "Fourth action"},
                {"description": "Fifth action"},
            ],
        }

        call_count = 0

        async def mock_get_agents(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return []

        with patch.object(
            core,
            "get_all_relevant_agents_for_generation",
            side_effect=mock_get_agents,
        ):
            await core.enrich_library_agents_from_steps(
                user_id="user-123",
                decomposition_result=decomposition_result,
                existing_agents=existing_agents,
            )

        # Should only make 3 calls (limited to first 3 terms)
        assert call_count == 3


class TestExtractUuidsFromText:
    """Test extract_uuids_from_text function."""

    def test_extracts_single_uuid(self):
        """Test extraction of a single UUID from text."""
        text = "Use my agent 46631191-e8a8-486f-ad90-84f89738321d for this task"
        result = core.extract_uuids_from_text(text)
        assert len(result) == 1
        assert "46631191-e8a8-486f-ad90-84f89738321d" in result

    def test_extracts_multiple_uuids(self):
        """Test extraction of multiple UUIDs from text."""
        text = (
            "Combine agents 11111111-1111-4111-8111-111111111111 "
            "and 22222222-2222-4222-9222-222222222222"
        )
        result = core.extract_uuids_from_text(text)
        assert len(result) == 2
        assert "11111111-1111-4111-8111-111111111111" in result
        assert "22222222-2222-4222-9222-222222222222" in result

    def test_deduplicates_uuids(self):
        """Test that duplicate UUIDs are deduplicated."""
        text = (
            "Use 46631191-e8a8-486f-ad90-84f89738321d twice: "
            "46631191-e8a8-486f-ad90-84f89738321d"
        )
        result = core.extract_uuids_from_text(text)
        assert len(result) == 1

    def test_normalizes_to_lowercase(self):
        """Test that UUIDs are normalized to lowercase."""
        text = "Use 46631191-E8A8-486F-AD90-84F89738321D"
        result = core.extract_uuids_from_text(text)
        assert result[0] == "46631191-e8a8-486f-ad90-84f89738321d"

    def test_returns_empty_for_no_uuids(self):
        """Test that empty list is returned when no UUIDs found."""
        text = "Create an email agent that sends notifications"
        result = core.extract_uuids_from_text(text)
        assert result == []

    def test_ignores_invalid_uuids(self):
        """Test that invalid UUID-like strings are ignored."""
        text = "Not a valid UUID: 12345678-1234-1234-1234-123456789abc"
        result = core.extract_uuids_from_text(text)
        # UUID v4 requires specific patterns (4 in third group, 8/9/a/b in fourth)
        assert len(result) == 0


class TestGetLibraryAgentById:
    """Test get_library_agent_by_id function (alias: get_library_agent_by_graph_id)."""

    @pytest.mark.asyncio
    async def test_returns_agent_when_found_by_graph_id(self):
        """Test that agent is returned when found by graph_id."""
        mock_agent = MagicMock()
        mock_agent.graph_id = "agent-123"
        mock_agent.graph_version = 1
        mock_agent.name = "Test Agent"
        mock_agent.description = "Test description"
        mock_agent.input_schema = {"properties": {}}
        mock_agent.output_schema = {"properties": {}}

        mock_db = MagicMock()
        mock_db.get_library_agent_by_graph_id = AsyncMock(return_value=mock_agent)

        with patch.object(core, "library_db", return_value=mock_db):
            result = await core.get_library_agent_by_id("user-123", "agent-123")

        assert result is not None
        assert result["graph_id"] == "agent-123"
        assert result["name"] == "Test Agent"

    @pytest.mark.asyncio
    async def test_falls_back_to_library_agent_id(self):
        """Test that lookup falls back to library agent ID when graph_id not found."""
        mock_agent = MagicMock()
        mock_agent.graph_id = "graph-456"  # Different from the lookup ID
        mock_agent.graph_version = 1
        mock_agent.name = "Library Agent"
        mock_agent.description = "Found by library ID"
        mock_agent.input_schema = {"properties": {}}
        mock_agent.output_schema = {"properties": {}}

        mock_db = MagicMock()
        mock_db.get_library_agent_by_graph_id = AsyncMock(return_value=None)
        mock_db.get_library_agent = AsyncMock(return_value=mock_agent)

        with patch.object(core, "library_db", return_value=mock_db):
            result = await core.get_library_agent_by_id("user-123", "library-id-123")

        assert result is not None
        assert result["graph_id"] == "graph-456"
        assert result["name"] == "Library Agent"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found_by_either_method(self):
        """Test that None is returned when agent not found by either method."""
        mock_db = MagicMock()
        mock_db.get_library_agent_by_graph_id = AsyncMock(return_value=None)
        mock_db.get_library_agent = AsyncMock(
            side_effect=core.NotFoundError("Not found")
        )

        with patch.object(core, "library_db", return_value=mock_db):
            result = await core.get_library_agent_by_id("user-123", "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        """Test that None is returned when exception occurs in both lookups."""
        mock_db = MagicMock()
        mock_db.get_library_agent_by_graph_id = AsyncMock(
            side_effect=Exception("Database error")
        )
        mock_db.get_library_agent = AsyncMock(side_effect=Exception("Database error"))

        with patch.object(core, "library_db", return_value=mock_db):
            result = await core.get_library_agent_by_id("user-123", "agent-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_alias_works(self):
        """Test that get_library_agent_by_graph_id is an alias."""
        assert core.get_library_agent_by_graph_id is core.get_library_agent_by_id


class TestGetAllRelevantAgentsWithUuids:
    """Test UUID extraction in get_all_relevant_agents_for_generation."""

    @pytest.mark.asyncio
    async def test_fetches_explicitly_mentioned_agents(self):
        """Test that agents mentioned by UUID are fetched directly."""
        mock_agent = MagicMock()
        mock_agent.graph_id = "46631191-e8a8-486f-ad90-84f89738321d"
        mock_agent.graph_version = 1
        mock_agent.name = "Mentioned Agent"
        mock_agent.description = "Explicitly mentioned"
        mock_agent.input_schema = {}
        mock_agent.output_schema = {}

        mock_response = MagicMock()
        mock_response.agents = []

        mock_db = MagicMock()
        mock_db.get_library_agent_by_graph_id = AsyncMock(return_value=mock_agent)
        mock_db.list_library_agents = AsyncMock(return_value=mock_response)

        with patch.object(core, "library_db", return_value=mock_db):
            result = await core.get_all_relevant_agents_for_generation(
                user_id="user-123",
                search_query="Use agent 46631191-e8a8-486f-ad90-84f89738321d",
                include_marketplace=False,
            )

        assert len(result) == 1
        assert result[0].get("graph_id") == "46631191-e8a8-486f-ad90-84f89738321d"
        assert result[0].get("name") == "Mentioned Agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
