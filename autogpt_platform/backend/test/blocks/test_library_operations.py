from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.system.library_operations import (
    AddToLibraryFromStoreBlock,
    LibraryAgent,
    ListLibraryAgentsBlock,
)


class TestListLibraryAgentsBlock:
    def setup_method(self):
        self.block = ListLibraryAgentsBlock()

    @pytest.mark.asyncio
    async def test_list_library_agents_maps_creator_name(self):
        mock_agent = SimpleNamespace(
            id="lib-agent-1",
            graph_id="graph-123",
            graph_version=3,
            name="Research Agent",
            description="Agent that conducts research",
            creator_name="Zoë O'Brien 🌙「東京」",
        )
        mock_result = SimpleNamespace(
            agents=[mock_agent],
            pagination=SimpleNamespace(
                total_items=1,
                current_page=1,
                total_pages=1,
            ),
        )

        with patch(
            "backend.blocks.system.library_operations.get_database_manager_async_client"
        ) as get_client:
            mock_client = MagicMock()
            mock_client.list_library_agents = AsyncMock(return_value=mock_result)
            get_client.return_value = mock_client

            result = await self.block._list_library_agents("user-test-id")

            assert len(result["agents"]) == 1
            agent = result["agents"][0]
            assert isinstance(agent, LibraryAgent)
            assert agent.creator == "Zoë O'Brien 🌙「東京」"
            assert agent.library_agent_id == "lib-agent-1"
            assert agent.agent_id == "graph-123"
            assert agent.agent_version == 3
            assert agent.agent_name == "Research Agent"
            assert agent.description == "Agent that conducts research"

    @pytest.mark.asyncio
    async def test_list_library_agents_fallback_when_creator_name_missing(self):
        mock_agent_with_creator = SimpleNamespace(
            id="lib-agent-2",
            graph_id="graph-456",
            graph_version=1,
            name="Fallback Agent",
            description="",
            creator="Alice",
        )
        mock_agent_empty = SimpleNamespace(
            id="lib-agent-3",
            graph_id="graph-789",
            graph_version=1,
            name="Unknown Agent",
        )
        mock_result = SimpleNamespace(
            agents=[mock_agent_with_creator, mock_agent_empty],
            pagination=SimpleNamespace(
                total_items=2,
                current_page=1,
                total_pages=1,
            ),
        )

        with patch(
            "backend.blocks.system.library_operations.get_database_manager_async_client"
        ) as get_client:
            mock_client = MagicMock()
            mock_client.list_library_agents = AsyncMock(return_value=mock_result)
            get_client.return_value = mock_client

            result = await self.block._list_library_agents("user-test-id")

            assert len(result["agents"]) == 2
            assert result["agents"][0].creator == "Alice"
            assert result["agents"][1].creator == ""

    @pytest.mark.asyncio
    async def test_run_yields_agents_and_agent(self):
        mock_agent = LibraryAgent(
            library_agent_id="lib-1",
            agent_id="graph-1",
            agent_version=1,
            agent_name="Agent One",
            creator="Zoë O'Brien 🌙「東京」",
        )

        with patch.object(
            self.block,
            "_list_library_agents",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = {
                "agents": [mock_agent],
                "total": 1,
                "page": 1,
                "total_pages": 1,
            }

            outputs = {}
            yielded_agents = []
            async for name, value in self.block.run(
                self.block.Input(), user_id="user-1"
            ):
                if name == "agent":
                    yielded_agents.append(value)
                else:
                    outputs[name] = value

            assert outputs["total_count"] == 1
            assert outputs["page"] == 1
            assert outputs["total_pages"] == 1
            assert len(outputs["agents"]) == 1
            assert outputs["agents"][0].creator == "Zoë O'Brien 🌙「東京」"
            assert len(yielded_agents) == 1
            assert yielded_agents[0].creator == "Zoë O'Brien 🌙「東京」"


class TestAddToLibraryFromStoreBlock:
    def setup_method(self):
        self.block = AddToLibraryFromStoreBlock()

    @pytest.mark.asyncio
    async def test_add_to_library_success(self):
        mock_db_agent = SimpleNamespace(
            id="lib-new-1",
            graph_id="graph-new-1",
            graph_version=1,
            name="Store Agent",
        )

        with patch(
            "backend.blocks.system.library_operations.get_database_manager_async_client"
        ) as get_client:
            mock_client = MagicMock()
            mock_client.add_store_agent_to_library = AsyncMock(
                return_value=mock_db_agent
            )
            get_client.return_value = mock_client

            outputs = {}
            async for name, value in self.block.run(
                self.block.Input(store_listing_version_id="version-123"),
                user_id="user-1",
            ):
                outputs[name] = value

            assert outputs["success"] is True
            assert outputs["library_agent_id"] == "lib-new-1"
            assert outputs["agent_id"] == "graph-new-1"
            assert outputs["agent_version"] == 1
            assert outputs["agent_name"] == "Store Agent"
            assert outputs["message"] == "Agent successfully added to library"
