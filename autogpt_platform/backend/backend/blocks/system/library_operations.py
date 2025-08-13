import logging
from typing import Optional

from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.server.v2.store import exceptions as store_exceptions
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)


class LibraryAgent(BaseModel):
    """Model representing an agent in the user's library."""

    library_agent_id: str
    agent_id: str
    agent_version: int
    agent_name: str


class AddToLibraryFromStoreBlock(Block):
    """
    Block that adds an agent from the store to the user's library.
    This enables users to easily import agents from the marketplace into their personal collection.
    """

    class Input(BlockSchema):
        store_listing_version_id: str = SchemaField(
            description="The ID of the store listing version to add to library"
        )
        agent_name: Optional[str] = SchemaField(
            description="Optional custom name for the agent in your library",
            default=None,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the agent was successfully added to library"
        )
        library_agent_id: str = SchemaField(
            description="The ID of the library agent entry"
        )
        agent_id: str = SchemaField(description="The ID of the agent graph")
        agent_version: int = SchemaField(
            description="The version number of the agent graph"
        )
        agent_name: str = SchemaField(description="The name of the agent")
        message: str = SchemaField(description="Success or error message")

    def __init__(self):
        super().__init__(
            id="2602a7b1-3f4d-4e5f-9c8b-1a2b3c4d5e6f",
            description="Add an agent from the store to your personal library",
            categories={BlockCategory.BASIC},
            input_schema=AddToLibraryFromStoreBlock.Input,
            output_schema=AddToLibraryFromStoreBlock.Output,
            test_input={
                "store_listing_version_id": "test-listing-id",
                "agent_name": "My Custom Agent",
            },
            test_output=[
                ("success", True),
                ("library_agent_id", "test-library-id"),
                ("agent_id", "test-agent-id"),
                ("agent_version", 1),
                ("agent_name", "Test Agent"),
                ("message", "Agent successfully added to library"),
            ],
            test_mock={
                "_add_to_library": lambda *_, **__: {
                    "library_agent_id": "test-library-id",
                    "agent_id": "test-agent-id",
                    "agent_version": 1,
                    "agent_name": "Test Agent",
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            library_agent = await self._add_to_library(
                user_id=user_id,
                store_listing_version_id=input_data.store_listing_version_id,
                custom_name=input_data.agent_name,
            )

            yield "success", True
            yield "library_agent_id", library_agent.library_agent_id
            yield "agent_id", library_agent.agent_id
            yield "agent_version", library_agent.agent_version
            yield "agent_name", library_agent.agent_name
            yield "message", "Agent successfully added to library"

        except store_exceptions.AgentNotFoundError as e:
            logger.warning(f"Agent not found: {str(e)}")
            yield "success", False
            yield "library_agent_id", ""
            yield "agent_id", ""
            yield "agent_version", 0
            yield "agent_name", ""
            yield "message", f"Agent not found: {str(e)}"
        except Exception as e:
            logger.error(f"Failed to add agent to library: {str(e)}")
            yield "success", False
            yield "library_agent_id", ""
            yield "agent_id", ""
            yield "agent_version", 0
            yield "agent_name", ""
            yield "message", f"Failed to add agent to library: {str(e)}"

    async def _add_to_library(
        self,
        user_id: str,
        store_listing_version_id: str,
        custom_name: Optional[str] = None,
    ) -> LibraryAgent:
        """
        Add a store agent to the user's library using the existing library database function.
        """
        library_agent = (
            await get_database_manager_async_client().add_store_agent_to_library(
                store_listing_version_id=store_listing_version_id, user_id=user_id
            )
        )

        # If custom name is provided, we could update the library agent name here
        # For now, we'll just return the agent info
        agent_name = custom_name if custom_name else library_agent.name

        return LibraryAgent(
            agent_id=library_agent.id,
            agent_version=library_agent.graph_version,
            agent_name=agent_name,
            library_agent_id=library_agent.graph_id,
        )
