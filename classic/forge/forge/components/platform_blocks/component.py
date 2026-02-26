"""Platform blocks component for classic agents.

Provides search_blocks and execute_block commands that call the platform API.
"""

import json
import logging
from typing import Any, Iterator

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema

from .client import PlatformClient, PlatformClientError
from .config import PlatformBlocksConfig

logger = logging.getLogger(__name__)


class PlatformBlocksComponent(
    DirectiveProvider,
    CommandProvider,
    ConfigurableComponent[PlatformBlocksConfig],
):
    """Provides search_blocks and execute_block commands via platform API."""

    config_class = PlatformBlocksConfig

    def __init__(self, config: PlatformBlocksConfig | None = None):
        ConfigurableComponent.__init__(self, config)
        self._client: PlatformClient | None = None
        self._blocks_cache: list[dict[str, Any]] | None = None

    @property
    def client(self) -> PlatformClient:
        """Get or create the platform client."""
        if self._client is None:
            api_key = ""
            if self.config.api_key:
                api_key = self.config.api_key.get_secret_value()
            self._client = PlatformClient(
                base_url=self.config.platform_url,
                api_key=api_key,
                timeout=self.config.timeout,
            )
        return self._client

    @property
    def is_configured(self) -> bool:
        """Check if the component is properly configured with an API key."""
        return bool(
            self.config.enabled
            and self.config.api_key
            and self.config.api_key.get_secret_value()
        )

    def get_resources(self) -> Iterator[str]:
        """Describe available resources."""
        if self.is_configured:
            yield (
                "Access to platform blocks via search_blocks and execute_block "
                "commands. Use search_blocks first to discover available blocks."
            )

    def get_commands(self) -> Iterator[Command]:
        """Provide available commands only if configured with API key."""
        if not self.is_configured:
            return
        yield self.search_blocks
        yield self.execute_block

    async def _get_blocks(self) -> list[dict[str, Any]]:
        """Get blocks from API, with caching."""
        if self._blocks_cache is not None:
            return self._blocks_cache

        try:
            self._blocks_cache = await self.client.list_blocks()
            logger.info(f"Loaded {len(self._blocks_cache)} blocks from platform API")
            return self._blocks_cache
        except PlatformClientError as e:
            logger.error(f"Failed to load blocks from API: {e}")
            return []

    @command(
        names=["search_blocks", "find_block"],
        description=(
            "Search for available platform blocks by name or description. "
            "Returns block IDs, names, descriptions, and input schemas. "
            "Use this FIRST to discover blocks before executing them."
        ),
        parameters={
            "query": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Search query (name, description, or category)",
                required=True,
            ),
        },
    )
    async def search_blocks(self, query: str) -> str:
        """Search blocks via platform API.

        Args:
            query: Search query for finding blocks.

        Returns:
            JSON string with search results.
        """
        try:
            blocks = await self._get_blocks()
            query_lower = query.lower()
            results: list[dict[str, Any]] = []

            for block in blocks:
                name = block.get("name", "")
                description = block.get("description", "")
                categories = [
                    c.get("category", "") for c in block.get("categories", [])
                ]

                # Check for match
                name_match = query_lower in name.lower()
                desc_match = query_lower in description.lower()
                cat_match = any(query_lower in c.lower() for c in categories)

                if name_match or desc_match or cat_match:
                    results.append(
                        {
                            "id": block.get("id"),
                            "name": name,
                            "description": description,
                            "categories": categories,
                            "input_schema": block.get("inputSchema", {}),
                        }
                    )

                    if len(results) >= 20:
                        break

            return json.dumps(
                {
                    "count": len(results),
                    "blocks": results,
                    "hint": "Use execute_block with the block 'id' to run a block",
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Error searching blocks: {e}")
            return json.dumps({"error": str(e)})

    @command(
        names=["execute_block", "run_block"],
        description=(
            "Execute a platform block by ID with input data. "
            "IMPORTANT: Use search_blocks FIRST to get the block ID and schema."
        ),
        parameters={
            "block_id": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Block ID (from search_blocks results)",
                required=True,
            ),
            "input_data": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="Input data matching the block's input schema",
                required=True,
            ),
        },
    )
    async def execute_block(self, block_id: str, input_data: dict[str, Any]) -> str:
        """Execute a block via platform API.

        Args:
            block_id: The block ID to execute.
            input_data: Input data matching the block's schema.

        Returns:
            JSON string with execution result.
        """
        try:
            # Get block name for better error messages
            blocks = await self._get_blocks()
            block_name = block_id
            for block in blocks:
                if block.get("id") == block_id:
                    block_name = block.get("name", block_id)
                    break

            # Execute the block
            result = await self.client.execute_block(block_id, input_data)

            return json.dumps(
                {
                    "success": True,
                    "block": block_name,
                    "block_id": block_id,
                    "outputs": result,
                },
                indent=2,
            )

        except PlatformClientError as e:
            logger.error(f"Platform API error executing block {block_id}: {e}")
            return json.dumps(
                {
                    "error": str(e),
                    "block_id": block_id,
                    "status_code": e.status_code,
                }
            )
        except Exception as e:
            logger.error(f"Error executing block {block_id}: {e}")
            return json.dumps({"error": str(e)})
