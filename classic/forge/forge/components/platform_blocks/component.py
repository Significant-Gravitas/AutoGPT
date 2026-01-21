"""Platform blocks component for classic agents.

Provides search_blocks and execute_block commands:
- search_blocks: Uses local block registry (fast, offline)
- execute_block: Uses platform API (handles credentials)
"""

import json
import logging
from typing import Any, Iterator

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema

from . import loader
from .client import PlatformClient, PlatformClientError
from .config import PlatformBlocksConfig

logger = logging.getLogger(__name__)


class PlatformBlocksComponent(
    DirectiveProvider,
    CommandProvider,
    ConfigurableComponent[PlatformBlocksConfig],
):
    """Provides search_blocks and execute_block commands.

    - search_blocks: Uses local block registry (fast, offline)
    - execute_block: Uses platform API (handles credentials)
    """

    config_class = PlatformBlocksConfig

    def __init__(self, config: PlatformBlocksConfig | None = None):
        ConfigurableComponent.__init__(self, config)
        self._client: PlatformClient | None = None
        self._platform_available = loader.is_platform_available()

        if not self._platform_available:
            logger.warning(
                "Platform blocks not available - "
                "install autogpt_platform or add to PYTHONPATH"
            )

    @property
    def client(self) -> PlatformClient:
        """Get or create the platform client."""
        if self._client is None:
            self._client = PlatformClient(
                base_url=self.config.platform_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        return self._client

    def get_resources(self) -> Iterator[str]:
        """Describe available resources."""
        if self.config.enabled and self._platform_available:
            try:
                block_count = len(loader.load_blocks())
                yield (
                    f"Access to {block_count} platform blocks via search_blocks "
                    "and execute_block commands."
                )
            except Exception as e:
                logger.warning(f"Could not count blocks: {e}")

    def get_commands(self) -> Iterator[Command]:
        """Provide available commands."""
        if not self.config.enabled:
            return
        if not self._platform_available:
            return
        yield self.search_blocks
        yield self.execute_block

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
    def search_blocks(self, query: str) -> str:
        """Search blocks locally (fast, no network call).

        Args:
            query: Search query for finding blocks.

        Returns:
            JSON string with search results.
        """
        try:
            results = loader.search_blocks(query, limit=20)

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
            "IMPORTANT: Use search_blocks FIRST to get the block ID and schema. "
            "Credentials are automatically resolved via platform API."
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
        user_id = self.config.user_id or "classic_agent"

        try:
            # Get block info locally for better error messages
            block = loader.get_block(block_id)
            block_name = getattr(block, "name", block_id) if block else block_id

            # Check credentials first
            try:
                cred_check = await self.client.check_credentials(block_id, user_id)
                if not cred_check.get("has_required_credentials", True):
                    missing = cred_check.get("missing_credentials", [])
                    return json.dumps(
                        {
                            "error": "Missing required credentials",
                            "block": block_name,
                            "missing_credentials": missing,
                            "message": (
                                "Please configure the required credentials at "
                                f"{self.config.platform_url}/settings/credentials"
                            ),
                        },
                        indent=2,
                    )
            except PlatformClientError as e:
                logger.warning(f"Could not check credentials: {e}")
                # Continue anyway - execution will fail if creds are missing

            # Execute the block
            result = await self.client.execute_block(block_id, input_data, user_id)

            return json.dumps(
                {
                    "success": True,
                    "block": block_name,
                    "block_id": block_id,
                    "outputs": result.get("outputs", {}),
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
