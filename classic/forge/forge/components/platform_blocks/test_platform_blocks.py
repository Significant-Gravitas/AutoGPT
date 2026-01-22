"""Tests for PlatformBlocksComponent."""

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from forge.components.platform_blocks import (
    PlatformBlocksComponent,
    PlatformBlocksConfig,
)
from forge.components.platform_blocks.client import PlatformClient, PlatformClientError


@pytest.fixture
def mock_blocks_response():
    """Mock response from platform API /blocks endpoint."""
    return [
        {
            "id": "email-block-id",
            "name": "SendEmailBlock",
            "description": "Send an email message",
            "categories": [{"category": "COMMUNICATION"}],
            "inputSchema": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            },
            "outputSchema": {
                "type": "object",
                "properties": {"success": {"type": "boolean"}},
            },
        },
        {
            "id": "search-block-id",
            "name": "WebSearchBlock",
            "description": "Search the web for information",
            "categories": [{"category": "SEARCH"}],
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            "outputSchema": {
                "type": "object",
                "properties": {"results": {"type": "array"}},
            },
        },
        {
            "id": "ai-block-id",
            "name": "AITextGeneratorBlock",
            "description": "Generate text using AI",
            "categories": [{"category": "AI"}],
            "inputSchema": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
            },
            "outputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
    ]


@pytest.fixture
def component():
    """Create a PlatformBlocksComponent for testing with API key configured."""
    return PlatformBlocksComponent(
        config=PlatformBlocksConfig(api_key=SecretStr("test-api-key"))
    )


class TestSearchBlocks:
    """Tests for the search_blocks command."""

    @pytest.mark.asyncio
    async def test_search_by_name(self, component, mock_blocks_response):
        """Search should find blocks by name."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.return_value = mock_blocks_response

        result_json = await component.search_blocks("email")
        result = json.loads(result_json)

        assert result["count"] == 1
        assert result["blocks"][0]["name"] == "SendEmailBlock"
        assert result["blocks"][0]["id"] == "email-block-id"

    @pytest.mark.asyncio
    async def test_search_by_description(self, component, mock_blocks_response):
        """Search should find blocks by description."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.return_value = mock_blocks_response

        result_json = await component.search_blocks("web")
        result = json.loads(result_json)

        assert result["count"] == 1
        assert result["blocks"][0]["name"] == "WebSearchBlock"

    @pytest.mark.asyncio
    async def test_search_by_category(self, component, mock_blocks_response):
        """Search should find blocks by category."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.return_value = mock_blocks_response

        result_json = await component.search_blocks("COMMUNICATION")
        result = json.loads(result_json)

        assert result["count"] == 1
        assert result["blocks"][0]["name"] == "SendEmailBlock"

    @pytest.mark.asyncio
    async def test_search_no_results(self, component, mock_blocks_response):
        """Search with no matches should return empty results."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.return_value = mock_blocks_response

        result_json = await component.search_blocks("nonexistent")
        result = json.loads(result_json)

        assert result["count"] == 0
        assert result["blocks"] == []

    @pytest.mark.asyncio
    async def test_search_includes_schema(self, component, mock_blocks_response):
        """Search results should include input schema."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.return_value = mock_blocks_response

        result_json = await component.search_blocks("email")
        result = json.loads(result_json)

        assert "input_schema" in result["blocks"][0]
        assert "properties" in result["blocks"][0]["input_schema"]

    @pytest.mark.asyncio
    async def test_search_api_error(self, component):
        """Search should handle API errors gracefully by returning empty results."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.side_effect = PlatformClientError(
            "Connection failed", status_code=500
        )

        result_json = await component.search_blocks("test")
        result = json.loads(result_json)

        # On API error, returns empty results (graceful degradation)
        assert result["count"] == 0
        assert result["blocks"] == []

    @pytest.mark.asyncio
    async def test_search_caches_blocks(self, component, mock_blocks_response):
        """Blocks should be cached after first fetch."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.return_value = mock_blocks_response

        await component.search_blocks("email")
        await component.search_blocks("web")

        # Should only call API once
        assert component._client.list_blocks.call_count == 1


class TestExecuteBlock:
    """Tests for the execute_block command."""

    @pytest.mark.asyncio
    async def test_execute_block_success(self, component, mock_blocks_response):
        """Execute should return success with outputs."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.return_value = mock_blocks_response
        component._client.execute_block.return_value = {"success": True}

        result_json = await component.execute_block(
            block_id="email-block-id",
            input_data={"to": "test@example.com", "body": "Hello"},
        )
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["block"] == "SendEmailBlock"
        assert result["outputs"]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_block_api_error(self, component, mock_blocks_response):
        """Execute should handle API errors gracefully."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.return_value = mock_blocks_response
        component._client.execute_block.side_effect = PlatformClientError(
            "Block execution failed", status_code=500
        )

        result_json = await component.execute_block(
            block_id="email-block-id",
            input_data={"to": "test@example.com"},
        )
        result = json.loads(result_json)

        assert "error" in result
        assert result["status_code"] == 500

    @pytest.mark.asyncio
    async def test_execute_unknown_block(self, component, mock_blocks_response):
        """Execute with unknown block ID should still attempt execution."""
        component._client = AsyncMock(spec=PlatformClient)
        component._client.list_blocks.return_value = mock_blocks_response
        component._client.execute_block.return_value = {"result": "ok"}

        result_json = await component.execute_block(
            block_id="unknown-id",
            input_data={"test": "data"},
        )
        result = json.loads(result_json)

        # Should use block_id as name when not found
        assert result["block"] == "unknown-id"
        assert result["success"] is True


class TestConfiguration:
    """Tests for PlatformBlocksConfig."""

    def test_default_configuration(self):
        """Default configuration should have expected values."""
        config = PlatformBlocksConfig()
        assert config.enabled is True
        assert config.platform_url == "https://platform.agpt.co"
        assert config.api_key is None
        assert config.timeout == 60

    def test_custom_configuration(self):
        """Custom configuration should be respected."""
        config = PlatformBlocksConfig(
            enabled=False,
            platform_url="https://dev-builder.agpt.co",
            api_key=SecretStr("test-key"),
            timeout=120,
        )
        assert config.enabled is False
        assert config.platform_url == "https://dev-builder.agpt.co"
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == "test-key"
        assert config.timeout == 120

    def test_component_respects_disabled_config(self):
        """Component should not yield commands when disabled."""
        component = PlatformBlocksComponent(config=PlatformBlocksConfig(enabled=False))
        commands = list(component.get_commands())
        assert len(commands) == 0

    def test_component_disabled_without_api_key(self):
        """Component should not yield commands when api_key is not set."""
        component = PlatformBlocksComponent(
            config=PlatformBlocksConfig(enabled=True, api_key=None)
        )
        commands = list(component.get_commands())
        assert len(commands) == 0

    def test_component_enabled_with_api_key(self):
        """Component should yield commands when api_key is set."""
        component = PlatformBlocksComponent(
            config=PlatformBlocksConfig(enabled=True, api_key=SecretStr("test-key"))
        )
        commands = list(component.get_commands())
        assert len(commands) == 2


class TestProtocols:
    """Tests for protocol implementations."""

    def test_get_commands(self, component):
        """CommandProvider.get_commands should yield commands."""
        commands = list(component.get_commands())
        command_names = [c.names[0] for c in commands]
        assert "search_blocks" in command_names
        assert "execute_block" in command_names

    def test_command_aliases(self, component):
        """Commands should have proper aliases."""
        commands = list(component.get_commands())

        for cmd in commands:
            if "search_blocks" in cmd.names:
                assert "find_block" in cmd.names
            if "execute_block" in cmd.names:
                assert "run_block" in cmd.names

    def test_get_resources(self, component):
        """DirectiveProvider.get_resources should yield resource info."""
        resources = list(component.get_resources())
        assert len(resources) == 1
        assert "search_blocks" in resources[0]


class TestPlatformClient:
    """Tests for PlatformClient."""

    def test_client_initialization(self):
        """Client should initialize with correct settings."""
        client = PlatformClient(
            base_url="https://platform.agpt.co/",
            api_key="test-key",
            timeout=30,
        )
        assert client.base_url == "https://platform.agpt.co"  # Trailing slash removed
        assert client.api_key == "test-key"

    def test_client_headers_with_api_key(self):
        """Client should include auth header when API key is set."""
        client = PlatformClient(
            base_url="https://platform.agpt.co",
            api_key="test-key",
            timeout=30,
        )
        headers = client._headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_client_headers_without_api_key(self):
        """Client should not include auth header when API key is empty."""
        client = PlatformClient(
            base_url="https://platform.agpt.co",
            api_key="",
            timeout=30,
        )
        headers = client._headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"
