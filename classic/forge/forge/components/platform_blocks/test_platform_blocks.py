"""Tests for PlatformBlocksComponent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.components.platform_blocks import (
    PlatformBlocksComponent,
    PlatformBlocksConfig,
)
from forge.components.platform_blocks.client import PlatformClient, PlatformClientError


@pytest.fixture
def mock_blocks():
    """Create mock block classes for testing."""

    class MockInputSchema:
        @classmethod
        def jsonschema(cls):
            return {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Input text"},
                },
                "required": ["text"],
            }

    class MockOutputSchema:
        @classmethod
        def jsonschema(cls):
            return {
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                },
            }

    class MockEmailBlock:
        def __init__(self):
            self.name = "SendEmailBlock"
            self.description = "Send an email message"
            self.categories = [MagicMock(value="Communication")]
            self.disabled = False
            self.input_schema = MockInputSchema
            self.output_schema = MockOutputSchema

    class MockSearchBlock:
        def __init__(self):
            self.name = "WebSearchBlock"
            self.description = "Search the web for information"
            self.categories = [MagicMock(value="Search")]
            self.disabled = False
            self.input_schema = MockInputSchema
            self.output_schema = MockOutputSchema

    class MockDisabledBlock:
        def __init__(self):
            self.name = "DisabledBlock"
            self.description = "A disabled block"
            self.categories = []
            self.disabled = True
            self.input_schema = MockInputSchema
            self.output_schema = MockOutputSchema

    return {
        "email-block-id": MockEmailBlock,
        "search-block-id": MockSearchBlock,
        "disabled-block-id": MockDisabledBlock,
    }


@pytest.fixture
def component_with_mocks(mock_blocks):
    """Create a PlatformBlocksComponent with mocked loader."""
    with (
        patch(
            "forge.components.platform_blocks.loader.is_platform_available",
            return_value=True,
        ),
        patch(
            "forge.components.platform_blocks.loader.load_blocks",
            return_value=mock_blocks,
        ),
    ):
        yield PlatformBlocksComponent()


@pytest.fixture
def component_unavailable():
    """Create a PlatformBlocksComponent when platform is unavailable."""
    with patch(
        "forge.components.platform_blocks.loader.is_platform_available",
        return_value=False,
    ):
        yield PlatformBlocksComponent()


class TestPlatformAvailability:
    """Tests for platform availability handling."""

    def test_component_disabled_when_platform_unavailable(self, component_unavailable):
        """Component should yield no commands when platform unavailable."""
        commands = list(component_unavailable.get_commands())
        assert len(commands) == 0

    def test_component_enabled_when_platform_available(self, component_with_mocks):
        """Component should yield commands when platform is available."""
        commands = list(component_with_mocks.get_commands())
        assert len(commands) == 2

    def test_get_resources_when_unavailable(self, component_unavailable):
        """Should not yield resources when platform unavailable."""
        resources = list(component_unavailable.get_resources())
        assert len(resources) == 0

    def test_get_resources_when_available(self, component_with_mocks):
        """Should yield resource info when platform available."""
        resources = list(component_with_mocks.get_resources())
        assert len(resources) == 1
        assert "3" in resources[0]  # 3 blocks loaded
        assert "search_blocks" in resources[0]


class TestSearchBlocks:
    """Tests for the search_blocks command."""

    def test_search_by_name(self, component_with_mocks, mock_blocks):
        """Search should find blocks by name."""
        with patch(
            "forge.components.platform_blocks.loader.load_blocks",
            return_value=mock_blocks,
        ):
            result_json = component_with_mocks.search_blocks("email")
            result = json.loads(result_json)

            assert result["count"] == 1
            assert result["blocks"][0]["name"] == "SendEmailBlock"
            assert result["blocks"][0]["id"] == "email-block-id"

    def test_search_by_description(self, component_with_mocks, mock_blocks):
        """Search should find blocks by description."""
        with patch(
            "forge.components.platform_blocks.loader.load_blocks",
            return_value=mock_blocks,
        ):
            result_json = component_with_mocks.search_blocks("web")
            result = json.loads(result_json)

            assert result["count"] == 1
            assert result["blocks"][0]["name"] == "WebSearchBlock"

    def test_search_by_category(self, component_with_mocks, mock_blocks):
        """Search should find blocks by category."""
        with patch(
            "forge.components.platform_blocks.loader.load_blocks",
            return_value=mock_blocks,
        ):
            result_json = component_with_mocks.search_blocks("Communication")
            result = json.loads(result_json)

            assert result["count"] == 1
            assert result["blocks"][0]["name"] == "SendEmailBlock"

    def test_search_excludes_disabled_blocks(self, component_with_mocks, mock_blocks):
        """Search should not return disabled blocks."""
        with patch(
            "forge.components.platform_blocks.loader.load_blocks",
            return_value=mock_blocks,
        ):
            result_json = component_with_mocks.search_blocks("disabled")
            result = json.loads(result_json)

            assert result["count"] == 0

    def test_search_no_results(self, component_with_mocks, mock_blocks):
        """Search with no matches should return empty results."""
        with patch(
            "forge.components.platform_blocks.loader.load_blocks",
            return_value=mock_blocks,
        ):
            result_json = component_with_mocks.search_blocks("nonexistent")
            result = json.loads(result_json)

            assert result["count"] == 0
            assert result["blocks"] == []

    def test_search_includes_schema(self, component_with_mocks, mock_blocks):
        """Search results should include input schema."""
        with patch(
            "forge.components.platform_blocks.loader.load_blocks",
            return_value=mock_blocks,
        ):
            result_json = component_with_mocks.search_blocks("email")
            result = json.loads(result_json)

            assert "input_schema" in result["blocks"][0]
            assert "properties" in result["blocks"][0]["input_schema"]


class TestExecuteBlock:
    """Tests for the execute_block command."""

    @pytest.mark.asyncio
    async def test_execute_block_success(self, component_with_mocks, mock_blocks):
        """Execute should return success with outputs."""
        mock_client = AsyncMock(spec=PlatformClient)
        mock_client.check_credentials.return_value = {
            "has_required_credentials": True,
        }
        mock_client.execute_block.return_value = {
            "outputs": {"result": "Email sent successfully"},
        }

        with patch(
            "forge.components.platform_blocks.loader.get_block",
            return_value=mock_blocks["email-block-id"](),
        ):
            component_with_mocks._client = mock_client

            result_json = await component_with_mocks.execute_block(
                block_id="email-block-id",
                input_data={"text": "Hello world"},
            )
            result = json.loads(result_json)

            assert result["success"] is True
            assert result["block"] == "SendEmailBlock"
            assert result["outputs"]["result"] == "Email sent successfully"

    @pytest.mark.asyncio
    async def test_execute_block_missing_credentials(
        self, component_with_mocks, mock_blocks
    ):
        """Execute should return error when credentials are missing."""
        mock_client = AsyncMock(spec=PlatformClient)
        mock_client.check_credentials.return_value = {
            "has_required_credentials": False,
            "missing_credentials": ["gmail_oauth"],
        }

        with patch(
            "forge.components.platform_blocks.loader.get_block",
            return_value=mock_blocks["email-block-id"](),
        ):
            component_with_mocks._client = mock_client

            result_json = await component_with_mocks.execute_block(
                block_id="email-block-id",
                input_data={"text": "Hello"},
            )
            result = json.loads(result_json)

            assert "error" in result
            assert result["error"] == "Missing required credentials"
            assert "gmail_oauth" in result["missing_credentials"]

    @pytest.mark.asyncio
    async def test_execute_block_api_error(self, component_with_mocks, mock_blocks):
        """Execute should handle API errors gracefully."""
        mock_client = AsyncMock(spec=PlatformClient)
        mock_client.check_credentials.return_value = {
            "has_required_credentials": True,
        }
        mock_client.execute_block.side_effect = PlatformClientError(
            "Block execution failed", status_code=500
        )

        with patch(
            "forge.components.platform_blocks.loader.get_block",
            return_value=mock_blocks["email-block-id"](),
        ):
            component_with_mocks._client = mock_client

            result_json = await component_with_mocks.execute_block(
                block_id="email-block-id",
                input_data={"text": "Hello"},
            )
            result = json.loads(result_json)

            assert "error" in result
            assert result["status_code"] == 500

    @pytest.mark.asyncio
    async def test_execute_block_credential_check_fails(
        self, component_with_mocks, mock_blocks
    ):
        """Execute should continue when credential check fails."""
        mock_client = AsyncMock(spec=PlatformClient)
        mock_client.check_credentials.side_effect = PlatformClientError(
            "Connection error"
        )
        mock_client.execute_block.return_value = {
            "outputs": {"result": "Success"},
        }

        with patch(
            "forge.components.platform_blocks.loader.get_block",
            return_value=mock_blocks["email-block-id"](),
        ):
            component_with_mocks._client = mock_client

            result_json = await component_with_mocks.execute_block(
                block_id="email-block-id",
                input_data={"text": "Hello"},
            )
            result = json.loads(result_json)

            # Should still succeed since execution worked
            assert result["success"] is True


class TestConfiguration:
    """Tests for PlatformBlocksConfig."""

    def test_default_configuration(self):
        """Default configuration should have expected values."""
        config = PlatformBlocksConfig()
        assert config.enabled is True
        assert config.platform_url == "https://platform.agpt.co"
        assert config.api_key == ""
        assert config.user_id == ""
        assert config.timeout == 60

    def test_custom_configuration(self):
        """Custom configuration should be respected."""
        config = PlatformBlocksConfig(
            enabled=False,
            platform_url="https://dev-builder.agpt.co",
            api_key="test-key",
            user_id="test-user",
            timeout=120,
        )
        assert config.enabled is False
        assert config.platform_url == "https://dev-builder.agpt.co"
        assert config.api_key == "test-key"
        assert config.user_id == "test-user"
        assert config.timeout == 120

    def test_component_respects_disabled_config(self):
        """Component should not yield commands when disabled."""
        with patch(
            "forge.components.platform_blocks.loader.is_platform_available",
            return_value=True,
        ):
            component = PlatformBlocksComponent(
                config=PlatformBlocksConfig(enabled=False)
            )
            commands = list(component.get_commands())
            assert len(commands) == 0


class TestProtocols:
    """Tests for protocol implementations."""

    def test_get_commands(self, component_with_mocks):
        """CommandProvider.get_commands should yield commands."""
        commands = list(component_with_mocks.get_commands())
        command_names = [c.names[0] for c in commands]
        assert "search_blocks" in command_names
        assert "execute_block" in command_names

    def test_command_aliases(self, component_with_mocks):
        """Commands should have proper aliases."""
        commands = list(component_with_mocks.get_commands())

        for cmd in commands:
            if "search_blocks" in cmd.names:
                assert "find_block" in cmd.names
            if "execute_block" in cmd.names:
                assert "run_block" in cmd.names


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
