"""Tests for BlockDetailsResponse in RunBlockTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockType
from backend.data.model import CredentialsMetaInput
from backend.integrations.providers import ProviderName

from ._test_data import make_session
from .models import BlockDetailsResponse
from .run_block import RunBlockTool

_TEST_USER_ID = "test-user-run-block-details"


def make_mock_block_with_inputs(
    block_id: str, name: str, description: str = "Test description"
):
    """Create a mock block with input/output schemas for testing."""
    mock = MagicMock()
    mock.id = block_id
    mock.name = name
    mock.description = description
    mock.block_type = BlockType.STANDARD
    mock.disabled = False

    # Input schema with non-credential fields
    mock.input_schema = MagicMock()
    mock.input_schema.jsonschema.return_value = {
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "method": {"type": "string", "description": "HTTP method"},
        },
        "required": ["url"],
    }
    mock.input_schema.get_credentials_fields.return_value = {}
    mock.input_schema.get_credentials_fields_info.return_value = {}

    # Output schema
    mock.output_schema = MagicMock()
    mock.output_schema.jsonschema.return_value = {
        "properties": {
            "response": {"type": "object", "description": "HTTP response"},
            "error": {"type": "string", "description": "Error message"},
        }
    }

    return mock


@pytest.mark.asyncio(loop_scope="session")
async def test_run_block_returns_details_when_no_input_provided():
    """When run_block is called without input_data, it should return BlockDetailsResponse."""
    session = make_session(user_id=_TEST_USER_ID)

    # Create a block with inputs
    http_block = make_mock_block_with_inputs(
        "http-block-id", "HTTP Request", "Send HTTP requests"
    )

    with patch(
        "backend.copilot.tools.run_block.get_block",
        return_value=http_block,
    ):
        # Mock credentials check to return no missing credentials
        with patch.object(
            RunBlockTool,
            "_resolve_block_credentials",
            new_callable=AsyncMock,
            return_value=({}, []),  # (matched_credentials, missing_credentials)
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="http-block-id",
                input_data={},  # Empty input data
            )

    # Should return BlockDetailsResponse showing the schema
    assert isinstance(response, BlockDetailsResponse)
    assert response.block.id == "http-block-id"
    assert response.block.name == "HTTP Request"
    assert response.block.description == "Send HTTP requests"
    assert "url" in response.block.inputs["properties"]
    assert "method" in response.block.inputs["properties"]
    assert "response" in response.block.outputs["properties"]
    assert response.user_authenticated is True


@pytest.mark.asyncio(loop_scope="session")
async def test_run_block_returns_details_when_only_credentials_provided():
    """When only credentials are provided (no actual input), should return details."""
    session = make_session(user_id=_TEST_USER_ID)

    # Create a block with both credential and non-credential inputs
    mock = MagicMock()
    mock.id = "api-block-id"
    mock.name = "API Call"
    mock.description = "Make API calls"
    mock.block_type = BlockType.STANDARD
    mock.disabled = False

    mock.input_schema = MagicMock()
    mock.input_schema.jsonschema.return_value = {
        "properties": {
            "credentials": {"type": "object", "description": "API credentials"},
            "endpoint": {"type": "string", "description": "API endpoint"},
        },
        "required": ["credentials", "endpoint"],
    }
    mock.input_schema.get_credentials_fields.return_value = {"credentials": True}
    mock.input_schema.get_credentials_fields_info.return_value = {}

    mock.output_schema = MagicMock()
    mock.output_schema.jsonschema.return_value = {
        "properties": {"result": {"type": "object"}}
    }

    with patch(
        "backend.copilot.tools.run_block.get_block",
        return_value=mock,
    ):
        with patch.object(
            RunBlockTool,
            "_resolve_block_credentials",
            new_callable=AsyncMock,
            return_value=(
                {
                    "credentials": CredentialsMetaInput(
                        id="cred-id",
                        provider=ProviderName("test_provider"),
                        type="api_key",
                        title="Test Credential",
                    )
                },
                [],
            ),
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="api-block-id",
                input_data={"credentials": {"some": "cred"}},  # Only credential
            )

    # Should return details because no non-credential inputs provided
    assert isinstance(response, BlockDetailsResponse)
    assert response.block.id == "api-block-id"
    assert response.block.name == "API Call"
