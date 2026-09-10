"""Tests for BlockDetailsResponse in RunBlockTool."""

from copy import deepcopy
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
        "backend.copilot.tools.helpers.get_block",
        return_value=http_block,
    ):
        # Mock credentials check to return no missing credentials
        with patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            new_callable=AsyncMock,
            return_value=({}, []),  # (matched_credentials, missing_credentials)
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="http-block-id",
                input_data={},  # Empty input data
                dry_run=False,
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
        "backend.copilot.tools.helpers.get_block",
        return_value=mock,
    ):
        with patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
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
                dry_run=False,
            )

    # Should return details because no non-credential inputs provided
    assert isinstance(response, BlockDetailsResponse)
    assert response.block.id == "api-block-id"
    assert response.block.name == "API Call"


def make_annotated_block(block_id: str = "annotated-block-id"):
    """A block whose schema carries builder-UI annotations, plus a property
    literally named ``secret`` so the strip can't be a blind key filter."""
    mock = make_mock_block_with_inputs(block_id, "Annotated Block")
    mock.input_schema.jsonschema.return_value = {
        "properties": {
            "model": {
                "type": "string",
                "enum": ["gpt-5", "claude-opus-5"],
                "llm_model": True,
                "llm_model_metadata": {
                    "gpt-5": {"price_tier": "high", "creator": "OpenAI"},
                },
                "advanced": False,
                "secret": False,
            },
            "secret": {"type": "string", "description": "A field named secret"},
        },
        "required": ["model"],
    }
    mock.output_schema.jsonschema.return_value = {
        "properties": {"result": {"type": "string", "advanced": True}},
    }
    return mock


async def _details_for(block, flag_on: bool) -> BlockDetailsResponse:
    session = make_session(user_id=_TEST_USER_ID)
    with (
        patch("backend.copilot.tools.helpers.get_block", return_value=block),
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            new_callable=AsyncMock,
            return_value=({}, []),
        ),
        patch(
            "backend.copilot.tools.run_block.is_feature_enabled",
            new_callable=AsyncMock,
            return_value=flag_on,
        ),
    ):
        response = await RunBlockTool()._execute(
            user_id=_TEST_USER_ID,
            session=session,
            block_id=block.id,
            input_data={},
            dry_run=False,
        )
    assert isinstance(response, BlockDetailsResponse)
    return response


@pytest.mark.asyncio(loop_scope="session")
async def test_flag_off_leaves_the_schema_byte_identical():
    block = make_annotated_block()
    response = await _details_for(block, flag_on=False)
    assert response.block.inputs == block.input_schema.jsonschema.return_value
    assert response.block.outputs == block.output_schema.jsonschema.return_value


@pytest.mark.asyncio(loop_scope="session")
async def test_flag_on_strips_presentation_annotations_from_both_schemas():
    block = make_annotated_block()
    response = await _details_for(block, flag_on=True)
    model = response.block.inputs["properties"]["model"]
    assert set(model) == {"type", "enum"}
    assert model["enum"] == ["gpt-5", "claude-opus-5"]
    assert "advanced" not in response.block.outputs["properties"]["result"]


@pytest.mark.asyncio(loop_scope="session")
async def test_flag_on_keeps_a_property_that_shares_an_annotation_name():
    """``secret`` is both a UI annotation and a legal field name."""
    response = await _details_for(make_annotated_block(), flag_on=True)
    assert "secret" in response.block.inputs["properties"]
    assert response.block.inputs["required"] == ["model"]


@pytest.mark.asyncio(loop_scope="session")
async def test_the_source_schema_is_not_mutated_by_the_strip():
    """The frontend reads the same registry object through other endpoints."""
    block = make_annotated_block()
    original = deepcopy(block.input_schema.jsonschema.return_value)
    await _details_for(block, flag_on=True)
    assert block.input_schema.jsonschema.return_value == original
