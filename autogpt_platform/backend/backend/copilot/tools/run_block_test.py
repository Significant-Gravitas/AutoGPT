"""Tests for block execution guards and input validation in RunBlockTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockType

from ._test_data import make_session
from .models import (
    BlockDetailsResponse,
    BlockOutputResponse,
    ErrorResponse,
    InputValidationErrorResponse,
)
from .run_block import RunBlockTool

_TEST_USER_ID = "test-user-run-block"


def make_mock_block(
    block_id: str, name: str, block_type: BlockType, disabled: bool = False
):
    """Create a mock block for testing."""
    mock = MagicMock()
    mock.id = block_id
    mock.name = name
    mock.block_type = block_type
    mock.disabled = disabled
    mock.input_schema = MagicMock()
    mock.input_schema.jsonschema.return_value = {"properties": {}, "required": []}
    mock.input_schema.get_credentials_fields_info.return_value = []
    return mock


def make_mock_block_with_schema(
    block_id: str,
    name: str,
    input_properties: dict,
    required_fields: list[str],
    output_properties: dict | None = None,
):
    """Create a mock block with a defined input/output schema for validation tests."""
    mock = MagicMock()
    mock.id = block_id
    mock.name = name
    mock.block_type = BlockType.STANDARD
    mock.disabled = False
    mock.description = f"Test block: {name}"

    input_schema = {
        "properties": input_properties,
        "required": required_fields,
    }
    mock.input_schema = MagicMock()
    mock.input_schema.jsonschema.return_value = input_schema
    mock.input_schema.get_credentials_fields_info.return_value = {}
    mock.input_schema.get_credentials_fields.return_value = {}

    output_schema = {
        "properties": output_properties or {"result": {"type": "string"}},
    }
    mock.output_schema = MagicMock()
    mock.output_schema.jsonschema.return_value = output_schema

    return mock


class TestRunBlockFiltering:
    """Tests for block execution guards in RunBlockTool."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_excluded_block_type_returns_error(self):
        """Attempting to execute a block with excluded BlockType returns error."""
        session = make_session(user_id=_TEST_USER_ID)

        input_block = make_mock_block("input-block-id", "Input Block", BlockType.INPUT)

        with patch(
            "backend.copilot.tools.run_block.get_block",
            return_value=input_block,
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="input-block-id",
                input_data={},
            )

        assert isinstance(response, ErrorResponse)
        assert "cannot be run directly in CoPilot" in response.message
        assert "designed for use within graphs only" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_excluded_block_id_returns_error(self):
        """Attempting to execute SmartDecisionMakerBlock returns error."""
        session = make_session(user_id=_TEST_USER_ID)

        smart_decision_id = "3b191d9f-356f-482d-8238-ba04b6d18381"
        smart_block = make_mock_block(
            smart_decision_id, "Smart Decision Maker", BlockType.STANDARD
        )

        with patch(
            "backend.copilot.tools.run_block.get_block",
            return_value=smart_block,
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id=smart_decision_id,
                input_data={},
            )

        assert isinstance(response, ErrorResponse)
        assert "cannot be run directly in CoPilot" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_non_excluded_block_passes_guard(self):
        """Non-excluded blocks pass the filtering guard (may fail later for other reasons)."""
        session = make_session(user_id=_TEST_USER_ID)

        standard_block = make_mock_block(
            "standard-id", "HTTP Request", BlockType.STANDARD
        )

        with patch(
            "backend.copilot.tools.run_block.get_block",
            return_value=standard_block,
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="standard-id",
                input_data={},
            )

        # Should NOT be an ErrorResponse about CoPilot exclusion
        # (may be other errors like missing credentials, but not the exclusion guard)
        if isinstance(response, ErrorResponse):
            assert "cannot be run directly in CoPilot" not in response.message


class TestRunBlockInputValidation:
    """Tests for input field validation in RunBlockTool.

    run_block rejects unknown input field names with InputValidationErrorResponse,
    preventing silent failures where incorrect keys would be ignored and the block
    would execute with default values instead of the caller's intended values.
    """

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unknown_input_fields_are_rejected(self):
        """run_block rejects unknown input fields instead of silently ignoring them.

        Scenario: The AI Text Generator block has a field called 'model' (for LLM model
        selection), but the LLM calling the tool guesses wrong and sends 'LLM_Model'
        instead. The block should reject the request and return the valid schema.
        """
        session = make_session(user_id=_TEST_USER_ID)

        mock_block = make_mock_block_with_schema(
            block_id="ai-text-gen-id",
            name="AI Text Generator",
            input_properties={
                "prompt": {"type": "string", "description": "The prompt to send"},
                "model": {
                    "type": "string",
                    "description": "The LLM model to use",
                    "default": "gpt-4o-mini",
                },
                "sys_prompt": {
                    "type": "string",
                    "description": "System prompt",
                    "default": "",
                },
            },
            required_fields=["prompt"],
            output_properties={"response": {"type": "string"}},
        )

        with patch(
            "backend.copilot.tools.run_block.get_block",
            return_value=mock_block,
        ):
            tool = RunBlockTool()

            # Provide 'prompt' (correct) but 'LLM_Model' instead of 'model' (wrong key)
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="ai-text-gen-id",
                input_data={
                    "prompt": "Write a haiku about coding",
                    "LLM_Model": "claude-opus-4-6",  # WRONG KEY - should be 'model'
                },
            )

        assert isinstance(response, InputValidationErrorResponse)
        assert "LLM_Model" in response.unrecognized_fields
        assert "Block was not executed" in response.message
        assert "inputs" in response.model_dump()  # valid schema included

    @pytest.mark.asyncio(loop_scope="session")
    async def test_multiple_wrong_keys_are_all_reported(self):
        """All unrecognized field names are reported in a single error response."""
        session = make_session(user_id=_TEST_USER_ID)

        mock_block = make_mock_block_with_schema(
            block_id="ai-text-gen-id",
            name="AI Text Generator",
            input_properties={
                "prompt": {"type": "string"},
                "model": {"type": "string", "default": "gpt-4o-mini"},
                "sys_prompt": {"type": "string", "default": ""},
                "retry": {"type": "integer", "default": 3},
            },
            required_fields=["prompt"],
        )

        with patch(
            "backend.copilot.tools.run_block.get_block",
            return_value=mock_block,
        ):
            tool = RunBlockTool()

            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="ai-text-gen-id",
                input_data={
                    "prompt": "Hello",  # correct
                    "llm_model": "claude-opus-4-6",  # WRONG - should be 'model'
                    "system_prompt": "Be helpful",  # WRONG - should be 'sys_prompt'
                    "retries": 5,  # WRONG - should be 'retry'
                },
            )

        assert isinstance(response, InputValidationErrorResponse)
        assert set(response.unrecognized_fields) == {
            "llm_model",
            "system_prompt",
            "retries",
        }
        assert "Block was not executed" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unknown_fields_rejected_even_with_missing_required(self):
        """Unknown fields are caught before the missing-required-fields check."""
        session = make_session(user_id=_TEST_USER_ID)

        mock_block = make_mock_block_with_schema(
            block_id="ai-text-gen-id",
            name="AI Text Generator",
            input_properties={
                "prompt": {"type": "string"},
                "model": {"type": "string", "default": "gpt-4o-mini"},
            },
            required_fields=["prompt"],
        )

        with patch(
            "backend.copilot.tools.run_block.get_block",
            return_value=mock_block,
        ):
            tool = RunBlockTool()

            # 'prompt' is missing AND 'LLM_Model' is an unknown field
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="ai-text-gen-id",
                input_data={
                    "LLM_Model": "claude-opus-4-6",  # wrong key, and 'prompt' is missing
                },
            )

        # Unknown fields are caught first
        assert isinstance(response, InputValidationErrorResponse)
        assert "LLM_Model" in response.unrecognized_fields

    @pytest.mark.asyncio(loop_scope="session")
    async def test_correct_inputs_still_execute(self):
        """Correct input field names pass validation and the block executes."""
        session = make_session(user_id=_TEST_USER_ID)

        mock_block = make_mock_block_with_schema(
            block_id="ai-text-gen-id",
            name="AI Text Generator",
            input_properties={
                "prompt": {"type": "string"},
                "model": {"type": "string", "default": "gpt-4o-mini"},
            },
            required_fields=["prompt"],
        )

        async def mock_execute(input_data, **kwargs):
            yield "response", "Generated text"

        mock_block.execute = mock_execute

        mock_workspace_db = MagicMock()
        mock_workspace_db.get_or_create_workspace = AsyncMock(
            return_value=MagicMock(id="test-workspace-id")
        )

        with (
            patch(
                "backend.copilot.tools.run_block.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.run_block.workspace_db",
                return_value=mock_workspace_db,
            ),
        ):
            tool = RunBlockTool()

            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="ai-text-gen-id",
                input_data={
                    "prompt": "Write a haiku",
                    "model": "gpt-4o-mini",  # correct field name
                },
            )

        assert isinstance(response, BlockOutputResponse)
        assert response.success is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_required_fields_returns_details(self):
        """Missing required fields returns BlockDetailsResponse with schema."""
        session = make_session(user_id=_TEST_USER_ID)

        mock_block = make_mock_block_with_schema(
            block_id="ai-text-gen-id",
            name="AI Text Generator",
            input_properties={
                "prompt": {"type": "string"},
                "model": {"type": "string", "default": "gpt-4o-mini"},
            },
            required_fields=["prompt"],
        )

        with patch(
            "backend.copilot.tools.run_block.get_block",
            return_value=mock_block,
        ):
            tool = RunBlockTool()

            # Only provide valid optional field, missing required 'prompt'
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="ai-text-gen-id",
                input_data={
                    "model": "gpt-4o-mini",  # valid but optional
                },
            )

        assert isinstance(response, BlockDetailsResponse)
