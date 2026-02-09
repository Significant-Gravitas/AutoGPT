"""Tests for block execution guards in RunBlockTool."""

from unittest.mock import MagicMock, patch

import pytest

from backend.api.features.chat.tools.models import ErrorResponse
from backend.api.features.chat.tools.run_block import RunBlockTool
from backend.data.block import BlockType

from ._test_data import make_session

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


class TestRunBlockFiltering:
    """Tests for block execution guards in RunBlockTool."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_excluded_block_type_returns_error(self):
        """Attempting to execute a block with excluded BlockType returns error."""
        session = make_session(user_id=_TEST_USER_ID)

        input_block = make_mock_block("input-block-id", "Input Block", BlockType.INPUT)

        with patch(
            "backend.api.features.chat.tools.run_block.get_block",
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
            "backend.api.features.chat.tools.run_block.get_block",
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
            "backend.api.features.chat.tools.run_block.get_block",
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
