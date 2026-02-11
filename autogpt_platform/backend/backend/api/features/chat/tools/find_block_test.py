"""Tests for block filtering in FindBlockTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.chat.tools.find_block import (
    COPILOT_EXCLUDED_BLOCK_IDS,
    COPILOT_EXCLUDED_BLOCK_TYPES,
    FindBlockTool,
)
from backend.api.features.chat.tools.models import BlockListResponse
from backend.data.block import BlockType

from ._test_data import make_session

_TEST_USER_ID = "test-user-find-block"


def make_mock_block(
    block_id: str, name: str, block_type: BlockType, disabled: bool = False
):
    """Create a mock block for testing."""
    mock = MagicMock()
    mock.id = block_id
    mock.name = name
    mock.description = f"{name} description"
    mock.block_type = block_type
    mock.disabled = disabled
    mock.input_schema = MagicMock()
    mock.input_schema.jsonschema.return_value = {"properties": {}, "required": []}
    mock.input_schema.get_credentials_fields.return_value = {}
    mock.output_schema = MagicMock()
    mock.output_schema.jsonschema.return_value = {}
    mock.categories = []
    return mock


class TestFindBlockFiltering:
    """Tests for block filtering in FindBlockTool."""

    def test_excluded_block_types_contains_expected_types(self):
        """Verify COPILOT_EXCLUDED_BLOCK_TYPES contains all graph-only types."""
        assert BlockType.INPUT in COPILOT_EXCLUDED_BLOCK_TYPES
        assert BlockType.OUTPUT in COPILOT_EXCLUDED_BLOCK_TYPES
        assert BlockType.WEBHOOK in COPILOT_EXCLUDED_BLOCK_TYPES
        assert BlockType.WEBHOOK_MANUAL in COPILOT_EXCLUDED_BLOCK_TYPES
        assert BlockType.NOTE in COPILOT_EXCLUDED_BLOCK_TYPES
        assert BlockType.HUMAN_IN_THE_LOOP in COPILOT_EXCLUDED_BLOCK_TYPES
        assert BlockType.AGENT in COPILOT_EXCLUDED_BLOCK_TYPES

    def test_excluded_block_ids_contains_smart_decision_maker(self):
        """Verify SmartDecisionMakerBlock is in COPILOT_EXCLUDED_BLOCK_IDS."""
        assert "3b191d9f-356f-482d-8238-ba04b6d18381" in COPILOT_EXCLUDED_BLOCK_IDS

    @pytest.mark.asyncio(loop_scope="session")
    async def test_excluded_block_type_filtered_from_results(self):
        """Verify blocks with excluded BlockTypes are filtered from search results."""
        session = make_session(user_id=_TEST_USER_ID)

        # Mock search returns an INPUT block (excluded) and a STANDARD block (included)
        search_results = [
            {"content_id": "input-block-id", "score": 0.9},
            {"content_id": "standard-block-id", "score": 0.8},
        ]

        input_block = make_mock_block("input-block-id", "Input Block", BlockType.INPUT)
        standard_block = make_mock_block(
            "standard-block-id", "HTTP Request", BlockType.STANDARD
        )

        def mock_get_block(block_id):
            return {
                "input-block-id": input_block,
                "standard-block-id": standard_block,
            }.get(block_id)

        with patch(
            "backend.api.features.chat.tools.find_block.unified_hybrid_search",
            new_callable=AsyncMock,
            return_value=(search_results, 2),
        ):
            with patch(
                "backend.api.features.chat.tools.find_block.get_block",
                side_effect=mock_get_block,
            ):
                tool = FindBlockTool()
                response = await tool._execute(
                    user_id=_TEST_USER_ID, session=session, query="test"
                )

        # Should only return the standard block, not the INPUT block
        assert isinstance(response, BlockListResponse)
        assert len(response.blocks) == 1
        assert response.blocks[0].id == "standard-block-id"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_excluded_block_id_filtered_from_results(self):
        """Verify SmartDecisionMakerBlock is filtered from search results."""
        session = make_session(user_id=_TEST_USER_ID)

        smart_decision_id = "3b191d9f-356f-482d-8238-ba04b6d18381"
        search_results = [
            {"content_id": smart_decision_id, "score": 0.9},
            {"content_id": "normal-block-id", "score": 0.8},
        ]

        # SmartDecisionMakerBlock has STANDARD type but is excluded by ID
        smart_block = make_mock_block(
            smart_decision_id, "Smart Decision Maker", BlockType.STANDARD
        )
        normal_block = make_mock_block(
            "normal-block-id", "Normal Block", BlockType.STANDARD
        )

        def mock_get_block(block_id):
            return {
                smart_decision_id: smart_block,
                "normal-block-id": normal_block,
            }.get(block_id)

        with patch(
            "backend.api.features.chat.tools.find_block.unified_hybrid_search",
            new_callable=AsyncMock,
            return_value=(search_results, 2),
        ):
            with patch(
                "backend.api.features.chat.tools.find_block.get_block",
                side_effect=mock_get_block,
            ):
                tool = FindBlockTool()
                response = await tool._execute(
                    user_id=_TEST_USER_ID, session=session, query="decision"
                )

        # Should only return normal block, not SmartDecisionMakerBlock
        assert isinstance(response, BlockListResponse)
        assert len(response.blocks) == 1
        assert response.blocks[0].id == "normal-block-id"
