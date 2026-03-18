"""Tests for block filtering in FindBlockTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockType

from ._test_data import make_session
from .find_block import (
    COPILOT_EXCLUDED_BLOCK_IDS,
    COPILOT_EXCLUDED_BLOCK_TYPES,
    FindBlockTool,
)
from .models import BlockListResponse

_TEST_USER_ID = "test-user-find-block"


def make_mock_block(
    block_id: str,
    name: str,
    block_type: BlockType,
    disabled: bool = False,
    input_schema: dict | None = None,
    output_schema: dict | None = None,
    credentials_fields: dict | None = None,
):
    """Create a mock block for testing."""
    mock = MagicMock()
    mock.id = block_id
    mock.name = name
    mock.description = f"{name} description"
    mock.block_type = block_type
    mock.disabled = disabled
    mock.input_schema = MagicMock()
    mock.input_schema.jsonschema.return_value = input_schema or {
        "properties": {},
        "required": [],
    }
    mock.input_schema.get_credentials_fields.return_value = credentials_fields or {}
    mock.output_schema = MagicMock()
    mock.output_schema.jsonschema.return_value = output_schema or {}
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

        mock_search_db = MagicMock()
        mock_search_db.unified_hybrid_search = AsyncMock(
            return_value=(search_results, 2)
        )

        with patch(
            "backend.copilot.tools.find_block.search",
            return_value=mock_search_db,
        ):
            with patch(
                "backend.copilot.tools.find_block.get_block",
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

        mock_search_db = MagicMock()
        mock_search_db.unified_hybrid_search = AsyncMock(
            return_value=(search_results, 2)
        )

        with patch(
            "backend.copilot.tools.find_block.search",
            return_value=mock_search_db,
        ):
            with patch(
                "backend.copilot.tools.find_block.get_block",
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

    @pytest.mark.asyncio(loop_scope="session")
    async def test_response_size_average_chars_per_block(self):
        """Measure average chars per block in the serialized response."""
        session = make_session(user_id=_TEST_USER_ID)

        # Realistic block definitions modeled after real blocks
        block_defs = [
            {
                "id": "http-block-id",
                "name": "Send Web Request",
                "input_schema": {
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to send the request to",
                        },
                        "method": {
                            "type": "string",
                            "description": "The HTTP method to use",
                        },
                        "headers": {
                            "type": "object",
                            "description": "Headers to include in the request",
                        },
                        "json_format": {
                            "type": "boolean",
                            "description": "If true, send the body as JSON",
                        },
                        "body": {
                            "type": "object",
                            "description": "Form/JSON body payload",
                        },
                        "credentials": {
                            "type": "object",
                            "description": "HTTP credentials",
                        },
                    },
                    "required": ["url", "method"],
                },
                "output_schema": {
                    "properties": {
                        "response": {
                            "type": "object",
                            "description": "The response from the server",
                        },
                        "client_error": {
                            "type": "object",
                            "description": "Errors on 4xx status codes",
                        },
                        "server_error": {
                            "type": "object",
                            "description": "Errors on 5xx status codes",
                        },
                        "error": {
                            "type": "string",
                            "description": "Errors for all other exceptions",
                        },
                    },
                },
                "credentials_fields": {"credentials": True},
            },
            {
                "id": "email-block-id",
                "name": "Send Email",
                "input_schema": {
                    "properties": {
                        "to_email": {
                            "type": "string",
                            "description": "Recipient email address",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Subject of the email",
                        },
                        "body": {
                            "type": "string",
                            "description": "Body of the email",
                        },
                        "config": {
                            "type": "object",
                            "description": "SMTP Config",
                        },
                        "credentials": {
                            "type": "object",
                            "description": "SMTP credentials",
                        },
                    },
                    "required": ["to_email", "subject", "body", "credentials"],
                },
                "output_schema": {
                    "properties": {
                        "status": {
                            "type": "string",
                            "description": "Status of the email sending operation",
                        },
                        "error": {
                            "type": "string",
                            "description": "Error message if sending failed",
                        },
                    },
                },
                "credentials_fields": {"credentials": True},
            },
            {
                "id": "claude-code-block-id",
                "name": "Claude Code",
                "input_schema": {
                    "properties": {
                        "e2b_credentials": {
                            "type": "object",
                            "description": "API key for E2B platform",
                        },
                        "anthropic_credentials": {
                            "type": "object",
                            "description": "API key for Anthropic",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Task or instruction for Claude Code",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Sandbox timeout in seconds",
                        },
                        "setup_commands": {
                            "type": "array",
                            "description": "Shell commands to run before execution",
                        },
                        "working_directory": {
                            "type": "string",
                            "description": "Working directory for Claude Code",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to resume a conversation",
                        },
                        "sandbox_id": {
                            "type": "string",
                            "description": "Sandbox ID to reconnect to",
                        },
                        "conversation_history": {
                            "type": "string",
                            "description": "Previous conversation history",
                        },
                        "dispose_sandbox": {
                            "type": "boolean",
                            "description": "Whether to dispose sandbox after execution",
                        },
                    },
                    "required": [
                        "e2b_credentials",
                        "anthropic_credentials",
                        "prompt",
                    ],
                },
                "output_schema": {
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "Output from Claude Code execution",
                        },
                        "files": {
                            "type": "array",
                            "description": "Files created/modified by Claude Code",
                        },
                        "conversation_history": {
                            "type": "string",
                            "description": "Full conversation history",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID for this conversation",
                        },
                        "sandbox_id": {
                            "type": "string",
                            "description": "ID of the sandbox instance",
                        },
                        "error": {
                            "type": "string",
                            "description": "Error message if execution failed",
                        },
                    },
                },
                "credentials_fields": {
                    "e2b_credentials": True,
                    "anthropic_credentials": True,
                },
            },
        ]

        search_results = [
            {"content_id": d["id"], "score": 0.9 - i * 0.1}
            for i, d in enumerate(block_defs)
        ]
        mock_blocks = {
            d["id"]: make_mock_block(
                block_id=d["id"],
                name=d["name"],
                block_type=BlockType.STANDARD,
                input_schema=d["input_schema"],
                output_schema=d["output_schema"],
                credentials_fields=d["credentials_fields"],
            )
            for d in block_defs
        }

        mock_search_db = MagicMock()
        mock_search_db.unified_hybrid_search = AsyncMock(
            return_value=(search_results, len(search_results))
        )

        with (
            patch(
                "backend.copilot.tools.find_block.search",
                return_value=mock_search_db,
            ),
            patch(
                "backend.copilot.tools.find_block.get_block",
                side_effect=lambda bid: mock_blocks.get(bid),
            ),
        ):
            tool = FindBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID, session=session, query="test"
            )

        assert isinstance(response, BlockListResponse)
        assert response.count == len(block_defs)

        total_chars = len(response.model_dump_json())
        avg_chars = total_chars // response.count

        # Print for visibility in test output
        print(f"\nTotal response size: {total_chars} chars")
        print(f"Number of blocks: {response.count}")
        print(f"Average chars per block: {avg_chars}")

        # The old response was ~90K for 10 blocks (~9K per block).
        # Previous optimization reduced it to ~1.5K per block (no raw JSON schemas).
        # Now with only id/name/description, we expect ~300 chars per block.
        assert avg_chars < 500, (
            f"Average chars per block ({avg_chars}) exceeds 500. "
            f"Total response: {total_chars} chars for {response.count} blocks."
        )
