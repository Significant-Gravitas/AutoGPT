"""Tests for block execution guards and input validation in RunBlockTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockType
from backend.copilot.context import _current_permissions
from backend.copilot.permissions import CopilotPermissions

from ._test_data import make_session
from .models import (
    BlockDetailsResponse,
    BlockOutputResponse,
    ErrorResponse,
    InputValidationErrorResponse,
    ReviewRequiredResponse,
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
    mock.is_sensitive_action = False
    mock.input_schema = MagicMock()
    mock.input_schema.jsonschema.return_value = {"properties": {}, "required": []}
    mock.input_schema.get_credentials_fields_info.return_value = {}
    mock.input_schema.get_credentials_fields.return_value = {}

    async def _no_review(input_data, **kwargs):
        return False, input_data

    mock.is_block_exec_need_review = _no_review
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
    mock.is_sensitive_action = False
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

    # Default: no review needed, pass through input_data unchanged
    async def _no_review(input_data, **kwargs):
        return False, input_data

    mock.is_block_exec_need_review = _no_review

    return mock


class TestRunBlockFiltering:
    """Tests for block execution guards in RunBlockTool."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_excluded_block_type_returns_error(self):
        """Attempting to execute a block with excluded BlockType returns error."""
        session = make_session(user_id=_TEST_USER_ID)

        input_block = make_mock_block("input-block-id", "Input Block", BlockType.INPUT)

        with patch(
            "backend.copilot.tools.helpers.get_block",
            return_value=input_block,
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="input-block-id",
                input_data={},
                dry_run=False,
            )

        assert isinstance(response, ErrorResponse)
        assert "cannot be run directly" in response.message
        assert "designed for use within graphs only" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_excluded_block_id_returns_error(self):
        """Attempting to execute OrchestratorBlock returns error."""
        session = make_session(user_id=_TEST_USER_ID)

        orchestrator_id = "3b191d9f-356f-482d-8238-ba04b6d18381"
        smart_block = make_mock_block(
            orchestrator_id, "Orchestrator", BlockType.STANDARD
        )

        with patch(
            "backend.copilot.tools.helpers.get_block",
            return_value=smart_block,
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id=orchestrator_id,
                input_data={},
                dry_run=False,
            )

        assert isinstance(response, ErrorResponse)
        assert "cannot be run directly" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_block_denied_by_permissions_returns_error(self):
        """A block denied by CopilotPermissions returns an ErrorResponse."""
        session = make_session(user_id=_TEST_USER_ID)
        # NB: must not match any id in COPILOT_EXCLUDED_BLOCK_IDS — we want
        # the permissions guard to fire, not the exclusion guard.
        block_id = "11111111-2222-3333-4444-555555555555"
        standard_block = make_mock_block(block_id, "HTTP Request", BlockType.STANDARD)

        perms = CopilotPermissions(blocks=[block_id], blocks_exclude=True)
        token = _current_permissions.set(perms)
        try:
            with patch(
                "backend.copilot.tools.helpers.get_block",
                return_value=standard_block,
            ):
                tool = RunBlockTool()
                response = await tool._execute(
                    user_id=_TEST_USER_ID,
                    session=session,
                    block_id=block_id,
                    input_data={},
                    dry_run=False,
                )
        finally:
            _current_permissions.reset(token)

        assert isinstance(response, ErrorResponse)
        assert "not permitted" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_allowed_by_permissions_passes_guard(self):
        """A block explicitly allowed by a whitelist CopilotPermissions passes the guard."""
        session = make_session(user_id=_TEST_USER_ID)
        block_id = "c069dc6b-c3ed-4c12-b6e5-d47361e64ce6"
        standard_block = make_mock_block(block_id, "HTTP Request", BlockType.STANDARD)

        perms = CopilotPermissions(blocks=[block_id], blocks_exclude=False)
        token = _current_permissions.set(perms)
        try:
            with (
                patch(
                    "backend.copilot.tools.helpers.get_block",
                    return_value=standard_block,
                ),
                patch(
                    "backend.copilot.tools.helpers.match_credentials_to_requirements",
                    return_value=({}, []),
                ),
            ):
                tool = RunBlockTool()
                response = await tool._execute(
                    user_id=_TEST_USER_ID,
                    session=session,
                    block_id=block_id,
                    input_data={},
                    dry_run=False,
                )
        finally:
            _current_permissions.reset(token)

        # Must NOT be blocked by permissions — assert it's not a permission error
        assert (
            not isinstance(response, ErrorResponse)
            or "not permitted" not in response.message
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_non_excluded_block_passes_guard(self):
        """Non-excluded blocks pass the filtering guard (may fail later for other reasons)."""
        session = make_session(user_id=_TEST_USER_ID)

        standard_block = make_mock_block(
            "standard-id", "HTTP Request", BlockType.STANDARD
        )

        with (
            patch(
                "backend.copilot.tools.helpers.get_block",
                return_value=standard_block,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="standard-id",
                input_data={},
                dry_run=False,
            )

        # Should NOT be an ErrorResponse about CoPilot exclusion
        # (may be other errors like missing credentials, but not the exclusion guard)
        if isinstance(response, ErrorResponse):
            assert "cannot be run directly" not in response.message


class TestRunBlockInputValidation:
    """Tests for input field validation in RunBlockTool.

    run_block rejects unknown input field names with InputValidationErrorResponse,
    preventing silent failures where incorrect keys would be ignored and the block
    would execute with default values instead of the caller's intended values.
    """

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unknown_input_fields_are_rejected(self):
        """run_block rejects unknown input fields instead of silently ignoring them."""
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

        with (
            patch(
                "backend.copilot.tools.helpers.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="ai-text-gen-id",
                input_data={
                    "prompt": "Write a haiku about coding",
                    "LLM_Model": "claude-opus-4-6",
                },
                dry_run=False,
            )

        assert isinstance(response, InputValidationErrorResponse)
        assert "LLM_Model" in response.unrecognized_fields
        assert "Block was not executed" in response.message
        assert "inputs" in response.model_dump()

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

        with (
            patch(
                "backend.copilot.tools.helpers.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="ai-text-gen-id",
                input_data={
                    "prompt": "Hello",
                    "llm_model": "claude-opus-4-6",
                    "system_prompt": "Be helpful",
                    "retries": 5,
                },
                dry_run=False,
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

        with (
            patch(
                "backend.copilot.tools.helpers.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="ai-text-gen-id",
                input_data={
                    "LLM_Model": "claude-opus-4-6",
                },
                dry_run=False,
            )

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
                "backend.copilot.tools.helpers.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
            patch(
                "backend.copilot.tools.helpers.workspace_db",
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
                    "model": "gpt-4o-mini",
                },
                dry_run=False,
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

        with (
            patch(
                "backend.copilot.tools.helpers.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
        ):
            tool = RunBlockTool()

            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="ai-text-gen-id",
                input_data={
                    "model": "gpt-4o-mini",
                },
                dry_run=False,
            )

        assert isinstance(response, BlockDetailsResponse)


class TestRunBlockSensitiveAction:
    """Tests for sensitive action HITL review in RunBlockTool.

    run_block calls is_block_exec_need_review() explicitly before execution.
    When review is needed (should_pause=True), ReviewRequiredResponse is returned.
    """

    @pytest.mark.asyncio(loop_scope="session")
    async def test_sensitive_block_paused_returns_review_required(self):
        """When is_block_exec_need_review returns should_pause=True, ReviewRequiredResponse is returned."""
        session = make_session(user_id=_TEST_USER_ID)

        input_data = {
            "repo_url": "https://github.com/test/repo",
            "branch": "feature-branch",
        }
        mock_block = make_mock_block_with_schema(
            block_id="delete-branch-id",
            name="Delete Branch",
            input_properties={
                "repo_url": {"type": "string"},
                "branch": {"type": "string"},
            },
            required_fields=["repo_url", "branch"],
        )
        mock_block.is_sensitive_action = True
        mock_block.is_block_exec_need_review = AsyncMock(
            return_value=(True, input_data)
        )

        with (
            patch(
                "backend.copilot.tools.helpers.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="delete-branch-id",
                input_data=input_data,
                dry_run=False,
            )

        assert isinstance(response, ReviewRequiredResponse)
        assert "requires human review" in response.message
        assert "continue_run_block" in response.message
        assert response.block_name == "Delete Branch"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_sensitive_block_executes_after_approval(self):
        """After approval (should_pause=False), sensitive blocks execute and return outputs."""
        session = make_session(user_id=_TEST_USER_ID)

        input_data = {
            "repo_url": "https://github.com/test/repo",
            "branch": "feature-branch",
        }
        mock_block = make_mock_block_with_schema(
            block_id="delete-branch-id",
            name="Delete Branch",
            input_properties={
                "repo_url": {"type": "string"},
                "branch": {"type": "string"},
            },
            required_fields=["repo_url", "branch"],
        )
        mock_block.is_sensitive_action = True
        mock_block.is_block_exec_need_review = AsyncMock(
            return_value=(False, input_data)
        )

        async def mock_execute(input_data, **kwargs):
            yield "result", "Branch deleted successfully"

        mock_block.execute = mock_execute

        mock_workspace_db = MagicMock()
        mock_workspace_db.get_or_create_workspace = AsyncMock(
            return_value=MagicMock(id="test-workspace-id")
        )

        with (
            patch(
                "backend.copilot.tools.helpers.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
            patch(
                "backend.copilot.tools.helpers.workspace_db",
                return_value=mock_workspace_db,
            ),
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="delete-branch-id",
                input_data=input_data,
                dry_run=False,
            )

        assert isinstance(response, BlockOutputResponse)
        assert response.success is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_non_sensitive_block_executes_normally(self):
        """Non-sensitive blocks skip review and execute directly."""
        session = make_session(user_id=_TEST_USER_ID)

        input_data = {"url": "https://example.com"}
        mock_block = make_mock_block_with_schema(
            block_id="http-request-id",
            name="HTTP Request",
            input_properties={
                "url": {"type": "string"},
            },
            required_fields=["url"],
        )
        mock_block.is_sensitive_action = False
        mock_block.is_block_exec_need_review = AsyncMock(
            return_value=(False, input_data)
        )

        async def mock_execute(input_data, **kwargs):
            yield "response", {"status": 200}

        mock_block.execute = mock_execute

        mock_workspace_db = MagicMock()
        mock_workspace_db.get_or_create_workspace = AsyncMock(
            return_value=MagicMock(id="test-workspace-id")
        )

        with (
            patch(
                "backend.copilot.tools.helpers.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
            patch(
                "backend.copilot.tools.helpers.workspace_db",
                return_value=mock_workspace_db,
            ),
        ):
            tool = RunBlockTool()
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                block_id="http-request-id",
                input_data=input_data,
                dry_run=False,
            )

        assert isinstance(response, BlockOutputResponse)
        assert response.success is True


class TestExecuteBlockTimeout:
    """``execute_block`` caps the block's generator consumption at
    MAX_TOOL_WAIT_SECONDS and must:
      1. Return an actionable ErrorResponse pointing at run_agent / run_sub_session.
      2. Log a ``copilot_tool_timeout`` warning (SECRT-2247 part 3).
      3. Still charge credits when outputs were produced before the timeout
         (sentry r3105079148 — cancellation must not leak billing)."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_timeout_returns_error_and_logs(self, caplog):
        import asyncio
        import logging

        from backend.copilot.tools.helpers import execute_block

        mock_block = MagicMock()
        mock_block.name = "SlowBlock"
        mock_block.id = "slow-block-id"
        mock_block.input_schema = MagicMock()
        mock_block.input_schema.jsonschema.return_value = {
            "properties": {},
            "required": [],
        }
        mock_block.input_schema.get_credentials_fields.return_value = {}

        async def _hang(_input, **_kwargs):
            await asyncio.sleep(10)
            yield "never", "never"

        mock_block.execute = _hang

        mock_workspace_db = MagicMock()
        mock_workspace_db.get_or_create_workspace = AsyncMock(
            return_value=MagicMock(id="ws-1")
        )

        with (
            patch(
                "backend.copilot.tools.helpers.workspace_db",
                return_value=mock_workspace_db,
            ),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(0, {}),
            ),
            patch(
                "backend.copilot.tools.helpers.MAX_TOOL_WAIT_SECONDS",
                0.05,
            ),
            caplog.at_level(logging.WARNING, logger="backend.copilot.tools.helpers"),
        ):
            response = await execute_block(
                block=mock_block,
                block_id="slow-block-id",
                input_data={"x": 1},
                user_id="u-1",
                session_id="s-1",
                node_exec_id="n-1",
                matched_credentials={},
                dry_run=False,
            )

        assert isinstance(response, ErrorResponse)
        assert "single-tool wait cap" in response.message
        assert "run_agent" in response.message
        assert any(
            "copilot_tool_timeout" in record.getMessage() for record in caplog.records
        ), "timeout must emit a grep-friendly log line for SECRT-2247 part 3"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_cancellation_after_output_still_charges_credits(self):
        """Regression for sentry r3105079148 — wait_for's CancelledError
        bypassed credit charging; fix uses a shielded finally. One output
        emitted, then timeout: spend_credits must still be called once."""
        import asyncio

        from backend.copilot.tools.helpers import execute_block

        mock_block = MagicMock()
        mock_block.name = "CostlyBlock"
        mock_block.id = "costly-block-id"
        mock_block.input_schema = MagicMock()
        mock_block.input_schema.jsonschema.return_value = {
            "properties": {},
            "required": [],
        }
        mock_block.input_schema.get_credentials_fields.return_value = {}

        # Generator: emit ONE output (simulating a side-effectful API call),
        # then hang — execute_block's internal wait_for cancels us.
        async def _one_output_then_hang(_input, **_kw):
            yield "result", "side effect happened"
            await asyncio.sleep(10)
            yield "extra", "should never arrive"

        mock_block.execute = _one_output_then_hang

        charged: dict[str, object] = {}

        class _FakeCreditDB:
            async def get_credits(self, _user_id: str) -> int:
                return 10_000

            async def spend_credits(self, **kwargs):
                charged["last"] = kwargs

        mock_workspace_db = MagicMock()
        mock_workspace_db.get_or_create_workspace = AsyncMock(
            return_value=MagicMock(id="ws-1")
        )

        with (
            patch(
                "backend.copilot.tools.helpers.workspace_db",
                return_value=mock_workspace_db,
            ),
            patch(
                "backend.copilot.tools.helpers.credit_db",
                return_value=_FakeCreditDB(),
            ),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(5, {}),
            ),
            patch(
                "backend.copilot.tools.helpers.MAX_TOOL_WAIT_SECONDS",
                0.2,
            ),
        ):
            response = await execute_block(
                block=mock_block,
                block_id="costly-block-id",
                input_data={},
                user_id="u-42",
                session_id="s-42",
                node_exec_id="n-42",
                matched_credentials={},
                dry_run=False,
            )

        # Cap fired → response is the timeout ErrorResponse
        assert isinstance(response, ErrorResponse)
        assert "single-tool wait cap" in response.message

        # Critical: billing ran via the shielded finally despite the cancellation
        assert charged.get("last") is not None, (
            "Credits were NOT charged after cancellation — billing leak "
            "(sentry r3105079148)"
        )
        assert charged["last"]["user_id"] == "u-42"
        assert charged["last"]["cost"] == 5

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_double_charge_on_cancellation_during_charge(self):
        """Regression for sentry r3105216985 — if the caller cancels during
        the normal-path credit charge, the finally must NOT charge a second
        time. The fix marks charge_handled BEFORE awaiting spend_credits."""
        import asyncio

        from backend.copilot.tools.helpers import execute_block

        mock_block = MagicMock()
        mock_block.name = "OnceOnlyBlock"
        mock_block.id = "once-only-id"
        mock_block.input_schema = MagicMock()
        mock_block.input_schema.jsonschema.return_value = {
            "properties": {},
            "required": [],
        }
        mock_block.input_schema.get_credentials_fields.return_value = {}

        async def _one_then_done(_input, **_kw):
            yield "result", "done"

        mock_block.execute = _one_then_done

        spend_calls: list[dict] = []

        class _CountingCreditDB:
            async def get_credits(self, _user_id: str) -> int:
                return 10_000

            async def spend_credits(self, **kwargs):
                # Cooperative suspension so an outer cancellation can
                # theoretically interleave — shield should still make this
                # complete exactly once.
                await asyncio.sleep(0)
                spend_calls.append(kwargs)

        mock_workspace_db = MagicMock()
        mock_workspace_db.get_or_create_workspace = AsyncMock(
            return_value=MagicMock(id="ws-1")
        )

        with (
            patch(
                "backend.copilot.tools.helpers.workspace_db",
                return_value=mock_workspace_db,
            ),
            patch(
                "backend.copilot.tools.helpers.credit_db",
                return_value=_CountingCreditDB(),
            ),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(7, {}),
            ),
        ):
            response = await execute_block(
                block=mock_block,
                block_id="once-only-id",
                input_data={},
                user_id="u-single",
                session_id="s-single",
                node_exec_id="n-single",
                matched_credentials={},
                dry_run=False,
            )

        assert isinstance(response, BlockOutputResponse)
        assert response.success is True
        assert len(spend_calls) == 1, (
            f"spend_credits must be called exactly once, got {len(spend_calls)} "
            "(double-charge — sentry r3105216985)"
        )
