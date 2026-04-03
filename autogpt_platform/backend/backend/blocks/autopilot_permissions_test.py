"""Tests for AutoPilotBlock permission fields and validation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.blocks.autopilot import (
    AutoPilotBlock,
    _build_and_validate_permissions,
    _inherited_permissions,
    _merge_inherited_permissions,
)
from backend.copilot.permissions import CopilotPermissions, all_known_tool_names
from backend.data.execution import ExecutionContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(**kwargs) -> AutoPilotBlock.Input:
    defaults = {
        "prompt": "Do something",
        "system_context": "",
        "session_id": "",
        "max_recursion_depth": 3,
        "tools": [],
        "tools_exclude": True,
        "blocks": [],
        "blocks_exclude": True,
    }
    defaults.update(kwargs)
    return AutoPilotBlock.Input(**defaults)


# ---------------------------------------------------------------------------
# _build_and_validate_permissions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBuildAndValidatePermissions:
    async def test_empty_inputs_returns_empty_permissions(self):
        inp = _make_input()
        result = await _build_and_validate_permissions(inp)
        assert isinstance(result, CopilotPermissions)
        assert result.is_empty()

    async def test_valid_tool_names_accepted(self):
        inp = _make_input(tools=["run_block", "web_fetch"], tools_exclude=True)
        result = await _build_and_validate_permissions(inp)
        assert isinstance(result, CopilotPermissions)
        assert result.tools == ["run_block", "web_fetch"]
        assert result.tools_exclude is True

    async def test_invalid_tool_rejected_by_pydantic(self):
        """Invalid tool names are now caught at Pydantic validation time
        (Literal type), before ``_build_and_validate_permissions`` is called."""
        with pytest.raises(ValidationError, match="not_a_real_tool"):
            _make_input(tools=["not_a_real_tool"])

    async def test_valid_block_name_accepted(self):
        mock_block_cls = MagicMock()
        mock_block_cls.return_value.name = "HTTP Request"
        with patch(
            "backend.blocks.get_blocks",
            return_value={"c069dc6b-c3ed-4c12-b6e5-d47361e64ce6": mock_block_cls},
        ):
            inp = _make_input(blocks=["HTTP Request"], blocks_exclude=True)
            result = await _build_and_validate_permissions(inp)
        assert isinstance(result, CopilotPermissions)
        assert result.blocks == ["HTTP Request"]

    async def test_valid_partial_uuid_accepted(self):
        mock_block_cls = MagicMock()
        mock_block_cls.return_value.name = "HTTP Request"
        with patch(
            "backend.blocks.get_blocks",
            return_value={"c069dc6b-c3ed-4c12-b6e5-d47361e64ce6": mock_block_cls},
        ):
            inp = _make_input(blocks=["c069dc6b"], blocks_exclude=False)
            result = await _build_and_validate_permissions(inp)
        assert isinstance(result, CopilotPermissions)

    async def test_invalid_block_identifier_returns_error(self):
        mock_block_cls = MagicMock()
        mock_block_cls.return_value.name = "HTTP Request"
        with patch(
            "backend.blocks.get_blocks",
            return_value={"c069dc6b-c3ed-4c12-b6e5-d47361e64ce6": mock_block_cls},
        ):
            inp = _make_input(blocks=["totally_fake_block"])
            result = await _build_and_validate_permissions(inp)
        assert isinstance(result, str)
        assert "totally_fake_block" in result
        assert "Unknown block identifier" in result

    async def test_sdk_builtin_tool_names_accepted(self):
        inp = _make_input(tools=["Read", "Task", "WebSearch"], tools_exclude=False)
        result = await _build_and_validate_permissions(inp)
        assert isinstance(result, CopilotPermissions)
        assert not result.tools_exclude

    async def test_empty_blocks_skips_validation(self):
        # Should not call validate_block_identifiers at all when blocks=[].
        with patch(
            "backend.copilot.permissions.validate_block_identifiers"
        ) as mock_validate:
            inp = _make_input(blocks=[])
            await _build_and_validate_permissions(inp)
            mock_validate.assert_not_called()


# ---------------------------------------------------------------------------
# _merge_inherited_permissions
# ---------------------------------------------------------------------------


class TestMergeInheritedPermissions:
    def test_no_permissions_no_parent_returns_none(self):
        merged, token = _merge_inherited_permissions(None)
        assert merged is None
        assert token is None

    def test_permissions_no_parent_returned_unchanged(self):
        perms = CopilotPermissions(tools=["bash_exec"], tools_exclude=True)
        merged, token = _merge_inherited_permissions(perms)
        try:
            assert merged is perms
            assert token is not None
        finally:
            if token is not None:
                _inherited_permissions.reset(token)

    def test_child_narrows_parent(self):
        parent = CopilotPermissions(tools=["bash_exec"], tools_exclude=True)
        # Set parent as inherited
        outer_token = _inherited_permissions.set(parent)
        try:
            child = CopilotPermissions(tools=["web_fetch"], tools_exclude=True)
            merged, inner_token = _merge_inherited_permissions(child)
            try:
                assert merged is not None
                all_t = all_known_tool_names()
                effective = merged.effective_allowed_tools(all_t)
                assert "bash_exec" not in effective
                assert "web_fetch" not in effective
            finally:
                if inner_token is not None:
                    _inherited_permissions.reset(inner_token)
        finally:
            _inherited_permissions.reset(outer_token)

    def test_none_permissions_with_parent_uses_parent(self):
        parent = CopilotPermissions(tools=["bash_exec"], tools_exclude=True)
        outer_token = _inherited_permissions.set(parent)
        try:
            merged, inner_token = _merge_inherited_permissions(None)
            try:
                assert merged is not None
                # Merged should have parent's restrictions
                effective = merged.effective_allowed_tools(all_known_tool_names())
                assert "bash_exec" not in effective
            finally:
                if inner_token is not None:
                    _inherited_permissions.reset(inner_token)
        finally:
            _inherited_permissions.reset(outer_token)

    def test_child_cannot_expand_parent_whitelist(self):
        parent = CopilotPermissions(tools=["run_block"], tools_exclude=False)
        outer_token = _inherited_permissions.set(parent)
        try:
            # Child tries to allow more tools
            child = CopilotPermissions(
                tools=["run_block", "bash_exec"], tools_exclude=False
            )
            merged, inner_token = _merge_inherited_permissions(child)
            try:
                assert merged is not None
                effective = merged.effective_allowed_tools(all_known_tool_names())
                assert "bash_exec" not in effective
                assert "run_block" in effective
            finally:
                if inner_token is not None:
                    _inherited_permissions.reset(inner_token)
        finally:
            _inherited_permissions.reset(outer_token)


# ---------------------------------------------------------------------------
# AutoPilotBlock.run — validation integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAutoPilotBlockRunPermissions:
    async def _collect_outputs(self, block, input_data, user_id="test-user"):
        """Helper to collect all yields from block.run()."""
        ctx = ExecutionContext(
            user_id=user_id,
            graph_id="g1",
            graph_exec_id="ge1",
            node_exec_id="ne1",
            node_id="n1",
        )
        outputs = {}
        async for key, val in block.run(input_data, execution_context=ctx):
            outputs[key] = val
        return outputs

    async def test_invalid_tool_rejected_by_pydantic(self):
        """Invalid tool names are caught at Pydantic validation (Literal type)."""
        with pytest.raises(ValidationError, match="not_a_tool"):
            _make_input(tools=["not_a_tool"])

    async def test_invalid_block_yields_error(self):
        mock_block_cls = MagicMock()
        mock_block_cls.return_value.name = "HTTP Request"
        with patch(
            "backend.blocks.get_blocks",
            return_value={"c069dc6b-c3ed-4c12-b6e5-d47361e64ce6": mock_block_cls},
        ):
            block = AutoPilotBlock()
            inp = _make_input(blocks=["nonexistent_block"])
            outputs = await self._collect_outputs(block, inp)
        assert "error" in outputs
        assert "nonexistent_block" in outputs["error"]

    async def test_empty_prompt_yields_error_before_permission_check(self):
        block = AutoPilotBlock()
        inp = _make_input(prompt="   ", tools=["run_block"])
        outputs = await self._collect_outputs(block, inp)
        assert "error" in outputs
        assert "Prompt cannot be empty" in outputs["error"]

    async def test_valid_permissions_passed_to_execute(self):
        """Permissions are forwarded to execute_copilot when valid."""
        block = AutoPilotBlock()
        captured: dict = {}

        async def fake_execute_copilot(self_inner, **kwargs):
            captured["permissions"] = kwargs.get("permissions")
            return (
                "ok",
                [],
                '[{"role":"user","content":"hi"}]',
                "test-sid",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        with patch.object(
            AutoPilotBlock, "create_session", new=AsyncMock(return_value="test-sid")
        ), patch.object(AutoPilotBlock, "execute_copilot", new=fake_execute_copilot):
            inp = _make_input(tools=["run_block"], tools_exclude=False)
            outputs = await self._collect_outputs(block, inp)

        assert "error" not in outputs
        perms = captured.get("permissions")
        assert isinstance(perms, CopilotPermissions)
        assert perms.tools == ["run_block"]
        assert perms.tools_exclude is False
