"""Tests for CopilotPermissions — tool/block capability filtering."""

from __future__ import annotations

import pytest

from backend.copilot.permissions import (
    ALL_TOOL_NAMES,
    PLATFORM_TOOL_NAMES,
    SDK_BUILTIN_TOOL_NAMES,
    CopilotPermissions,
    _block_matches,
    all_known_tool_names,
    apply_tool_permissions,
    validate_block_identifiers,
    validate_tool_names,
)
from backend.copilot.tools import TOOL_REGISTRY

# ---------------------------------------------------------------------------
# _block_matches
# ---------------------------------------------------------------------------


class TestBlockMatches:
    BLOCK_ID = "c069dc6b-c3ed-4c12-b6e5-d47361e64ce6"
    BLOCK_NAME = "HTTP Request"

    def test_full_uuid_match(self):
        assert _block_matches(self.BLOCK_ID, self.BLOCK_ID, self.BLOCK_NAME)

    def test_full_uuid_case_insensitive(self):
        assert _block_matches(self.BLOCK_ID.upper(), self.BLOCK_ID, self.BLOCK_NAME)

    def test_full_uuid_no_match(self):
        other = "aaaaaaaa-0000-0000-0000-000000000000"
        assert not _block_matches(other, self.BLOCK_ID, self.BLOCK_NAME)

    def test_partial_uuid_match(self):
        assert _block_matches("c069dc6b", self.BLOCK_ID, self.BLOCK_NAME)

    def test_partial_uuid_case_insensitive(self):
        assert _block_matches("C069DC6B", self.BLOCK_ID, self.BLOCK_NAME)

    def test_partial_uuid_no_match(self):
        assert not _block_matches("deadbeef", self.BLOCK_ID, self.BLOCK_NAME)

    def test_name_match(self):
        assert _block_matches("HTTP Request", self.BLOCK_ID, self.BLOCK_NAME)

    def test_name_case_insensitive(self):
        assert _block_matches("http request", self.BLOCK_ID, self.BLOCK_NAME)
        assert _block_matches("HTTP REQUEST", self.BLOCK_ID, self.BLOCK_NAME)

    def test_name_no_match(self):
        assert not _block_matches("Unknown Block", self.BLOCK_ID, self.BLOCK_NAME)

    def test_partial_uuid_not_matching_as_name(self):
        # "c069dc6b" is 8 hex chars → treated as partial UUID, NOT name match
        assert not _block_matches(
            "c069dc6b", "ffffffff-0000-0000-0000-000000000000", "c069dc6b"
        )


# ---------------------------------------------------------------------------
# CopilotPermissions.effective_allowed_tools
# ---------------------------------------------------------------------------


ALL_TOOLS = frozenset(
    ["run_block", "web_fetch", "bash_exec", "find_agent", "Task", "Read"]
)


class TestEffectiveAllowedTools:
    def test_empty_list_allows_all(self):
        perms = CopilotPermissions(tools=[], tools_exclude=True)
        assert perms.effective_allowed_tools(ALL_TOOLS) == ALL_TOOLS

    def test_empty_whitelist_allows_all(self):
        # edge: tools_exclude=False but empty list → allow all
        perms = CopilotPermissions(tools=[], tools_exclude=False)
        assert perms.effective_allowed_tools(ALL_TOOLS) == ALL_TOOLS

    def test_blacklist_removes_listed(self):
        perms = CopilotPermissions(tools=["bash_exec", "web_fetch"], tools_exclude=True)
        result = perms.effective_allowed_tools(ALL_TOOLS)
        assert "bash_exec" not in result
        assert "web_fetch" not in result
        assert "run_block" in result
        assert "Task" in result

    def test_whitelist_keeps_only_listed(self):
        perms = CopilotPermissions(tools=["run_block", "Task"], tools_exclude=False)
        result = perms.effective_allowed_tools(ALL_TOOLS)
        assert result == frozenset(["run_block", "Task"])

    def test_whitelist_unknown_tool_yields_empty(self):
        perms = CopilotPermissions(tools=["nonexistent"], tools_exclude=False)
        result = perms.effective_allowed_tools(ALL_TOOLS)
        assert result == frozenset()

    def test_blacklist_unknown_tool_ignored(self):
        perms = CopilotPermissions(tools=["nonexistent"], tools_exclude=True)
        result = perms.effective_allowed_tools(ALL_TOOLS)
        assert result == ALL_TOOLS


# ---------------------------------------------------------------------------
# CopilotPermissions.is_block_allowed
# ---------------------------------------------------------------------------


BLOCK_ID = "c069dc6b-c3ed-4c12-b6e5-d47361e64ce6"
BLOCK_NAME = "HTTP Request"


class TestIsBlockAllowed:
    def test_empty_allows_everything(self):
        perms = CopilotPermissions(blocks=[], blocks_exclude=True)
        assert perms.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_blacklist_blocks_listed(self):
        perms = CopilotPermissions(blocks=["HTTP Request"], blocks_exclude=True)
        assert not perms.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_blacklist_allows_unlisted(self):
        perms = CopilotPermissions(blocks=["Other Block"], blocks_exclude=True)
        assert perms.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_whitelist_allows_listed(self):
        perms = CopilotPermissions(blocks=["HTTP Request"], blocks_exclude=False)
        assert perms.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_whitelist_blocks_unlisted(self):
        perms = CopilotPermissions(blocks=["Other Block"], blocks_exclude=False)
        assert not perms.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_partial_uuid_blacklist(self):
        perms = CopilotPermissions(blocks=["c069dc6b"], blocks_exclude=True)
        assert not perms.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_full_uuid_whitelist(self):
        perms = CopilotPermissions(blocks=[BLOCK_ID], blocks_exclude=False)
        assert perms.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_parent_blocks_when_child_allows(self):
        parent = CopilotPermissions(blocks=["HTTP Request"], blocks_exclude=True)
        child = CopilotPermissions(blocks=[], blocks_exclude=True)
        child._parent = parent
        assert not child.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_parent_allows_when_child_blocks(self):
        parent = CopilotPermissions(blocks=[], blocks_exclude=True)
        child = CopilotPermissions(blocks=["HTTP Request"], blocks_exclude=True)
        child._parent = parent
        assert not child.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_both_must_allow(self):
        parent = CopilotPermissions(blocks=["HTTP Request"], blocks_exclude=False)
        child = CopilotPermissions(blocks=["HTTP Request"], blocks_exclude=False)
        child._parent = parent
        assert child.is_block_allowed(BLOCK_ID, BLOCK_NAME)

    def test_grandparent_blocks_propagate(self):
        grandparent = CopilotPermissions(blocks=["HTTP Request"], blocks_exclude=True)
        parent = CopilotPermissions(blocks=[], blocks_exclude=True)
        parent._parent = grandparent
        child = CopilotPermissions(blocks=[], blocks_exclude=True)
        child._parent = parent
        assert not child.is_block_allowed(BLOCK_ID, BLOCK_NAME)


# ---------------------------------------------------------------------------
# CopilotPermissions.merged_with_parent
# ---------------------------------------------------------------------------


class TestMergedWithParent:
    def test_tool_intersection(self):
        all_t = frozenset(["run_block", "web_fetch", "bash_exec"])
        parent = CopilotPermissions(tools=["bash_exec"], tools_exclude=True)
        child = CopilotPermissions(tools=["web_fetch"], tools_exclude=True)
        merged = child.merged_with_parent(parent, all_t)
        effective = merged.effective_allowed_tools(all_t)
        assert "bash_exec" not in effective
        assert "web_fetch" not in effective
        assert "run_block" in effective

    def test_parent_whitelist_narrows_child(self):
        all_t = frozenset(["run_block", "web_fetch", "bash_exec"])
        parent = CopilotPermissions(tools=["run_block"], tools_exclude=False)
        child = CopilotPermissions(tools=[], tools_exclude=True)  # allow all
        merged = child.merged_with_parent(parent, all_t)
        effective = merged.effective_allowed_tools(all_t)
        assert effective == frozenset(["run_block"])

    def test_child_cannot_expand_parent_whitelist(self):
        all_t = frozenset(["run_block", "web_fetch", "bash_exec"])
        parent = CopilotPermissions(tools=["run_block"], tools_exclude=False)
        child = CopilotPermissions(
            tools=["run_block", "bash_exec"], tools_exclude=False
        )
        merged = child.merged_with_parent(parent, all_t)
        effective = merged.effective_allowed_tools(all_t)
        # bash_exec was not in parent's whitelist → must not appear
        assert "bash_exec" not in effective
        assert "run_block" in effective

    def test_merged_stored_as_whitelist(self):
        all_t = frozenset(["run_block", "web_fetch"])
        parent = CopilotPermissions(tools=[], tools_exclude=True)
        child = CopilotPermissions(tools=[], tools_exclude=True)
        merged = child.merged_with_parent(parent, all_t)
        assert not merged.tools_exclude  # stored as whitelist
        assert set(merged.tools) == {"run_block", "web_fetch"}

    def test_block_parent_stored(self):
        all_t = frozenset(["run_block"])
        parent = CopilotPermissions(blocks=["HTTP Request"], blocks_exclude=True)
        child = CopilotPermissions(blocks=[], blocks_exclude=True)
        merged = child.merged_with_parent(parent, all_t)
        # Parent restriction is preserved via _parent
        assert not merged.is_block_allowed(BLOCK_ID, BLOCK_NAME)


# ---------------------------------------------------------------------------
# CopilotPermissions.is_empty
# ---------------------------------------------------------------------------


class TestIsEmpty:
    def test_default_is_empty(self):
        assert CopilotPermissions().is_empty()

    def test_with_tools_not_empty(self):
        assert not CopilotPermissions(tools=["bash_exec"]).is_empty()

    def test_with_blocks_not_empty(self):
        assert not CopilotPermissions(blocks=["HTTP Request"]).is_empty()

    def test_with_parent_not_empty(self):
        perms = CopilotPermissions()
        perms._parent = CopilotPermissions(tools=["bash_exec"])
        assert not perms.is_empty()


# ---------------------------------------------------------------------------
# validate_tool_names
# ---------------------------------------------------------------------------


class TestValidateToolNames:
    def test_valid_registry_tool(self):
        assert validate_tool_names(["run_block", "web_fetch"]) == []

    def test_valid_sdk_builtin(self):
        assert validate_tool_names(["Read", "Task", "WebSearch"]) == []

    def test_invalid_tool(self):
        result = validate_tool_names(["nonexistent_tool"])
        assert "nonexistent_tool" in result

    def test_mixed(self):
        result = validate_tool_names(["run_block", "fake_tool"])
        assert "fake_tool" in result
        assert "run_block" not in result

    def test_empty_list(self):
        assert validate_tool_names([]) == []


# ---------------------------------------------------------------------------
# validate_block_identifiers (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestValidateBlockIdentifiers:
    async def test_empty_list(self):
        result = await validate_block_identifiers([])
        assert result == []

    async def test_valid_full_uuid(self, mocker):
        mock_block = mocker.MagicMock()
        mock_block.return_value.name = "HTTP Request"
        mocker.patch(
            "backend.blocks.get_blocks",
            return_value={"c069dc6b-c3ed-4c12-b6e5-d47361e64ce6": mock_block},
        )
        result = await validate_block_identifiers(
            ["c069dc6b-c3ed-4c12-b6e5-d47361e64ce6"]
        )
        assert result == []

    async def test_invalid_identifier(self, mocker):
        mock_block = mocker.MagicMock()
        mock_block.return_value.name = "HTTP Request"
        mocker.patch(
            "backend.blocks.get_blocks",
            return_value={"c069dc6b-c3ed-4c12-b6e5-d47361e64ce6": mock_block},
        )
        result = await validate_block_identifiers(["totally_unknown"])
        assert "totally_unknown" in result

    async def test_partial_uuid_match(self, mocker):
        mock_block = mocker.MagicMock()
        mock_block.return_value.name = "HTTP Request"
        mocker.patch(
            "backend.blocks.get_blocks",
            return_value={"c069dc6b-c3ed-4c12-b6e5-d47361e64ce6": mock_block},
        )
        result = await validate_block_identifiers(["c069dc6b"])
        assert result == []

    async def test_name_match(self, mocker):
        mock_block = mocker.MagicMock()
        mock_block.return_value.name = "HTTP Request"
        mocker.patch(
            "backend.blocks.get_blocks",
            return_value={"c069dc6b-c3ed-4c12-b6e5-d47361e64ce6": mock_block},
        )
        result = await validate_block_identifiers(["http request"])
        assert result == []


# ---------------------------------------------------------------------------
# apply_tool_permissions
# ---------------------------------------------------------------------------


class TestApplyToolPermissions:
    def test_empty_permissions_returns_base_unchanged(self, mocker):
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_copilot_tool_names",
            return_value=["mcp__copilot__run_block", "mcp__copilot__web_fetch", "Task"],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_sdk_disallowed_tools",
            return_value=["Bash"],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": object(), "web_fetch": object()},
        )
        perms = CopilotPermissions()
        allowed, disallowed = apply_tool_permissions(perms, use_e2b=False)
        assert "mcp__copilot__run_block" in allowed
        assert "mcp__copilot__web_fetch" in allowed

    def test_blacklist_removes_tool(self, mocker):
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_copilot_tool_names",
            return_value=[
                "mcp__copilot__run_block",
                "mcp__copilot__web_fetch",
                "mcp__copilot__bash_exec",
                "Task",
            ],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_sdk_disallowed_tools",
            return_value=["Bash"],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {
                "run_block": object(),
                "web_fetch": object(),
                "bash_exec": object(),
            },
        )
        mocker.patch(
            "backend.copilot.permissions.all_known_tool_names",
            return_value=frozenset(["run_block", "web_fetch", "bash_exec", "Task"]),
        )
        perms = CopilotPermissions(tools=["bash_exec"], tools_exclude=True)
        allowed, _ = apply_tool_permissions(perms, use_e2b=False)
        assert "mcp__copilot__bash_exec" not in allowed
        assert "mcp__copilot__run_block" in allowed

    def test_whitelist_keeps_only_listed(self, mocker):
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_copilot_tool_names",
            return_value=[
                "mcp__copilot__run_block",
                "mcp__copilot__web_fetch",
                "Task",
                "WebSearch",
            ],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_sdk_disallowed_tools",
            return_value=["Bash"],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": object(), "web_fetch": object()},
        )
        mocker.patch(
            "backend.copilot.permissions.all_known_tool_names",
            return_value=frozenset(["run_block", "web_fetch", "Task", "WebSearch"]),
        )
        perms = CopilotPermissions(tools=["run_block"], tools_exclude=False)
        allowed, _ = apply_tool_permissions(perms, use_e2b=False)
        assert "mcp__copilot__run_block" in allowed
        assert "mcp__copilot__web_fetch" not in allowed
        assert "Task" not in allowed

    def test_read_tool_always_included_even_when_blacklisted(self, mocker):
        """mcp__copilot__Read must stay in allowed even if Read is explicitly blacklisted."""
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_copilot_tool_names",
            return_value=[
                "mcp__copilot__run_block",
                "mcp__copilot__Read",
                "Task",
            ],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_sdk_disallowed_tools",
            return_value=[],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": object()},
        )
        mocker.patch(
            "backend.copilot.permissions.all_known_tool_names",
            return_value=frozenset(["run_block", "Read", "Task"]),
        )
        # Explicitly blacklist Read
        perms = CopilotPermissions(tools=["Read"], tools_exclude=True)
        allowed, _ = apply_tool_permissions(perms, use_e2b=False)
        assert "mcp__copilot__Read" in allowed  # always preserved for SDK internals
        assert "mcp__copilot__run_block" in allowed
        assert "Task" in allowed

    def test_read_tool_always_included_with_narrow_whitelist(self, mocker):
        """mcp__copilot__Read must stay in allowed even when not in a whitelist."""
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_copilot_tool_names",
            return_value=[
                "mcp__copilot__run_block",
                "mcp__copilot__Read",
                "Task",
            ],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_sdk_disallowed_tools",
            return_value=[],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": object()},
        )
        mocker.patch(
            "backend.copilot.permissions.all_known_tool_names",
            return_value=frozenset(["run_block", "Read", "Task"]),
        )
        # Whitelist only run_block — Read not listed
        perms = CopilotPermissions(tools=["run_block"], tools_exclude=False)
        allowed, _ = apply_tool_permissions(perms, use_e2b=False)
        assert "mcp__copilot__Read" in allowed  # always preserved for SDK internals
        assert "mcp__copilot__run_block" in allowed

    def test_e2b_file_tools_included_when_sdk_builtin_whitelisted(self, mocker):
        """In E2B mode, whitelisting 'Read' must include mcp__copilot__read_file."""
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_copilot_tool_names",
            return_value=[
                "mcp__copilot__run_block",
                "mcp__copilot__Read",
                "mcp__copilot__read_file",
                "mcp__copilot__write_file",
                "Task",
            ],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_sdk_disallowed_tools",
            return_value=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": object()},
        )
        mocker.patch(
            "backend.copilot.permissions.all_known_tool_names",
            return_value=frozenset(["run_block", "Read", "Write", "Task"]),
        )
        mocker.patch(
            "backend.copilot.sdk.e2b_file_tools.E2B_FILE_TOOL_NAMES",
            ["read_file", "write_file", "edit_file", "glob", "grep"],
        )
        # Whitelist Read and run_block — E2B read_file should be included
        perms = CopilotPermissions(tools=["Read", "run_block"], tools_exclude=False)
        allowed, _ = apply_tool_permissions(perms, use_e2b=True)
        assert "mcp__copilot__read_file" in allowed
        assert "mcp__copilot__run_block" in allowed
        # Write not whitelisted — write_file should NOT be included
        assert "mcp__copilot__write_file" not in allowed

    def test_e2b_file_tools_excluded_when_sdk_builtin_blacklisted(self, mocker):
        """In E2B mode, blacklisting 'Read' must also remove mcp__copilot__read_file."""
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_copilot_tool_names",
            return_value=[
                "mcp__copilot__run_block",
                "mcp__copilot__Read",
                "mcp__copilot__read_file",
                "Task",
            ],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.get_sdk_disallowed_tools",
            return_value=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
        )
        mocker.patch(
            "backend.copilot.sdk.tool_adapter.TOOL_REGISTRY",
            {"run_block": object()},
        )
        mocker.patch(
            "backend.copilot.permissions.all_known_tool_names",
            return_value=frozenset(["run_block", "Read", "Task"]),
        )
        mocker.patch(
            "backend.copilot.sdk.e2b_file_tools.E2B_FILE_TOOL_NAMES",
            ["read_file", "write_file", "edit_file", "glob", "grep"],
        )
        # Blacklist Read — E2B read_file should also be removed
        perms = CopilotPermissions(tools=["Read"], tools_exclude=True)
        allowed, _ = apply_tool_permissions(perms, use_e2b=True)
        assert "mcp__copilot__read_file" not in allowed
        assert "mcp__copilot__run_block" in allowed
        # mcp__copilot__Read is always preserved for SDK internals
        assert "mcp__copilot__Read" in allowed


# ---------------------------------------------------------------------------
# SDK_BUILTIN_TOOL_NAMES sanity check
# ---------------------------------------------------------------------------


class TestSdkBuiltinToolNames:
    def test_expected_builtins_present(self):
        expected = {
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "Task",
            "WebSearch",
            "TodoWrite",
        }
        assert expected.issubset(SDK_BUILTIN_TOOL_NAMES)

    def test_platform_names_match_tool_registry(self):
        """PLATFORM_TOOL_NAMES (derived from ToolName Literal) must match TOOL_REGISTRY keys."""
        registry_keys = frozenset(TOOL_REGISTRY.keys())
        assert PLATFORM_TOOL_NAMES == registry_keys, (
            f"ToolName Literal is out of sync with TOOL_REGISTRY. "
            f"Missing: {registry_keys - PLATFORM_TOOL_NAMES}, "
            f"Extra: {PLATFORM_TOOL_NAMES - registry_keys}"
        )

    def test_all_tool_names_is_union(self):
        """ALL_TOOL_NAMES must equal PLATFORM_TOOL_NAMES | SDK_BUILTIN_TOOL_NAMES."""
        assert ALL_TOOL_NAMES == PLATFORM_TOOL_NAMES | SDK_BUILTIN_TOOL_NAMES

    def test_no_overlap_between_platform_and_sdk(self):
        """Platform and SDK built-in names must not overlap."""
        assert PLATFORM_TOOL_NAMES.isdisjoint(SDK_BUILTIN_TOOL_NAMES)

    def test_known_tools_includes_registry_and_builtins(self):
        known = all_known_tool_names()
        assert "run_block" in known
        assert "Read" in known
        assert "Task" in known
