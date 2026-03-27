"""Tests for OrchestratorBlock tool name disambiguation.

When multiple nodes use the same block type, their tool names collide.
The Anthropic API requires unique tool names, so the orchestrator must
disambiguate them and enrich descriptions with hardcoded defaults.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.blocks.orchestrator import OrchestratorBlock, _disambiguate_tool_names
from backend.blocks.text import MatchTextPatternBlock


def _make_mock_node(
    block,
    node_id: str,
    input_default: dict | None = None,
    metadata: dict | None = None,
):
    """Create a mock Node with the given block and defaults."""
    node = Mock()
    node.block = block
    node.block_id = block.id
    node.id = node_id
    node.input_default = input_default or {}
    node.metadata = metadata or {}
    return node


def _make_mock_link(source_name: str, sink_name: str, sink_id: str, source_id: str):
    """Create a mock Link."""
    return Mock(
        source_name=source_name,
        sink_name=sink_name,
        sink_id=sink_id,
        source_id=source_id,
    )


@pytest.mark.asyncio
async def test_duplicate_block_names_get_suffixed():
    """Two nodes using the same block type should produce unique tool names."""
    block = MatchTextPatternBlock()
    node_a = _make_mock_node(block, "node_a", input_default={"match": "foo"})
    node_b = _make_mock_node(block, "node_b", input_default={"match": "bar"})

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(names) == 2
    assert len(set(names)) == 2, f"Tool names are not unique: {names}"
    # Descriptive suffixes derived from the first default key/value
    base = OrchestratorBlock.cleanup(block.name)
    assert f"{base}_match_foo" in names
    assert f"{base}_match_bar" in names


@pytest.mark.asyncio
async def test_duplicate_tools_include_defaults_in_description():
    """Duplicate tools should have hardcoded defaults in description."""
    block = MatchTextPatternBlock()
    node_a = _make_mock_node(
        block, "node_a", input_default={"match": "error", "case_sensitive": True}
    )
    node_b = _make_mock_node(
        block, "node_b", input_default={"match": "warning", "case_sensitive": False}
    )

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    # Find each tool by descriptive suffix
    base = OrchestratorBlock.cleanup(block.name)
    tool_error = next(
        t for t in tools if t["function"]["name"] == f"{base}_match_error"
    )
    tool_warning = next(
        t for t in tools if t["function"]["name"] == f"{base}_match_warning"
    )

    # Descriptions should contain the hardcoded defaults (not the linked 'text' field)
    assert "[Pre-configured:" in tool_error["function"]["description"]
    assert "[Pre-configured:" in tool_warning["function"]["description"]
    assert '"error"' in tool_error["function"]["description"]
    assert '"warning"' in tool_warning["function"]["description"]


@pytest.mark.asyncio
async def test_unique_tool_names_unchanged():
    """When all tool names are already unique, no suffixing should occur."""
    block_a = MatchTextPatternBlock()
    node_a = _make_mock_node(
        block_a, "node_a", metadata={"customized_name": "search_errors"}
    )
    node_b = _make_mock_node(
        block_a, "node_b", metadata={"customized_name": "search_warnings"}
    )

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert "search_errors" in names
    assert "search_warnings" in names
    # No suffixing
    assert all("_1" not in n and "_2" not in n for n in names)


@pytest.mark.asyncio
async def test_no_hardcoded_defaults_key_leaks_to_tool_schema():
    """_hardcoded_defaults should be cleaned up and not sent to the LLM API."""
    block = MatchTextPatternBlock()
    node_a = _make_mock_node(block, "node_a", input_default={"match": "foo"})
    node_b = _make_mock_node(block, "node_b", input_default={"match": "bar"})

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    for tool in tools:
        assert "_hardcoded_defaults" not in tool["function"]


@pytest.mark.asyncio
async def test_single_tool_no_suffixing():
    """A single tool should never get suffixed."""
    block = MatchTextPatternBlock()
    node = _make_mock_node(block, "node_a", input_default={"match": "foo"})
    link = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [(link, node)]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    assert len(tools) == 1
    name = tools[0]["function"]["name"]
    assert not name.endswith("_1")
    assert not name.endswith("_2")
    # No Pre-configured in description for single tools
    assert "[Pre-configured:" not in tools[0]["function"].get("description", "")


@pytest.mark.asyncio
async def test_three_duplicates_all_get_unique_names():
    """Three nodes with same block type should all get unique suffixed names."""
    block = MatchTextPatternBlock()
    nodes_and_links = []
    for i, pattern in enumerate(["error", "warning", "info"]):
        node = _make_mock_node(block, f"node_{i}", input_default={"match": pattern})
        link = _make_mock_link(f"tools_^_{i}_~_text", "text", f"node_{i}", "orch")
        nodes_and_links.append((link, node))

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = nodes_and_links

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(names) == 3
    assert len(set(names)) == 3, f"Tool names are not unique: {names}"
    # Descriptive suffixes derived from match=error/warning/info
    base = OrchestratorBlock.cleanup(block.name)
    assert f"{base}_match_error" in names
    assert f"{base}_match_warning" in names
    assert f"{base}_match_info" in names


@pytest.mark.asyncio
async def test_linked_fields_excluded_from_defaults():
    """Fields that are linked (LLM provides them) should not appear in defaults."""
    block = MatchTextPatternBlock()
    # 'text' is linked, 'match' and 'case_sensitive' are hardcoded
    node_a = _make_mock_node(
        block,
        "node_a",
        input_default={"text": "ignored", "match": "error", "case_sensitive": True},
    )
    # Duplicate to trigger disambiguation
    node_b = _make_mock_node(
        block, "node_b", input_default={"text": "ignored", "match": "warning"}
    )

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    base = OrchestratorBlock.cleanup(block.name)
    tool_error = next(
        t for t in tools if t["function"]["name"] == f"{base}_match_error"
    )
    desc = tool_error["function"]["description"]
    # 'text' is linked so should NOT appear in Pre-configured
    assert "text=" not in desc
    # 'match' is hardcoded so should appear
    assert "match=" in desc


@pytest.mark.asyncio
async def test_mixed_unique_and_duplicate_names():
    """Only duplicate names get suffixed; unique names are left untouched."""
    block_a = MatchTextPatternBlock()
    node_a1 = _make_mock_node(block_a, "node_a1", input_default={"match": "foo"})
    node_a2 = _make_mock_node(block_a, "node_a2", input_default={"match": "bar"})

    # Use a different block with a custom name to be unique
    node_b = _make_mock_node(
        block_a, "node_b", metadata={"customized_name": "unique_tool"}
    )

    link_a1 = _make_mock_link("tools_^_a1_~_text", "text", "node_a1", "orch")
    link_a2 = _make_mock_link("tools_^_a2_~_text", "text", "node_a2", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a1, node_a1),
        (link_a2, node_a2),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 3
    assert "unique_tool" in names
    # Duplicates get descriptive suffixes from their defaults
    base = OrchestratorBlock.cleanup(block_a.name)
    assert f"{base}_match_foo" in names
    assert f"{base}_match_bar" in names


@pytest.mark.asyncio
async def test_sensitive_fields_excluded_from_defaults():
    """Credentials and other sensitive fields must not leak into descriptions."""
    block = MatchTextPatternBlock()
    node_a = _make_mock_node(
        block,
        "node_a",
        input_default={
            "match": "error",
            "credentials": {"api_key": "sk-secret"},
            "api_key": "my-key",
            "password": "hunter2",
        },
    )
    node_b = _make_mock_node(
        block, "node_b", input_default={"match": "warning", "credentials": {"x": "y"}}
    )

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    for tool in tools:
        desc = tool["function"].get("description", "")
        assert "sk-secret" not in desc
        assert "my-key" not in desc
        assert "hunter2" not in desc
        assert "credentials=" not in desc
        assert "api_key=" not in desc
        assert "password=" not in desc


@pytest.mark.asyncio
async def test_long_tool_name_truncated():
    """Tool names exceeding 64 chars should be truncated before suffixing."""
    block = MatchTextPatternBlock()
    long_name = "a" * 63  # 63 chars, adding _1 would make 65

    node_a = _make_mock_node(block, "node_a", metadata={"customized_name": long_name})
    node_b = _make_mock_node(block, "node_b", metadata={"customized_name": long_name})

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    for tool in tools:
        name = tool["function"]["name"]
        assert len(name) <= 64, f"Tool name exceeds 64 chars: {name!r} ({len(name)})"


@pytest.mark.asyncio
async def test_suffix_collision_with_user_named_tool():
    """If a user-named tool matches a descriptive suffix, dedup should handle it."""
    block = MatchTextPatternBlock()
    base = OrchestratorBlock.cleanup(block.name)

    # Two nodes with same block name and same first default -> same descriptive suffix
    # Both have match="foo", so they would both try _match_foo -> collision
    node_a = _make_mock_node(block, "node_a", input_default={"match": "foo"})
    node_b = _make_mock_node(block, "node_b", input_default={"match": "foo"})

    # A third node that a user has customized to match the descriptive suffix
    node_c = _make_mock_node(
        block, "node_c", metadata={"customized_name": f"{base}_match_foo"}
    )

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")
    link_c = _make_mock_link("tools_^_c_~_text", "text", "node_c", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
        (link_c, node_c),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == len(names), f"Tool names are not unique: {names}"
    # The user-named tool keeps its name
    assert f"{base}_match_foo" in names


def test_disambiguate_skips_malformed_tools():
    """Malformed tools (missing function/name) should not crash disambiguation."""
    tools: list = [
        {"function": {"name": "good_tool", "description": "A tool"}},
        {"function": {"name": "good_tool", "description": "Another tool"}},
        # Missing 'function' key entirely
        {"type": "function"},
        # 'function' present but missing 'name'
        {"function": {"description": "no name"}},
        # Not even a dict
        "not_a_dict",
    ]
    # Should not raise
    _disambiguate_tool_names(tools)

    # The two good tools should be disambiguated
    names = [
        t.get("function", {}).get("name")
        for t in tools
        if isinstance(t, dict)
        and isinstance(t.get("function"), dict)
        and "name" in t.get("function", {})
    ]
    assert "good_tool_1" in names
    assert "good_tool_2" in names


def test_disambiguate_handles_missing_description():
    """Tools with no description key should still get Pre-configured appended."""
    tools: list[dict] = [
        {
            "function": {
                "name": "my_tool",
                "_hardcoded_defaults": {"key": "val1"},
            }
        },
        {
            "function": {
                "name": "my_tool",
                "description": "Has desc",
                "_hardcoded_defaults": {"key": "val2"},
            }
        },
    ]
    _disambiguate_tool_names(tools)

    # Descriptive suffixes derived from key=val1 and key=val2
    tool_val1 = next(t for t in tools if t["function"]["name"] == "my_tool_key_val1")
    tool_val2 = next(t for t in tools if t["function"]["name"] == "my_tool_key_val2")
    # Both should have Pre-configured
    assert "[Pre-configured:" in tool_val1["function"].get("description", "")
    assert "[Pre-configured:" in tool_val2["function"].get("description", "")


# ---------------------------------------------------------------------------
# Additional edge-case tests requested during review
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_blocks_no_hardcoded_defaults():
    """Two identical blocks with NO hardcoded defaults still get unique suffixes."""
    block = MatchTextPatternBlock()
    # No input_default at all -- both nodes are "blank"
    node_a = _make_mock_node(block, "node_a")
    node_b = _make_mock_node(block, "node_b")

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(names) == 2
    assert len(set(names)) == 2, f"Tool names are not unique: {names}"
    # With no defaults, falls back to numeric suffixes
    base = OrchestratorBlock.cleanup(block.name)
    assert f"{base}_1" in names
    assert f"{base}_2" in names


@pytest.mark.asyncio
async def test_very_long_block_names_truncated_with_suffix():
    """Block names > 64 chars must be truncated so name+suffix fits in 64 chars."""
    block = MatchTextPatternBlock()
    # A name that is exactly 70 characters long
    long_name = "x" * 70

    node_a = _make_mock_node(block, "node_a", metadata={"customized_name": long_name})
    node_b = _make_mock_node(block, "node_b", metadata={"customized_name": long_name})

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(names) == 2
    assert len(set(names)) == 2, f"Tool names are not unique: {names}"
    for name in names:
        assert len(name) <= 64, f"Tool name exceeds 64 chars: {name!r} ({len(name)})"
    # Suffixes should still be present
    assert any(n.endswith("_1") for n in names)
    assert any(n.endswith("_2") for n in names)


@pytest.mark.asyncio
async def test_five_plus_duplicates_all_unique():
    """Five duplicate blocks should produce _1 through _5, all unique."""
    block = MatchTextPatternBlock()
    nodes_and_links = []
    for i in range(5):
        node = _make_mock_node(
            block, f"node_{i}", input_default={"match": f"pattern_{i}"}
        )
        link = _make_mock_link(f"tools_^_{i}_~_text", "text", f"node_{i}", "orch")
        nodes_and_links.append((link, node))

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = nodes_and_links

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(names) == 5
    assert len(set(names)) == 5, f"Tool names are not unique: {names}"


@pytest.mark.asyncio
async def test_mixed_duplicates_and_custom_named_same_type():
    """Two same-type unnamed blocks + one same-type with custom name.

    Only the unnamed duplicates should get suffixed; the custom-named one
    keeps its name.
    """
    block = MatchTextPatternBlock()
    # Two unnamed (will collide on default block name)
    node_a = _make_mock_node(block, "node_a", input_default={"match": "alpha"})
    node_b = _make_mock_node(block, "node_b", input_default={"match": "beta"})
    # Same block type, but with a custom name -- unique, no suffix needed
    node_c = _make_mock_node(
        block, "node_c", metadata={"customized_name": "summarizer"}
    )

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")
    link_c = _make_mock_link("tools_^_c_~_text", "text", "node_c", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
        (link_c, node_c),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 3, f"Tool names are not unique: {names}"
    # Custom-named tool keeps its name
    assert "summarizer" in names
    # The two duplicates get descriptive suffixes from defaults
    base = OrchestratorBlock.cleanup(block.name)
    assert f"{base}_match_alpha" in names
    assert f"{base}_match_beta" in names
    # "summarizer" should NOT have a numeric suffix
    assert not any(n.startswith("summarizer_") and n[-1].isdigit() for n in names)


@pytest.mark.asyncio
async def test_sensitive_fields_filtered_non_sensitive_shown():
    """Duplicate blocks with sensitive AND non-sensitive defaults.

    Sensitive fields (credentials, api_key, token, etc.) must be filtered
    from descriptions, while non-sensitive defaults must still appear.
    """
    block = MatchTextPatternBlock()
    node_a = _make_mock_node(
        block,
        "node_a",
        input_default={
            "match": "important",
            "token": "tok-secret-123",
            "secret": "my_secret_value",
            "case_sensitive": True,
        },
    )
    node_b = _make_mock_node(
        block,
        "node_b",
        input_default={
            "match": "other",
            "auth": "bearer xyz",
            "access_token": "at-456",
        },
    )

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    for tool in tools:
        desc = tool["function"].get("description", "")
        # Sensitive values must NOT appear
        assert "tok-secret-123" not in desc
        assert "my_secret_value" not in desc
        assert "bearer xyz" not in desc
        assert "at-456" not in desc
        assert "token=" not in desc
        assert "secret=" not in desc
        assert "auth=" not in desc
        assert "access_token=" not in desc

    # Non-sensitive defaults SHOULD appear in the descriptions
    all_descs = " ".join(t["function"].get("description", "") for t in tools)
    assert "match=" in all_descs
    assert '"important"' in all_descs or '"other"' in all_descs


@pytest.mark.asyncio
async def test_empty_input_default_no_crash():
    """Blocks with empty ({}) or None input_default must not crash."""
    block = MatchTextPatternBlock()
    # Explicit empty dict
    node_a = _make_mock_node(block, "node_a", input_default={})
    # None (falls back to {} in _make_mock_node, but test the path)
    node_b = _make_mock_node(block, "node_b", input_default={})

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(names) == 2
    assert len(set(names)) == 2, f"Tool names are not unique: {names}"


def test_disambiguate_empty_and_none_defaults():
    """_disambiguate_tool_names handles empty/None _hardcoded_defaults gracefully."""
    tools: list[dict] = [
        {"function": {"name": "tool", "_hardcoded_defaults": {}}},
        {"function": {"name": "tool", "_hardcoded_defaults": None}},
        {"function": {"name": "tool"}},  # key missing entirely
    ]
    # Must not raise
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 3, f"Tool names are not unique: {names}"


@pytest.mark.asyncio
async def test_unicode_in_block_names_and_defaults():
    """Unicode characters in custom names and defaults must not cause errors."""
    block = MatchTextPatternBlock()
    # Unicode custom name -- cleanup will sanitise non-alphanumeric chars
    node_a = _make_mock_node(
        block,
        "node_a",
        input_default={"match": "cafe\u0301"},
        metadata={"customized_name": "caf\u00e9_finder"},
    )
    node_b = _make_mock_node(
        block,
        "node_b",
        input_default={"match": "\u00fc\u00f6\u00e4"},
        metadata={"customized_name": "caf\u00e9_finder"},
    )

    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(names) == 2
    assert len(set(names)) == 2, f"Tool names are not unique: {names}"
    # Names should only contain [a-zA-Z0-9_-] after cleanup
    import re

    for name in names:
        assert re.fullmatch(
            r"[a-zA-Z0-9_-]+", name
        ), f"Invalid chars in tool name: {name!r}"


def test_disambiguate_unicode_in_defaults_description():
    """Unicode default values should appear in descriptions without encoding errors."""
    tools: list[dict] = [
        {
            "function": {
                "name": "searcher",
                "description": "Search tool",
                "_hardcoded_defaults": {"query": "\u00fc\u00f6\u00e4\u00df"},
            }
        },
        {
            "function": {
                "name": "searcher",
                "description": "Search tool",
                "_hardcoded_defaults": {"query": "\u65e5\u672c\u8a9e"},
            }
        },
    ]
    # Must not raise
    _disambiguate_tool_names(tools)

    for tool in tools:
        desc = tool["function"].get("description", "")
        assert "[Pre-configured:" in desc


def test_disambiguate_numeric_fallback_skips_taken_suffix():
    """When descriptive suffix collides, the while-loop should skip taken names.

    If a tool is already named 'my_tool_1', and two other 'my_tool' tools
    need numeric fallback, they should use _2 and _3 (skipping _1).
    """
    tools: list[dict] = [
        # A tool already named with what looks like a numeric suffix
        {"function": {"name": "my_tool_1", "description": "Custom named"}},
        # Two tools that will collide -- no defaults, so numeric fallback
        {"function": {"name": "my_tool", "description": "First"}},
        {"function": {"name": "my_tool", "description": "Second"}},
    ]
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == len(names), f"Tool names are not unique: {names}"
    # The pre-existing tool keeps its name
    assert "my_tool_1" in names
    # The duplicates should skip _1 (taken) and use _2, _3
    assert "my_tool_2" in names
    assert "my_tool_3" in names


@pytest.mark.asyncio
async def test_tool_name_collision_with_existing_suffix():
    """End-to-end: a custom-named block collides with a would-be suffix.

    If a block is already named 'aitextgeneratorblock_1', and two other blocks
    of the same type need suffixing, the while-loop should skip _1 and use _2.
    This exercises the numeric fallback path with no defaults.
    """
    block = MatchTextPatternBlock()
    base = OrchestratorBlock.cleanup(block.name)

    # One node with custom name that matches what _1 suffix would produce
    node_existing = _make_mock_node(
        block, "node_existing", metadata={"customized_name": f"{base}_1"}
    )
    # Two more nodes with NO defaults -- forces numeric fallback, which should
    # skip _1 (taken by the custom-named node)
    node_a = _make_mock_node(block, "node_a")
    node_b = _make_mock_node(block, "node_b")

    link_existing = _make_mock_link("tools_^_e_~_text", "text", "node_existing", "orch")
    link_a = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")
    link_b = _make_mock_link("tools_^_b_~_text", "text", "node_b", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [
        (link_existing, node_existing),
        (link_a, node_a),
        (link_b, node_b),
    ]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == len(names), f"Tool names are not unique: {names}"
    # The custom-named node keeps "{base}_1"
    assert f"{base}_1" in names
    # The two colliding nodes must skip _1 (taken) and use _2, _3
    assert f"{base}_2" in names
    assert f"{base}_3" in names
