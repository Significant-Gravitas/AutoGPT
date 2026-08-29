"""Tests for OrchestratorBlock tool name disambiguation.

When multiple nodes use the same block type, their tool names collide.
The Anthropic API requires unique tool names, so the orchestrator must
disambiguate them and enrich descriptions with hardcoded defaults.
"""

from types import SimpleNamespace
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
    # Should be suffixed with _1, _2
    base = OrchestratorBlock.cleanup(block.name)
    assert f"{base}_1" in names
    assert f"{base}_2" in names


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

    # Find each tool by suffix
    tool_1 = next(t for t in tools if t["function"]["name"].endswith("_1"))
    tool_2 = next(t for t in tools if t["function"]["name"].endswith("_2"))

    # Descriptions should contain the hardcoded defaults (not the linked 'text' field)
    assert "[Pre-configured:" in tool_1["function"]["description"]
    assert "[Pre-configured:" in tool_2["function"]["description"]
    assert '"error"' in tool_1["function"]["description"]
    assert '"warning"' in tool_2["function"]["description"]


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
    base = OrchestratorBlock.cleanup(block.name)
    assert f"{base}_1" in names
    assert f"{base}_2" in names
    assert f"{base}_3" in names


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

    tool_1 = next(t for t in tools if t["function"]["name"].endswith("_1"))
    desc = tool_1["function"]["description"]
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
    base = OrchestratorBlock.cleanup(block_a.name)
    assert f"{base}_1" in names
    assert f"{base}_2" in names


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
    """If a user-named tool is 'my_tool_1', dedup of 'my_tool' should skip to _2."""
    block = MatchTextPatternBlock()
    # Two nodes with same block name (will collide)
    node_a = _make_mock_node(block, "node_a", input_default={"match": "foo"})
    node_b = _make_mock_node(block, "node_b", input_default={"match": "bar"})

    # A third node that a user has customized to match the _1 suffix pattern
    base = OrchestratorBlock.cleanup(block.name)
    node_c = _make_mock_node(block, "node_c", metadata={"customized_name": f"{base}_1"})

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
    assert f"{base}_1" in names
    # The duplicates should skip _1 (taken) and use _2, _3
    assert f"{base}_2" in names
    assert f"{base}_3" in names


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


def test_disambiguate_skips_non_string_tool_names():
    """Tools whose 'name' is not a string (None, int, list) must not crash."""
    tools: list = [
        {"function": {"name": "good_tool", "description": "A tool"}},
        {"function": {"name": "good_tool", "description": "Another tool"}},
        # name is None
        {"function": {"name": None, "description": "null name"}},
        # name is an integer
        {"function": {"name": 123, "description": "int name"}},
        # name is a list
        {"function": {"name": ["a", "b"], "description": "list name"}},
    ]
    # Should not raise
    _disambiguate_tool_names(tools)

    # The two good tools should be disambiguated
    names = [
        t["function"]["name"]
        for t in tools
        if isinstance(t, dict)
        and isinstance(t.get("function"), dict)
        and isinstance(t["function"].get("name"), str)
    ]
    assert "good_tool_1" in names
    assert "good_tool_2" in names
    # Non-string names should be left untouched (skipped, not mutated)
    assert tools[2]["function"]["name"] is None
    assert tools[3]["function"]["name"] == 123
    assert tools[4]["function"]["name"] == ["a", "b"]


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

    tool_1 = next(t for t in tools if t["function"]["name"] == "my_tool_1")
    tool_2 = next(t for t in tools if t["function"]["name"] == "my_tool_2")
    # Both should have Pre-configured
    assert "[Pre-configured:" in tool_1["function"].get("description", "")
    assert "[Pre-configured:" in tool_2["function"].get("description", "")


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
    base = OrchestratorBlock.cleanup(block.name)
    assert f"{base}_1" in names
    assert f"{base}_2" in names
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
    """The while-loop should skip taken names when assigning numeric suffixes.

    If a tool is already named 'my_tool_1', and two other 'my_tool' tools
    need suffixing, they should use _2 and _3 (skipping _1).
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


# ---------------------------------------------------------------------------
# Edge-case tests added to address reviewer feedback (ntindle)
# ---------------------------------------------------------------------------


def test_disambiguate_empty_list():
    """An empty tool list should be a no-op without errors."""
    tools: list[dict] = []
    _disambiguate_tool_names(tools)
    assert tools == []


def test_disambiguate_single_tool_no_change():
    """A single tool should never be modified (no duplicates to resolve)."""
    tools: list[dict] = [
        {
            "function": {
                "name": "only_tool",
                "description": "Solo tool",
                "_hardcoded_defaults": {"key": "val"},
            }
        },
    ]
    _disambiguate_tool_names(tools)
    assert tools[0]["function"]["name"] == "only_tool"
    # _hardcoded_defaults should be cleaned up even for non-duplicates
    assert "_hardcoded_defaults" not in tools[0]["function"]
    # Description should NOT have Pre-configured (no duplicates)
    assert "[Pre-configured:" not in tools[0]["function"]["description"]


def test_disambiguate_all_same_name_ten_tools():
    """Ten tools all sharing the same name should produce _1 through _10."""
    tools: list[dict] = [
        {"function": {"name": "searcher", "description": f"Tool {i}"}}
        for i in range(10)
    ]
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 10, f"Tool names are not unique: {names}"
    for i in range(1, 11):
        assert f"searcher_{i}" in names, f"Missing searcher_{i} in {names}"


def test_disambiguate_multiple_distinct_duplicate_groups():
    """Two groups of duplicates (group_a x2 and group_b x2) should each get suffixed."""
    tools: list[dict] = [
        {"function": {"name": "group_a", "description": "A1"}},
        {"function": {"name": "group_a", "description": "A2"}},
        {"function": {"name": "group_b", "description": "B1"}},
        {"function": {"name": "group_b", "description": "B2"}},
        {"function": {"name": "unique_c", "description": "C1"}},
    ]
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 5, f"Tool names are not unique: {names}"
    assert "group_a_1" in names
    assert "group_a_2" in names
    assert "group_b_1" in names
    assert "group_b_2" in names
    # unique tool is untouched
    assert "unique_c" in names


def test_disambiguate_large_default_value_truncated_in_description():
    """Default values exceeding 100 chars should be truncated in description."""
    long_value = "x" * 200
    tools: list[dict] = [
        {
            "function": {
                "name": "tool",
                "description": "Base desc",
                "_hardcoded_defaults": {"data": long_value},
            }
        },
        {
            "function": {
                "name": "tool",
                "description": "Base desc",
                "_hardcoded_defaults": {"data": "short"},
            }
        },
    ]
    _disambiguate_tool_names(tools)

    tool_with_long = next(
        t for t in tools if "truncated" in t["function"].get("description", "")
    )
    desc = tool_with_long["function"]["description"]
    assert "...<truncated>" in desc
    # The full 200-char value should NOT appear untruncated
    assert long_value not in desc


def test_disambiguate_suffix_collision_cascade():
    """When user-named tools occupy _1 through _4, new duplicates skip to _5, _6."""
    tools: list[dict] = [
        # User-named tools that look like suffixed names
        {"function": {"name": "search_1", "description": "User 1"}},
        {"function": {"name": "search_2", "description": "User 2"}},
        {"function": {"name": "search_3", "description": "User 3"}},
        {"function": {"name": "search_4", "description": "User 4"}},
        # Two actual duplicates that need dedup
        {"function": {"name": "search", "description": "Dup A"}},
        {"function": {"name": "search", "description": "Dup B"}},
    ]
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 6, f"Tool names are not unique: {names}"
    # User-named tools stay unchanged
    for i in range(1, 5):
        assert f"search_{i}" in names
    # The two duplicates must skip 1-4 and use 5, 6
    assert "search_5" in names
    assert "search_6" in names


def test_disambiguate_preserves_original_description():
    """The original description should be preserved as a prefix before [Pre-configured:]."""
    tools: list[dict] = [
        {
            "function": {
                "name": "my_tool",
                "description": "This is the original description.",
                "_hardcoded_defaults": {"mode": "fast"},
            }
        },
        {
            "function": {
                "name": "my_tool",
                "description": "This is the original description.",
                "_hardcoded_defaults": {"mode": "slow"},
            }
        },
    ]
    _disambiguate_tool_names(tools)

    for tool in tools:
        desc = tool["function"]["description"]
        assert desc.startswith("This is the original description.")
        assert "[Pre-configured:" in desc


def test_disambiguate_empty_string_names():
    """Tools with empty string names should still be disambiguated without errors."""
    tools: list[dict] = [
        {"function": {"name": "", "description": "Empty 1"}},
        {"function": {"name": "", "description": "Empty 2"}},
        {"function": {"name": "valid_tool", "description": "OK"}},
    ]
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 3, f"Tool names are not unique: {names}"
    assert "valid_tool" in names


@pytest.mark.asyncio
async def test_cleanup_special_characters_in_tool_name():
    """Tool names with special characters should be sanitised by cleanup()."""
    # cleanup replaces non-alphanumeric (except _ and -) with _
    result = OrchestratorBlock.cleanup("My Tool! @#$% v2.0")
    assert result == "my_tool_______v2_0"
    # Only [a-zA-Z0-9_-] should remain
    import re

    assert re.fullmatch(r"[a-zA-Z0-9_-]+", result)


def test_disambiguate_tools_with_boolean_and_numeric_defaults():
    """Boolean and numeric default values should serialize correctly in description."""
    tools: list[dict] = [
        {
            "function": {
                "name": "processor",
                "description": "Proc",
                "_hardcoded_defaults": {
                    "enabled": True,
                    "count": 42,
                    "ratio": 3.14,
                },
            }
        },
        {
            "function": {
                "name": "processor",
                "description": "Proc",
                "_hardcoded_defaults": {"enabled": False, "count": 0},
            }
        },
    ]
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 2

    tool_1 = next(t for t in tools if t["function"]["name"] == "processor_1")
    desc = tool_1["function"]["description"]
    assert "enabled=true" in desc
    assert "count=42" in desc
    assert "ratio=3.14" in desc


def test_disambiguate_preserves_non_duplicate_hardcoded_defaults_cleanup():
    """Non-duplicate tools should have _hardcoded_defaults removed but desc untouched."""
    tools: list[dict] = [
        {
            "function": {
                "name": "unique_a",
                "description": "A desc",
                "_hardcoded_defaults": {"key": "val"},
            }
        },
        {
            "function": {
                "name": "unique_b",
                "description": "B desc",
                "_hardcoded_defaults": {"key": "val2"},
            }
        },
    ]
    _disambiguate_tool_names(tools)

    for tool in tools:
        assert "_hardcoded_defaults" not in tool["function"]
        # Descriptions should NOT have Pre-configured (no duplicates)
        assert "[Pre-configured:" not in tool["function"]["description"]

    assert tools[0]["function"]["name"] == "unique_a"
    assert tools[1]["function"]["name"] == "unique_b"


# ---------------------------------------------------------------------------
# Additional test conditions — reviewer-requested coverage expansion
# ---------------------------------------------------------------------------


def test_disambiguate_preserves_parameters_and_metadata():
    """Disambiguation must NOT strip parameters, _field_mapping, or _sink_node_id."""
    tools: list[dict] = [
        {
            "function": {
                "name": "tool",
                "description": "Tool A",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "_field_mapping": {"query": "query"},
                "_sink_node_id": "node_a",
                "_hardcoded_defaults": {"mode": "fast"},
            }
        },
        {
            "function": {
                "name": "tool",
                "description": "Tool B",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "_field_mapping": {"query": "query"},
                "_sink_node_id": "node_b",
                "_hardcoded_defaults": {"mode": "slow"},
            }
        },
    ]
    _disambiguate_tool_names(tools)

    for tool in tools:
        func = tool["function"]
        # parameters must be untouched
        assert "parameters" in func
        assert func["parameters"]["properties"]["query"]["type"] == "string"
        # Internal metadata must survive
        assert "_field_mapping" in func
        assert "_sink_node_id" in func
        # _hardcoded_defaults must be cleaned up
        assert "_hardcoded_defaults" not in func


def test_disambiguate_name_with_leading_trailing_underscores():
    """Tool names with leading/trailing underscores should still disambiguate."""
    tools: list[dict] = [
        {"function": {"name": "_private_tool_", "description": "A"}},
        {"function": {"name": "_private_tool_", "description": "B"}},
    ]
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 2, f"Names not unique: {names}"


def test_disambiguate_name_at_exactly_64_chars():
    """Tool name at exactly 64 chars with no suffix needed stays unchanged."""
    name_64 = "a" * 64
    tools: list[dict] = [
        {"function": {"name": name_64, "description": "Only one"}},
    ]
    _disambiguate_tool_names(tools)
    assert tools[0]["function"]["name"] == name_64


def test_disambiguate_name_at_62_chars_fits_suffix():
    """Tool name at 62 chars + _1 suffix = 64 chars, should fit without truncation."""
    name_62 = "a" * 62
    tools: list[dict] = [
        {"function": {"name": name_62, "description": "A"}},
        {"function": {"name": name_62, "description": "B"}},
    ]
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 2
    for n in names:
        assert len(n) <= 64, f"Name too long: {n!r} ({len(n)} chars)"
    # _1 = 2 chars, 62 + 2 = 64 — fits exactly
    assert f"{name_62}_1" in names
    assert f"{name_62}_2" in names


def test_disambiguate_two_digit_suffix_truncates_base():
    """When suffix is _10 (3 chars), base must be truncated to 61 chars."""
    tools: list[dict] = [
        {"function": {"name": "a" * 63, "description": f"Tool {i}"}} for i in range(11)
    ]
    _disambiguate_tool_names(tools)

    names = [t["function"]["name"] for t in tools]
    assert len(set(names)) == 11, f"Names not unique: {names}"
    for n in names:
        assert len(n) <= 64, f"Name too long: {n!r} ({len(n)} chars)"


def test_disambiguate_defaults_with_nested_dict_values():
    """Nested dict/list values in defaults should serialize as JSON in description."""
    tools: list[dict] = [
        {
            "function": {
                "name": "proc",
                "description": "Processor",
                "_hardcoded_defaults": {
                    "config": {"nested": {"key": "val"}, "list": [1, 2, 3]},
                },
            }
        },
        {
            "function": {
                "name": "proc",
                "description": "Processor",
                "_hardcoded_defaults": {"config": {"nested": {"key": "other"}}},
            }
        },
    ]
    _disambiguate_tool_names(tools)

    for tool in tools:
        desc = tool["function"]["description"]
        assert "[Pre-configured:" in desc
        assert "config=" in desc


def test_disambiguate_defaults_with_null_value():
    """None values in defaults should serialize as 'null' in JSON."""
    tools: list[dict] = [
        {
            "function": {
                "name": "tool",
                "description": "A",
                "_hardcoded_defaults": {"optional_field": None},
            }
        },
        {
            "function": {
                "name": "tool",
                "description": "B",
                "_hardcoded_defaults": {"optional_field": "present"},
            }
        },
    ]
    _disambiguate_tool_names(tools)

    tool_1 = next(t for t in tools if t["function"]["name"] == "tool_1")
    assert "null" in tool_1["function"]["description"]


# ---------------------------------------------------------------------------
# Round-trip routing test: suffixed name -> correct node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suffixed_tool_call_routes_to_correct_node():
    """Round-trip: LLM calls a suffixed tool name and it routes to the right node.

    This verifies the reverse path of disambiguation.  After
    ``_create_tool_node_signatures`` produces suffixed names (_1, _2),
    ``_process_tool_calls`` must map the suffixed name back to the correct
    tool definition (and therefore the correct ``_sink_node_id``).

    Steps:
      1. Build two duplicate tools via ``_create_tool_node_signatures``.
      2. Simulate an LLM response that calls ``<base_name>_1``.
      3. Run ``_process_tool_calls`` and verify the resolved tool_def
         contains ``_sink_node_id == "node_a"`` (not "node_b").
    """
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
        tool_functions = await OrchestratorBlock._create_tool_node_signatures("orch")

    # Determine the suffixed names and their associated node IDs.
    base = OrchestratorBlock.cleanup(block.name)
    name_1 = f"{base}_1"
    name_2 = f"{base}_2"

    # Sanity: both suffixed names must exist.
    names = [t["function"]["name"] for t in tool_functions]
    assert name_1 in names
    assert name_2 in names

    # Build the node-id lookup the same way _process_tool_calls will use.
    tool_1_def = next(t for t in tool_functions if t["function"]["name"] == name_1)
    tool_2_def = next(t for t in tool_functions if t["function"]["name"] == name_2)

    # Simulate an LLM response calling name_1 with some arguments.
    mock_tool_call = SimpleNamespace(
        id="call_abc123",
        function=SimpleNamespace(
            name=name_1,
            arguments='{"text": "hello world"}',
        ),
    )
    mock_response = SimpleNamespace(tool_calls=[mock_tool_call])

    orchestrator = OrchestratorBlock()
    processed = orchestrator._process_tool_calls(mock_response, tool_functions)

    # Exactly one tool call was processed.
    assert len(processed) == 1
    result = processed[0]

    # The resolved tool_def must point to the FIRST node ("node_a"),
    # not the second ("node_b").
    assert result.tool_name == name_1
    assert (
        result.tool_def["function"]["_sink_node_id"]
        == tool_1_def["function"]["_sink_node_id"]
    )
    assert (
        result.tool_def["function"]["_sink_node_id"]
        != tool_2_def["function"]["_sink_node_id"]
    )

    # Verify the input data was correctly extracted via the field mapping.
    assert "text" in result.input_data

    # Now do the same for name_2 to confirm it routes to node_b.
    mock_tool_call_2 = SimpleNamespace(
        id="call_def456",
        function=SimpleNamespace(
            name=name_2,
            arguments='{"text": "goodbye world"}',
        ),
    )
    mock_response_2 = SimpleNamespace(tool_calls=[mock_tool_call_2])

    processed_2 = orchestrator._process_tool_calls(mock_response_2, tool_functions)
    assert len(processed_2) == 1
    result_2 = processed_2[0]

    assert result_2.tool_name == name_2
    assert (
        result_2.tool_def["function"]["_sink_node_id"]
        == tool_2_def["function"]["_sink_node_id"]
    )
    assert (
        result_2.tool_def["function"]["_sink_node_id"]
        != tool_1_def["function"]["_sink_node_id"]
    )


@pytest.mark.asyncio
async def test_customized_name_takes_priority_over_block_name():
    """When a node has customized_name in metadata, that should be the tool name."""
    block = MatchTextPatternBlock()
    custom = "my_custom_tool"
    node = _make_mock_node(block, "node_a", metadata={"customized_name": custom})
    link = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [(link, node)]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    assert tools[0]["function"]["name"] == custom


@pytest.mark.asyncio
async def test_customized_names_collide_get_suffixed():
    """Two nodes with the SAME customized_name should get suffixed."""
    block = MatchTextPatternBlock()
    node_a = _make_mock_node(
        block,
        "node_a",
        metadata={"customized_name": "searcher"},
        input_default={"match": "alpha"},
    )
    node_b = _make_mock_node(
        block,
        "node_b",
        metadata={"customized_name": "searcher"},
        input_default={"match": "beta"},
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
    assert len(set(names)) == 2, f"Names not unique: {names}"
    assert "searcher_1" in names
    assert "searcher_2" in names


@pytest.mark.asyncio
async def test_tool_has_correct_required_fields():
    """Tool parameters should include required fields from the block schema."""
    block = MatchTextPatternBlock()
    node = _make_mock_node(block, "node_a")
    link = _make_mock_link("tools_^_a_~_text", "text", "node_a", "orch")

    mock_db = AsyncMock()
    mock_db.get_connected_output_nodes.return_value = [(link, node)]

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        tools = await OrchestratorBlock._create_tool_node_signatures("orch")

    params = tools[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert "text" in params["properties"]
    assert params["additionalProperties"] is False


@pytest.mark.asyncio
async def test_disambiguation_does_not_modify_parameters():
    """After disambiguation, tool parameters should be identical to pre-disambiguation."""
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
        params = tool["function"]["parameters"]
        # Parameters must survive disambiguation intact
        assert "properties" in params
        assert "text" in params["properties"]
        assert params["type"] == "object"
