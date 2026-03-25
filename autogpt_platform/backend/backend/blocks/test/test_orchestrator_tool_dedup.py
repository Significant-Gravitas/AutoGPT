"""Tests for OrchestratorBlock tool name disambiguation.

When multiple nodes use the same block type, their tool names collide.
The Anthropic API requires unique tool names, so the orchestrator must
disambiguate them and enrich descriptions with hardcoded defaults.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.blocks.orchestrator import OrchestratorBlock
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
