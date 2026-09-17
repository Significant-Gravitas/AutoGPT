"""The assembled platform registry over the real tool, block and catalog sources."""

import json

import pytest

from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY
from backend.integrations.mcp_catalog import get_mcp_catalog

from .index import CapabilityIndex
from .registry import MERGED_IMPLEMENTATIONS, build_entries
from .sources import RETIRED_TOOLS

EXPLICIT_PRIMITIVES = {
    "ExecuteCodeBlock",
    "SQLQueryBlock",
    "SendAuthenticatedWebRequestBlock",
    "ReplicateModelBlock",
    "ClaudeCodeBlock",
    "CodeGenerationBlock",
}


@pytest.fixture(scope="module")
def entries():
    return build_entries(TOOL_REGISTRY, TOOL_GROUPS)


@pytest.fixture(scope="module")
def index(entries) -> CapabilityIndex:
    return CapabilityIndex(entries)


def test_ids_are_unique_and_prefixed(entries):
    ids = [entry.id for entry in entries]
    assert len(ids) == len(set(ids))
    assert all(i.split(":", 1)[0] in {"tool", "block", "mcp"} for i in ids)


def test_retired_tools_have_no_entry(index):
    for name in RETIRED_TOOLS:
        assert index.get(f"tool:{name}") is None


def test_bash_exec_and_execute_code_block_are_one_capability(index):
    entry = index.get("tool:bash_exec")
    assert entry is not None and entry.eager
    kinds = {impl.kind for impl in entry.implementations}
    assert kinds == {"tool", "block"} and entry.context == "both"
    folded = MERGED_IMPLEMENTATIONS["tool:bash_exec"]
    assert all(e.name != folded for e in index.entries)
    assert index.search("execute python code").names[0] == "bash_exec"


def test_block_capability_kind_drives_class(index):
    by_name = {entry.name: entry for entry in index.entries}
    for name in EXPLICIT_PRIMITIVES - {"ExecuteCodeBlock"}:  # folded into bash_exec
        assert by_name[name].klass == "primitive", name
    assert by_name["LinearCreateIssueBlock"].klass == "service"
    assert by_name["LinearCreateIssueBlock"].connection.key == "linear"
    # Multi-provider LLM blocks and credential-free blocks derive to primitive.
    assert by_name["AITextGeneratorBlock"].klass == "primitive"
    assert by_name["FillTextTemplateBlock"].klass == "primitive"
    assert by_name["SendAuthenticatedWebRequestBlock"].connection.key_type == "host"


def test_graph_only_blocks_keep_graph_context(index):
    assert "AgentInputBlock" not in index.search("AgentInputBlock").names
    assert (
        index.search("AgentInputBlock", context="graph").names[0] == "AgentInputBlock"
    )


def test_catalog_servers_are_entries_keyed_by_host(index):
    linear = index.get("mcp:mcp.linear.app")
    assert linear is not None and linear.kind == "mcp_server"
    assert linear.connection.key_type == "server_url"
    assert index.search("sentry").ids[0] == "mcp:mcp.sentry.dev"


def test_every_catalog_preset_survives_as_an_entry(index):
    """Atlassian ships two servers on ``mcp.atlassian.com`` (v2 and Forge).
    Keying both by host dropped one of them, so entries on a shared host fall
    back to their slug and every preset keeps an id of its own."""
    presets = get_mcp_catalog()
    mcp_ids = [e.id for e in index.entries if e.id.startswith("mcp:")]
    assert len(mcp_ids) == len(presets)
    assert len(set(mcp_ids)) == len(mcp_ids)
    assert {"mcp:atlassian", "mcp:atlassian_forge"} <= set(mcp_ids)


def test_listing_stays_compact(index):
    longest = max(len(json.dumps(e.listing())) for e in index.entries)
    assert longest <= 320  # ≈ 60-80 tokens worst case
