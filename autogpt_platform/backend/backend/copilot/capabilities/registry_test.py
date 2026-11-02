"""The assembled platform registry over the real tool, block and catalog sources."""

import json

import pytest

from backend.blocks import get_blocks
from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY
from backend.integrations.mcp_catalog import get_mcp_catalog

from .index import CapabilityIndex
from .registry import MERGED_IMPLEMENTATIONS, build_entries
from .resolve import resolve_entry
from .sources import RETIRED_TOOLS
from .sources.blocks import _block_entry
from .sources.mcp_catalog import setup_hint

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
    # The folded block's description joins the tool's, so a query phrased
    # for the block ("sandbox") still lands on the one capability.
    assert "sandbox" in entry.description
    assert index.search("code sandbox").names[0] == "bash_exec"


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


def test_optimized_description_is_indexed_alongside_the_source_text():
    """``optimized_description`` is curated for retrieval and loaded from the
    database at runtime, so no block carries one in CI; indexing only the
    source description would drop it from the index in production."""
    block = next(
        cls() for cls in get_blocks().values() if cls.__name__ == "FileStoreBlock"
    )
    block.optimized_description = "Curated: keeps a download in the workspace."
    entry = _block_entry(block)
    assert entry.purpose.startswith("Curated:")
    assert (
        "Curated" in entry.description and "Downloads and stores" in entry.description
    )


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
    v2 = resolve_entry(index, "https://mcp.atlassian.com/v2/mcp")
    forge = resolve_entry(index, "https://mcp.atlassian.com/v1/forge/mcp")
    assert v2 is not None and forge is not None and v2.id != forge.id


def test_custom_presets_carry_no_url_in_ref(index):
    """A ``custom`` preset has no shared endpoint. Putting its catalog name
    in the implementation ref sent it downstream as a hostname, where it
    failed validation as "Hostname 'mcp_amplitude' has unsupported
    characters" instead of telling the user to supply their own URL."""
    custom = [
        entry
        for entry in index.entries
        if entry.kind == "mcp_server" and not entry.implementations[0].ref
    ]
    assert custom, "expected catalog presets with no server_url"
    assert all(not entry.connection.key for entry in custom)
    assert "mcp:amplitude" in {entry.id for entry in custom}
    for entry in custom:
        hint = setup_hint(entry.schema_ref or entry.id)
        assert "Settings" in hint and entry.name in hint


def test_hosted_presets_keep_their_url(index):
    hosted = [
        entry
        for entry in index.entries
        if entry.kind == "mcp_server" and entry.implementations[0].ref
    ]
    assert all(entry.implementations[0].ref.startswith("https://") for entry in hosted)


def test_listing_stays_compact(index):
    longest = max(len(json.dumps(e.listing())) for e in index.entries)
    assert longest <= 320  # ≈ 60-80 tokens worst case
