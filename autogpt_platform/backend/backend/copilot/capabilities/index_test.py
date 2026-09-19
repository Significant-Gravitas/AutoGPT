"""Index behaviour on a small synthetic registry (no blocks loaded)."""

import pytest

from backend.copilot.permissions import CopilotPermissions

from .index import CapabilityIndex
from .models import CapabilityEntry, Connection, Implementation
from .ranking import ConnectionState

LINEAR_ID = "11111111-1111-1111-1111-111111111111"
HTTP_ID = "22222222-2222-2222-2222-222222222222"
GITHUB_ID = "33333333-3333-3333-3333-333333333333"
INPUT_ID = "44444444-4444-4444-4444-444444444444"


def _block(
    block_id, name, purpose, *, provider=None, klass="service", context="both", args=()
):
    connection = (
        Connection(required=True, key_type="provider", key=provider)
        if provider
        else Connection(required=False)
    )
    return CapabilityEntry(
        id=f"block:{block_id}",
        kind="block",
        klass=klass,
        name=name,
        purpose=purpose,
        tags=[*([provider] if provider else []), "issue tracking"],
        context=context,
        implementations=[Implementation(kind="block", ref=block_id)],
        connection=connection,
        argument_names=list(args),
    )


@pytest.fixture
def index() -> CapabilityIndex:
    return CapabilityIndex(
        [
            _block(
                LINEAR_ID,
                "LinearCreateIssueBlock",
                "Create a new issue in Linear.",
                provider="linear",
                args=("title", "team"),
            ),
            _block(
                GITHUB_ID,
                "GithubMakeIssueBlock",
                "Create an issue on a GitHub repository.",
                provider="github",
                args=("title", "repo"),
            ),
            _block(
                HTTP_ID,
                "SendWebRequestBlock",
                "Make an HTTP request to any URL.",
                klass="primitive",
                args=("url", "method", "body"),
            ),
            _block(INPUT_ID, "AgentInputBlock", "Graph input.", context="graph"),
            CapabilityEntry(
                id="tool:web_search",
                kind="tool",
                name="web_search",
                purpose="Search the web.",
                tags=["web", "search"],
                context="direct",
                implementations=[
                    Implementation(kind="tool", ref="web_search", context="direct")
                ],
                argument_names=["query"],
                eager=True,
            ),
            CapabilityEntry(
                id="mcp:mcp.sentry.dev",
                kind="mcp_server",
                name="Sentry",
                purpose="Query Sentry issues and events.",
                tags=["sentry", "mcp", "mcp.sentry.dev"],
                context="direct",
                implementations=[
                    Implementation(kind="mcp_server", ref="https://mcp.sentry.dev/mcp")
                ],
                connection=Connection(
                    required=True,
                    key_type="server_url",
                    key="https://mcp.sentry.dev/mcp",
                ),
            ),
        ]
    )


def test_exact_class_name_wins(index):
    result = index.search("OrchestratorBlock")
    assert result.hits == []
    result = index.search("linear create issue block")
    assert result.names[0] == "LinearCreateIssueBlock"
    assert result.hits[0].reason == "exact_name"
    assert index.search("LinearCreateIssue").names[0] == "LinearCreateIssueBlock"


def test_uuid_lookup(index):
    result = index.search(GITHUB_ID)
    assert result.ids == [f"block:{GITHUB_ID}"] and result.hits[0].reason == "exact_id"
    assert index.get(GITHUB_ID) is index.get(f"block:{GITHUB_ID}")


def test_service_query_restricts_main_list_and_offers_primitive_fallback(index):
    result = index.search("linear issue")
    assert result.service == "linear"
    assert result.names == ["LinearCreateIssueBlock"]
    assert [h.entry.name for h in result.fallback] == ["SendWebRequestBlock"]


def test_mcp_host_and_slug_are_service_tags(index):
    assert index.search("sentry").ids == ["mcp:mcp.sentry.dev"]
    assert index.search("mcp.sentry.dev").ids == ["mcp:mcp.sentry.dev"]


def test_primitives_rank_below_services_on_a_generic_query(index):
    result = index.search("create issue")
    assert set(result.names[:2]) == {"LinearCreateIssueBlock", "GithubMakeIssueBlock"}
    assert result.service is None


def test_connected_service_ranks_first(index):
    state = ConnectionState(providers=frozenset({"github"}))
    result = index.search("create issue", connections=state)
    assert result.names[0] == "GithubMakeIssueBlock"
    assert result.hits[0].connected is True and result.hits[1].connected is False


def test_graph_only_entries_hidden_in_direct_context(index):
    assert index.search("agent input").hits == []
    assert index.search("agent input", context="graph").names == ["AgentInputBlock"]


def test_kind_filter(index):
    assert index.search("issue", kind="mcp_server").ids == ["mcp:mcp.sentry.dev"]


def test_permissions_filter_blocks_and_tools(index):
    perms = CopilotPermissions(blocks=["LinearCreateIssueBlock"], tools=["web_search"])
    result = index.search("create issue", permissions=perms)
    assert "LinearCreateIssueBlock" not in result.names
    assert "web_search" not in index.search("search the web", permissions=perms).names
    whitelist = CopilotPermissions(blocks=[LINEAR_ID[:8]], blocks_exclude=False)
    names = index.search("create issue", permissions=whitelist).names
    assert names[0] == "LinearCreateIssueBlock" and "GithubMakeIssueBlock" not in names


def test_argument_names_are_searchable(index):
    assert index.search("repo").names[0] == "GithubMakeIssueBlock"


def test_empty_and_nonsense_queries(index):
    assert index.search("   ").hits == []
    assert index.search("zzzz qqqq").hits == []


def test_eager_tools_are_flagged_in_listing(index):
    hit = index.search("web search").hits[0]
    assert hit.entry.listing()["eager"] is True
