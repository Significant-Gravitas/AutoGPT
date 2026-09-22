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
FILE_ID = "55555555-5555-5555-5555-555555555555"

# Four sentences, of which ``clip_purpose`` keeps the first two: the one that
# says what the block does in CoPilot is the one a "save" query needs.
FILE_STORE_DESCRIPTION = (
    "Downloads and stores a file from a URL, data URI, or local path. "
    "Use this to fetch images, documents, or other files for processing. "
    "In CoPilot: saves to workspace (use list_workspace_files to see it). "
    "In graphs: outputs a data URI to pass to other blocks."
)


def _block(
    block_id,
    name,
    purpose,
    *,
    provider=None,
    klass="service",
    context="both",
    args=(),
    description="",
    tags=("issue tracking",),
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
        description=description,
        tags=[*([provider] if provider else []), *tags],
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
            _block(
                FILE_ID,
                "FileStoreBlock",
                "Downloads and stores a file from a URL, data URI, or local path.",
                klass="primitive",
                args=("file_in",),
                description=FILE_STORE_DESCRIPTION,
                tags=("file",),
            ),
            CapabilityEntry(
                id="tool:read_workspace_file",
                kind="tool",
                name="read_workspace_file",
                purpose="Read a file from persistent workspace.",
                description=(
                    "Read a file from persistent workspace. Use save_to_path "
                    "to copy to working dir for processing."
                ),
                tags=["file", "read", "workspace"],
                context="direct",
                implementations=[
                    Implementation(
                        kind="tool", ref="read_workspace_file", context="direct"
                    )
                ],
                argument_names=["path", "save_to_path"],
            ),
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


def test_description_past_the_listed_purpose_is_indexed_but_not_listed(index):
    """ "save file output" names FileStoreBlock only through the sentence
    ``clip_purpose`` drops; indexing the whole description finds it, and
    the listing the model reads stays the clipped purpose."""
    result = index.search("save file output")
    assert result.names[0] == "FileStoreBlock"
    assert "description" not in result.hits[0].entry.listing()
    assert result.hits[0].entry.listing()["purpose"] != FILE_STORE_DESCRIPTION


def test_save_query_reaches_a_store_entry_through_the_synonym(index):
    hits = index.search("save a file").hits
    store = next(h for h in hits if h.entry.name == "FileStoreBlock")
    assert store.coverage == 2.0  # "save" as written (description), "file"
    assert store.entry in [h.entry for h in hits[:2]]


def test_eager_tools_are_flagged_in_listing(index):
    hit = index.search("web search").hits[0]
    assert hit.entry.listing()["eager"] is True


# ------------------------------------------------- the per-session skill layer


def _skill(name: str, description: str) -> CapabilityEntry:
    return CapabilityEntry(
        id=f"skill:{name}",
        kind="skill",
        name=name,
        purpose=description,
        description=description,
        tags=["skill"],
        context="direct",
        implementations=[Implementation(kind="skill", ref=name, context="direct")],
    )


TRIAGE = _skill("triage-linear-issues", "Triage and prioritise a Linear issue.")
PLAYBOOK = _skill("issue-playbook", "Create an issue the way this team does.")


def test_with_entries_layers_skills_without_touching_the_base_index(index):
    layered = index.with_entries([TRIAGE])
    assert len(layered) == len(index) + 1
    assert layered.get(TRIAGE.id) is TRIAGE and index.get(TRIAGE.id) is None
    assert index.with_entries([]) is index


def test_a_skill_matching_as_well_as_a_block_ranks_with_connected_services(index):
    layered = index.with_entries([PLAYBOOK])
    # Nothing connected: the skill leads both unconnected issue blocks.
    assert layered.search("create issue").names[0] == "issue-playbook"
    state = ConnectionState(providers=frozenset({"github"}))
    top = layered.search("create issue", connections=state).names[:2]
    assert set(top) == {"issue-playbook", "GithubMakeIssueBlock"}
    assert layered.search("create issue").hits[0].coverage == 2.0


def test_skills_stay_in_the_main_list_of_a_service_query(index):
    layered = index.with_entries([TRIAGE])
    result = layered.search("linear issue")
    assert result.service == "linear"
    assert set(result.names) == {"LinearCreateIssueBlock", "triage-linear-issues"}
    assert "triage-linear-issues" not in [h.entry.name for h in result.fallback]


def test_skills_answer_to_the_read_skill_permission(index):
    layered = index.with_entries([TRIAGE])
    denied = CopilotPermissions(tools=["read_skill"])
    assert (
        "triage-linear-issues" not in layered.search("triage", permissions=denied).names
    )
    allowed = CopilotPermissions(tools=["web_search"])
    assert "triage-linear-issues" in layered.search("triage", permissions=allowed).names


def test_kind_filter_selects_skills(index):
    layered = index.with_entries([TRIAGE, PLAYBOOK])
    assert layered.search("issue", kind="skill").ids == [PLAYBOOK.id, TRIAGE.id]


def test_a_platform_entry_keeps_a_bare_ref_a_skill_shares(index):
    clone = _skill("web_search", "How this team searches the web.")
    layered = index.with_entries([clone])
    shared = layered.get("web_search")
    assert shared is not None and shared.kind == "tool"
    assert layered.get("skill:web_search") is clone
