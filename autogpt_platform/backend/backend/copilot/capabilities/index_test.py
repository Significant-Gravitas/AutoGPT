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


# --- SECRT-2676: connected MCP and first-party tools must not lose to blocks ---

DISCORD_ID = "55555555-5555-5555-5555-555555555555"
DISCORD_READ_ID = "66666666-6666-6666-6666-666666666666"
LINEAR_MCP_URL = "https://mcp.linear.app/mcp"
SENTRY_MCP_URL = "https://mcp.sentry.dev/mcp"
SLACK_MCP_URL = "https://mcp.slack.com/mcp"


def _mcp(entry_id, name, purpose, url, tags):
    return CapabilityEntry(
        id=entry_id,
        kind="mcp_server",
        name=name,
        purpose=purpose,
        tags=tags,
        context="direct",
        implementations=[Implementation(kind="mcp_server", ref=url)],
        connection=Connection(required=True, key_type="server_url", key=url),
    )


@pytest.fixture
def rival_index() -> CapabilityIndex:
    """A block and a connected service competing for the same job."""
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
                DISCORD_ID,
                "SendDiscordMessageBlock",
                "Send a message to a Discord channel.",
                provider="discord",
                args=("channel", "message"),
            ),
            # Covers every concept of the Discord query, so it sets the best
            # coverage the lift is measured against. Without a block of this
            # shape the fixture cannot reproduce a connected family of blocks
            # being carried over the first-party tool.
            _block(
                DISCORD_READ_ID,
                "ReadDiscordMessagesBlock",
                "Read and post messages on a Discord channel.",
                provider="discord",
                args=("channel",),
            ),
            _mcp(
                "mcp:mcp.linear.app",
                "Linear",
                "Search issues, projects, and documents, with optional "
                "updates to your team's work.",
                LINEAR_MCP_URL,
                ["linear", "mcp", "linear", "mcp.linear.app"],
            ),
            _mcp(
                "mcp:mcp.sentry.dev",
                "Sentry",
                "Investigate errors and performance, triage issues.",
                SENTRY_MCP_URL,
                ["sentry", "mcp", "sentry", "mcp.sentry.dev"],
            ),
            CapabilityEntry(
                id="tool:read_workspace_file",
                kind="tool",
                name="read_workspace_file",
                purpose="Read a file from the agent workspace.",
                tags=["workspace", "file"],
                context="direct",
                implementations=[
                    Implementation(
                        kind="tool", ref="read_workspace_file", context="direct"
                    )
                ],
                argument_names=["path"],
            ),
            CapabilityEntry(
                id="tool:post_to_chat_platform",
                kind="tool",
                name="post_to_chat_platform",
                purpose=(
                    "Post to a linked chat platform (Discord, Slack, "
                    "Telegram or Microsoft Teams)."
                ),
                tags=["chat", "platform", "post"],
                context="direct",
                implementations=[
                    Implementation(
                        kind="tool", ref="post_to_chat_platform", context="direct"
                    )
                ],
                argument_names=["platform", "channel", "content"],
            ),
        ]
    )


def test_platform_tool_survives_a_service_query(rival_index):
    """Naming a service must not hide the first-party tool that does the job.

    The service restriction used to drop every tool, so this query could only
    ever return the Discord blocks.
    """
    result = rival_index.search("post a message to discord")
    assert result.service == "discord"
    assert "post_to_chat_platform" in result.names
    # It need not lead — an entry that covers every concept of the query may
    # legitimately rank above it — but it must beat the write block the bug
    # report watched it lose to.
    assert result.names.index("post_to_chat_platform") < result.names.index(
        "SendDiscordMessageBlock"
    )


def test_platform_tool_survives_even_when_the_block_provider_is_connected(rival_index):
    """A Discord credential may be a read-only OAuth the write block cannot
    use, so the tool stays listed even when every block reads connected.

    A connected family of blocks must not be carried over the tool by the
    coverage lift: ReadDiscordMessagesBlock sets the best coverage here, and
    lifting the connected 2.0 blocks to it once pushed the tool off the list.
    """
    state = ConnectionState(providers=frozenset({"discord"}))
    result = rival_index.search("post a message to discord", connections=state)
    assert "post_to_chat_platform" in result.names
    assert result.names.index("post_to_chat_platform") < result.names.index(
        "SendDiscordMessageBlock"
    )


def test_service_query_does_not_pull_in_unrelated_tools(rival_index):
    """The SECRT-2433 boundary again: a tool only joins a service list when it
    names that service, not merely because it shares a verb."""
    result = rival_index.search("read a discord message", connections=None)
    assert result.service == "discord"
    assert "read_workspace_file" not in result.names
    assert "post_to_chat_platform" in result.names


def test_connected_mcp_outranks_a_block_for_the_same_job(rival_index):
    state = ConnectionState(server_urls=frozenset({LINEAR_MCP_URL}))
    result = rival_index.search("create a linear issue", connections=state)
    assert result.names[0] == "Linear"
    assert result.hits[0].connected is True
    assert "LinearCreateIssueBlock" in result.names


def test_block_still_wins_when_nothing_is_connected(rival_index):
    """The SECRT-2433 boundary: with no connected alternative the block ranks
    exactly as it did before."""
    result = rival_index.search("create a linear issue", connections=ConnectionState())
    assert result.names[0] == "LinearCreateIssueBlock"


def test_an_exact_block_name_in_a_phrase_does_not_pin_it_over_a_connected_mcp(
    rival_index,
):
    """AutoPilot searches "Linear create issue", which is also the block's
    name.  With no Linear credential for the block, the connected server
    leads, and the block is still listed."""
    state = ConnectionState(server_urls=frozenset({LINEAR_MCP_URL}))
    result = rival_index.search("Linear create issue", connections=state)
    assert result.names[0] == "Linear"
    assert "LinearCreateIssueBlock" in result.names
    assert all(hit.reason == "search" for hit in result.hits)


def test_an_exact_block_name_keeps_its_pin_when_the_block_is_connected(rival_index):
    state = ConnectionState(
        providers=frozenset({"linear"}), server_urls=frozenset({LINEAR_MCP_URL})
    )
    result = rival_index.search("Linear create issue", connections=state)
    assert result.names[0] == "LinearCreateIssueBlock"
    assert result.hits[0].reason == "exact_name"


def test_a_query_spelling_the_class_name_keeps_the_pin(rival_index):
    state = ConnectionState(server_urls=frozenset({LINEAR_MCP_URL}))
    for query in ("LinearCreateIssueBlock", "linear create issue block"):
        result = rival_index.search(query, connections=state)
        assert result.names[0] == "LinearCreateIssueBlock", query
        assert result.hits[0].reason == "exact_name", query


def test_an_exact_block_name_keeps_its_pin_with_no_connected_alternative(
    rival_index,
):
    for state in (ConnectionState(), None):
        result = rival_index.search("Linear create issue", connections=state)
        assert result.names[0] == "LinearCreateIssueBlock"
        assert result.hits[0].reason == "exact_name"


def test_a_distant_connected_service_is_not_lifted(rival_index):
    """A connected server the query does not name competes on coverage like
    anything else: Sentry matches only "issue" here, so it must not displace
    the block that matches both words.  The query names no service, so
    nothing is filtered out before ranking and the lift itself is tested."""
    state = ConnectionState(server_urls=frozenset({SENTRY_MCP_URL}))
    result = rival_index.search("create issue", connections=state)
    assert result.service is None
    assert "Sentry" in result.names
    assert result.names[0] == "LinearCreateIssueBlock"


def test_a_connected_server_is_not_lifted_on_a_query_that_does_not_name_it(
    rival_index,
):
    """kcze's case on #15011: Linear matches "update" through "updates" in
    its description, half of a two-word query, and was lifted over the block
    that matches both words the moment it was connected."""
    sheets = _block(
        "77777777-7777-7777-7777-777777777777",
        "GoogleSheetsUpdateCellBlock",
        "Update a single cell in a Google Sheets spreadsheet.",
        provider="google_sheets",
        args=("spreadsheet_id", "cell", "value"),
        tags=("spreadsheet",),
    )
    index = rival_index.with_entries([sheets])
    state = ConnectionState(server_urls=frozenset({LINEAR_MCP_URL}))
    result = index.search("update spreadsheet", connections=state)
    assert result.service is None
    assert "Linear" in result.names
    assert result.names[0] == "GoogleSheetsUpdateCellBlock"
    unconnected = index.search("update spreadsheet", connections=None)
    assert unconnected.names[0] == "GoogleSheetsUpdateCellBlock"


def test_an_unrelated_connected_server_does_not_lead_a_service_less_query(
    rival_index,
):
    """A connected Slack server shares only "send" with "send email"; the
    email block that matches both words still leads.  The block is not
    called SendEmailBlock, which the query would pin as an exact name."""
    email = _block(
        "88888888-8888-8888-8888-888888888888",
        "GmailSendBlock",
        "Send an email from a Gmail account.",
        provider="google",
        args=("to", "subject", "body"),
        tags=("email",),
    )
    slack = _mcp(
        "mcp:mcp.slack.com",
        "Slack",
        "Send messages, search channels and read threads in Slack.",
        SLACK_MCP_URL,
        ["slack", "mcp", "slack", "mcp.slack.com"],
    )
    state = ConnectionState(server_urls=frozenset({SLACK_MCP_URL}))
    result = rival_index.with_entries([email, slack]).search(
        "send email", connections=state
    )
    assert result.service is None
    assert "Slack" in result.names
    assert result.names[0] == "GmailSendBlock"


def test_the_lift_stays_one_concept_wide_on_a_query_naming_the_server(
    rival_index,
):
    """Naming the service lets its connected server be lifted, but only past a
    one-concept gap: here the block matches two more words than Linear."""
    state = ConnectionState(server_urls=frozenset({LINEAR_MCP_URL}))
    result = rival_index.search("linear create issue title", connections=state)
    assert result.service == "linear"
    by_name = {hit.entry.name: hit.coverage for hit in result.hits}
    assert by_name["LinearCreateIssueBlock"] - by_name["Linear"] > 1
    assert result.names[0] == "LinearCreateIssueBlock"


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


def test_a_service_query_keeps_both_skills_and_platform_tools(rival_index):
    """Both layers that carry no service tag stay in a service query's main
    list: the owner's skill for the service and the first-party tool."""
    announce = _skill("discord-announce", "Post a release announcement to Discord.")
    result = rival_index.with_entries([announce]).search("post to discord")
    assert result.service == "discord"
    assert {"discord-announce", "post_to_chat_platform"} <= set(result.names)
    assert not {"discord-announce", "post_to_chat_platform"} & {
        h.entry.name for h in result.fallback
    }


# --- SECRT-2820: a kind filter must not hide the user's connected blocks ---

GMAIL_LIST_ID = "99999999-9999-9999-9999-999999999991"
GMAIL_READ_ID = "99999999-9999-9999-9999-999999999992"
GOOGLE = ConnectionState(providers=frozenset({"google"}))


def _tool(name, purpose, *, description="", tags=()):
    return CapabilityEntry(
        id=f"tool:{name}",
        kind="tool",
        name=name,
        purpose=purpose,
        description=description or purpose,
        tags=sorted({*name.split("_"), *tags}),
        context="direct",
        implementations=[Implementation(kind="tool", ref=name, context="direct")],
    )


CONNECT_INTEGRATION = _tool(
    "connect_integration",
    "Prompt the user to connect a required integration. Supported providers: "
    "'github'.",
    description=(
        "Prompt the user to connect a required integration. Supported "
        "providers: 'github'. ONLY call this tool for one of the supported "
        "providers listed above - do NOT call it for Google, Gmail, Slack, or "
        "any other provider not in the list."
    ),
)


@pytest.fixture
def gmail_index(rival_index) -> CapabilityIndex:
    """The 2901dcfb shape: Gmail is a family of Google blocks, and the tools
    that share a word with the query only by accident."""
    return rival_index.with_entries(
        [
            _block(
                GMAIL_LIST_ID,
                "GmailListLabelsBlock",
                "Retrieve all labels from a Gmail account for organising emails.",
                provider="google",
                tags=("email",),
            ),
            _block(
                GMAIL_READ_ID,
                "GmailReadBlock",
                "Read recent emails from a Gmail inbox.",
                provider="google",
                args=("query", "max_results"),
                tags=("email",),
            ),
            CONNECT_INTEGRATION,
            _tool("web_search", "Search the web for live info (news, recent docs)."),
            _tool("list_skills", "List the skills available here."),
            _tool(
                "create_feature_request",
                "File a feature request; the team follows up by email.",
            ),
        ]
    )


def test_a_kind_filter_surfaces_the_connected_blocks_it_hid(gmail_index):
    """2901dcfb: kind="tool" with Google connected returned eight unrelated
    tools and the model went looking for a Gmail MCP server.  The connected
    Gmail blocks come back beside the filtered list, with a count."""
    result = gmail_index.search(
        "Gmail list recent emails", kind="tool", connections=GOOGLE
    )
    assert all(hit.entry.kind == "tool" for hit in result.hits)
    hidden = [hit.entry.name for hit in result.other_kinds]
    assert "GmailReadBlock" in hidden
    assert all(hit.connected for hit in result.other_kinds)
    assert result.hidden_by_kind >= len(result.other_kinds) > 0


def test_a_kind_filter_that_empties_the_list_still_surfaces_other_kinds(
    rival_index,
):
    """Prod Sep 30: kind="mcp_server" for a service with no server in the
    catalog came back empty.  The blocks it hid are surfaced instead."""
    result = rival_index.search(
        "send discord message",
        kind="mcp_server",
        connections=ConnectionState(providers=frozenset({"discord"})),
    )
    assert result.hits == []
    assert "SendDiscordMessageBlock" in [h.entry.name for h in result.other_kinds]


def test_other_kinds_are_capped_at_three(gmail_index):
    state = ConnectionState(providers=frozenset({"google", "discord"}))
    result = gmail_index.search("email message", kind="skill", connections=state)
    assert result.hits == []
    assert len(result.other_kinds) == 3
    assert result.hidden_by_kind > 3


def test_a_by_name_kind_lookup_gets_no_other_kinds(gmail_index):
    """Most kind calls are deliberate lookups by name; they must not be told
    to search again."""
    result = gmail_index.search("web_search", kind="tool", connections=GOOGLE)
    assert result.names[0] == "web_search"
    assert result.other_kinds == [] and result.hidden_by_kind == 0


def test_an_unconnected_weaker_match_of_another_kind_is_not_surfaced(rival_index):
    """Only a connected entry, or one covering more of the query than the
    filtered list does, is worth a second search."""
    result = rival_index.search(
        "post to discord", kind="tool", connections=ConnectionState()
    )
    assert result.names[0] == "post_to_chat_platform"
    assert result.other_kinds == []


def test_no_kind_means_no_other_kinds(gmail_index):
    result = gmail_index.search("Gmail list recent emails", connections=GOOGLE)
    assert result.other_kinds == [] and result.hidden_by_kind == 0
    assert result.names[0] in {"GmailReadBlock", "GmailListLabelsBlock"}


def test_a_tool_does_not_lead_connected_blocks_on_a_query_naming_no_service(
    gmail_index,
):
    """"email" names no service; create_feature_request only ties on
    coverage, so the user's connected Gmail blocks lead.  #15011 put tools
    first for every query and moved 19 such queries."""
    result = gmail_index.search("email", connections=GOOGLE)
    assert result.service is None
    assert result.hits[0].entry.kind == "block"
    assert result.hits[0].connected is True
    assert "create_feature_request" in result.names


def test_a_negative_mention_does_not_pull_a_tool_into_a_service_query(gmail_index):
    """connect_integration says "do NOT call it for Google, Gmail, Slack":
    only a tool's name, purpose and tags count as naming the service."""
    slack = _mcp(
        "mcp:mcp.slack.com",
        "Slack",
        "Send messages, search channels and read threads in Slack.",
        SLACK_MCP_URL,
        ["slack", "mcp", "slack", "mcp.slack.com"],
    )
    result = gmail_index.with_entries([slack]).search("slack send message")
    assert result.service == "slack"
    assert "connect_integration" not in result.names
    assert "post_to_chat_platform" in result.names
