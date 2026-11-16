"""A ``kind`` filter lists what it hid (SECRT-2820).

Integrations such as Gmail are blocks, so ``kind="tool"`` dropped every one of
them before scoring and the model went looking for an MCP server off the
catalog. A filtered search now also runs without the filter and returns the
better or connected entries of other kinds it hid as ``other_kinds``.
"""

import pytest

from backend.copilot.permissions import CopilotPermissions

from .index import OTHER_KINDS_LIMIT, CapabilityIndex
from .models import CapabilityEntry, Connection, Implementation
from .ranking import ConnectionState

GOOGLE = ConnectionState(providers=frozenset({"google"}))


def _block(block_id: str, name: str, purpose: str, provider: str, tags=()):
    return CapabilityEntry(
        id=f"block:{block_id}",
        kind="block",
        name=name,
        purpose=purpose,
        tags=[provider, *tags],
        context="both",
        implementations=[Implementation(kind="block", ref=block_id)],
        connection=Connection(required=True, key_type="provider", key=provider),
    )


def _tool(name: str, purpose: str):
    return CapabilityEntry(
        id=f"tool:{name}",
        kind="tool",
        name=name,
        purpose=purpose,
        tags=sorted(set(name.split("_"))),
        context="direct",
        implementations=[Implementation(kind="tool", ref=name, context="direct")],
    )


GMAIL_BLOCKS = [
    _block(
        f"aaaaaaaa-aaaa-aaaa-aaaa-00000000000{n}",
        name,
        purpose,
        "google",
        tags=("email",),
    )
    for n, (name, purpose) in enumerate(
        [
            ("GmailReadBlock", "Read recent emails from a Gmail inbox."),
            ("GmailListLabelsBlock", "List the labels of a Gmail account."),
            ("GmailSendBlock", "Send an email from a Gmail account."),
            ("GmailReplyBlock", "Reply to an email in a Gmail thread."),
        ]
    )
]
TOOLS = [
    _tool("web_search", "Search the web for live info (news, recent docs)."),
    _tool("web_fetch", "Fetch a url and return its page text."),
    _tool("create_feature_request", "File a feature request; we follow up by email."),
    # Unrelated tools, so a word the Gmail family shares is still rare enough
    # for BM25 to weight it: in an index this small, a term in half of the
    # entries scores zero.
    *(_tool(f"filler_{n}", "Unrelated platform helper.") for n in range(6)),
]
DISCORD_BLOCK = _block(
    "bbbbbbbb-bbbb-bbbb-bbbb-000000000001",
    "SendDiscordMessageBlock",
    "Send a message to a Discord channel.",
    "discord",
)


@pytest.fixture
def index() -> CapabilityIndex:
    return CapabilityIndex([*GMAIL_BLOCKS, *TOOLS, DISCORD_BLOCK])


def test_a_kind_filter_lists_the_connected_blocks_it_hid(index):
    """Dev trace 2901dcfb: kind="tool" with Google connected listed only
    unrelated tools, and the model web-searched for a Gmail MCP server."""
    result = index.search("Gmail read recent emails", kind="tool", connections=GOOGLE)

    assert all(hit.entry.kind == "tool" for hit in result.hits)
    assert result.other_kinds[0].entry.name == "GmailReadBlock"
    assert all(hit.connected for hit in result.other_kinds)
    assert result.hidden_by_kind >= len(result.other_kinds) > 0


def test_a_kind_filter_that_empties_the_list_still_lists_other_kinds(index):
    result = index.search("discord", kind="mcp_server", connections=ConnectionState())

    assert result.hits == []
    assert [hit.entry.name for hit in result.other_kinds] == ["SendDiscordMessageBlock"]


def test_other_kinds_are_capped_and_the_count_is_not(index):
    result = index.search("gmail", kind="tool", connections=GOOGLE)

    assert len(result.other_kinds) == OTHER_KINDS_LIMIT
    assert result.hidden_by_kind == len(GMAIL_BLOCKS)


def test_the_hidden_count_is_independent_of_the_main_result_limit(index):
    result = index.search("gmail", kind="tool", connections=GOOGLE, limit=1)

    assert len(result.other_kinds) == OTHER_KINDS_LIMIT
    assert result.hidden_by_kind == len(GMAIL_BLOCKS)


def test_hidden_connected_blocks_still_respect_permissions(index):
    result = index.search(
        "gmail",
        kind="tool",
        connections=GOOGLE,
        permissions=CopilotPermissions(blocks=[b.name for b in GMAIL_BLOCKS]),
    )

    assert result.other_kinds == [] and result.hidden_by_kind == 0


def test_a_by_name_kind_lookup_gets_no_other_kinds(index):
    result = index.search("web_search", kind="tool", connections=GOOGLE)

    assert result.hits[0].entry.name == "web_search"
    assert result.other_kinds == [] and result.hidden_by_kind == 0


def test_no_kind_means_no_other_kinds(index):
    result = index.search("gmail", connections=GOOGLE)

    assert result.other_kinds == [] and result.hidden_by_kind == 0


def test_an_unconnected_weaker_match_is_not_listed(index):
    """create_feature_request covers the whole query; GmailSendBlock only
    "send" and "email", and nothing is connected."""
    result = index.search(
        "send email feature request", kind="tool", connections=ConnectionState()
    )

    assert result.hits[0].entry.name == "create_feature_request"
    assert result.other_kinds == [] and result.hidden_by_kind == 0


def test_a_connected_weaker_match_is_not_listed(index):
    """Review on #15011: "fetch a url" is fully answered by web_fetch, yet a
    connected Gmail block sharing one word was listed as hidden, and the
    message sent the model off to search again."""
    connected = _block(
        "cccccccc-cccc-cccc-cccc-000000000001",
        "GmailFetchBlock",
        "Fetch emails from Gmail.",
        "google",
    )
    result = index.with_entries([connected]).search(
        "fetch a url", kind="tool", connections=GOOGLE
    )

    assert result.hits[0].entry.name == "web_fetch"
    assert result.other_kinds == [] and result.hidden_by_kind == 0


def test_a_block_filter_does_not_hide_a_platform_tool_it_ties(index):
    """kind="block" must never leave the model with only blocks when a
    first-party tool, which needs no connection, does the job as well."""
    fetch_block = _block(
        "dddddddd-dddd-dddd-dddd-000000000001",
        "PageReaderBlock",
        "Fetch a url and return its page text.",
        "acme",
    )
    result = index.with_entries([fetch_block]).search(
        "fetch a url", kind="block", connections=ConnectionState()
    )

    assert result.hits[0].entry.name == "PageReaderBlock"
    assert [hit.entry.name for hit in result.other_kinds] == ["web_fetch"]
