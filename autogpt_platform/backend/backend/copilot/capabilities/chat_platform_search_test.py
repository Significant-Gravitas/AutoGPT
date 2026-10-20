"""post_to_chat_platform is found when a search names one chat platform.

Discord, Slack and Telegram each have their own blocks, so a query naming
one is a service query restricted to that service. The first-party tool posts
as the AutoGPT bot with no user credential, and must stay in that list rather
than leave the model with only the per-platform send blocks.
"""

import pytest

from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY, chat_platform
from backend.copilot.tools.chat_platform import (
    SUPPORTED_PLATFORMS,
    PostToChatPlatformTool,
)

from .index import CapabilityIndex
from .ranking import ConnectionState
from .registry import build_entries

TOOL = "post_to_chat_platform"
# How each platform is spelled in a request, and its legacy send block.
SPELLINGS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "discord": (("discord",), "SendDiscordMessageBlock"),
    "slack": (("slack",), "SendSlackMessageBlock"),
    "telegram": (("telegram",), "SendTelegramMessageBlock"),
    "teams": (("teams", "microsoft teams"), None),
}
QUERIES = [
    (platform, query.format(name=name), block)
    for platform, (names, block) in SPELLINGS.items()
    for name in names
    for query in ("{name}", "post a message to {name}", "send a {name} message")
]


def _index(tools) -> CapabilityIndex:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chat_platform, "_any_chat_platform_configured", lambda: True)
        return CapabilityIndex(build_entries(tools, TOOL_GROUPS))


@pytest.fixture(scope="module")
def index() -> CapabilityIndex:
    return _index(TOOL_REGISTRY)


def test_every_supported_platform_has_queries():
    assert set(SPELLINGS) == set(SUPPORTED_PLATFORMS)


@pytest.mark.parametrize("connected", [False, True], ids=["unlinked", "linked"])
@pytest.mark.parametrize("platform,query,block", QUERIES)
def test_naming_one_platform_finds_the_tool(index, platform, query, block, connected):
    connections = ConnectionState(
        providers=frozenset({platform}) if connected else frozenset()
    )
    names = index.search(query, connections=connections).names
    assert TOOL in names, f"{query!r} -> {names}"
    if block in names:
        assert names.index(TOOL) < names.index(block), f"{query!r} -> {names}"


def test_the_listing_says_it_needs_no_credential(index):
    """The model picks from the listed purpose; one that reads like any other
    integration loses to a block it has seen before."""
    entry = index.get(f"tool:{TOOL}")
    assert entry is not None
    assert "AutoGPT bot" in entry.purpose
    assert "no credential" in entry.purpose


@pytest.mark.parametrize(
    "query",
    ["post this report to our discord channel", "post a report to slack"],
)
def test_asking_to_post_a_report_lands_on_post_not_edit(index, query):
    """The edit tool once matched "report" through "is always reported" and
    outranked the tool that posts."""
    assert index.search(query).names[0] == TOOL


class _TerseTool(PostToChatPlatformTool):
    @property
    def description(self) -> str:
        return "Post a message."


@pytest.fixture(scope="module")
def terse_index() -> CapabilityIndex:
    return _index({**TOOL_REGISTRY, TOOL: _TerseTool()})


@pytest.mark.parametrize("platform", SUPPORTED_PLATFORMS)
def test_found_by_platform_whatever_the_description_says(terse_index, platform):
    """Each supported platform is indexed by name, so trimming the
    description cannot drop the tool from that platform's search."""
    names = terse_index.search(f"post to {platform}").names
    assert TOOL in names, f"{platform!r} -> {names}"
