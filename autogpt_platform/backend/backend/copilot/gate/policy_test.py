"""Tier assignment is the security boundary, so it is asserted, not assumed.

The classifier is allowed to be wrong; these sets are not. Every test here
reads as "would this still be safe if the classifier always said allow".
"""

from backend.copilot.gate.policy import (
    ALWAYS_ASK_TOOLS,
    DEFER_TOOLS,
    MCP_FILE_READ_TOOLS,
    MCP_FILE_WRITE_TOOLS,
    READ_TOOLS,
    TAINT_ESCALATES,
    TAINT_SOURCES,
    Tier,
    escalates_under_taint,
    tier_for,
)
from backend.copilot.tools import TOOL_REGISTRY


def test_every_registry_tool_is_tiered_exactly_once():
    names = set(TOOL_REGISTRY)
    for tier_set in (READ_TOOLS, ALWAYS_ASK_TOOLS, DEFER_TOOLS):
        assert tier_set <= names | MCP_FILE_READ_TOOLS

    assert not READ_TOOLS & ALWAYS_ASK_TOOLS
    assert not READ_TOOLS & DEFER_TOOLS
    assert not ALWAYS_ASK_TOOLS & DEFER_TOOLS


def test_unknown_tool_is_judged_not_allowed():
    """A tool added later must cost friction, never silence."""
    assert tier_for("a_tool_that_does_not_exist_yet") is Tier.JUDGED


def test_delegation_is_always_ask():
    """Each opens a new session with a fresh taint bit, and
    ``child_session_origin`` hands an interactive parent an interactive child —
    so the delegation itself is the last point a human can see it."""
    for tool in ("delegate_to_expert", "handoff_to_expert", "run_sub_session"):
        assert tier_for(tool) is Tier.ALWAYS_ASK


def test_opaque_argument_tools_are_never_classified():
    """Their args are bare identifiers; ``check_hitl_review`` gates them
    against the resolved block instead."""
    for tool in ("run_block", "run_agent", "continue_run_block"):
        assert tier_for(tool) is Tier.DEFER


def test_remote_semantics_tool_is_always_ask():
    assert tier_for("run_mcp_tool") is Tier.ALWAYS_ASK


def test_preset_update_cannot_route_around_webhook_setup():
    """Reactivating a preset re-registers its webhook and returns the ingress
    URL — the effect ``setup_agent_webhook_trigger`` is gated for."""
    assert tier_for("update_preset") is Tier.ALWAYS_ASK
    assert tier_for("setup_agent_webhook_trigger") is Tier.ALWAYS_ASK


def test_outward_and_destructive_tools_are_always_ask():
    for tool in (
        "post_to_chat_platform",
        "delete_workspace_file",
        "delete_preset",
        "delete_schedule",
        "delete_folder",
        "delete_skill",
        "memory_forget_confirm",
        "connect_integration",
    ):
        assert tier_for(tool) is Tier.ALWAYS_ASK


def test_existing_confirm_gates_are_never_auto_satisfied():
    """Auto mode must not become a way to satisfy a gate that already exists."""
    assert tier_for("confirm_expert_change") is Tier.ALWAYS_ASK
    assert tier_for("confirm_expert_soul_update") is Tier.ALWAYS_ASK


def test_mcp_file_tools_are_covered():
    """They never reach ``BaseTool.execute``; the second seam gates them."""
    for tool in MCP_FILE_WRITE_TOOLS:
        assert escalates_under_taint(tool)
    for tool in MCP_FILE_READ_TOOLS:
        assert tier_for(tool) is Tier.READ
        assert tool in TAINT_SOURCES


def test_memory_laundering_loop_is_closed_on_both_sides():
    """Injected text stored in one session is recalled in the next, so the
    writes must escalate under taint AND the readers must be taint sources."""
    for writer in ("memory_store", "add_understanding", "store_skill"):
        assert escalates_under_taint(writer)
    for reader in ("memory_search", "memory_forget_search", "read_skill"):
        assert tier_for(reader) is Tier.READ
        assert reader in TAINT_SOURCES


def test_scheduling_escalates_because_the_scheduled_run_is_ungated():
    assert escalates_under_taint("schedule_followup")


def test_reading_stays_free_after_taint():
    """Escalating these would make every post-fetch research turn a prompt,
    which is the nagging the feature exists to remove. Exfiltration by GET is
    a documented, accepted limit — see the plan, §5.3."""
    for tool in ("web_fetch", "web_search", "browser_navigate"):
        assert not escalates_under_taint(tool)


def test_taint_sources_cover_every_ingestion_path():
    for tool in ("web_fetch", "run_mcp_tool", "run_agent", "view_agent_output"):
        assert tool in TAINT_SOURCES


# Hand-kept ledger of tools whose effects outlive the conversation. Listed
# here rather than in policy.py because its only job is the completeness
# check below — a new effectful tool left at plain JUDGED should fail.
_EFFECTFUL = frozenset(
    {
        "bash_exec",
        "browser_act",
        "continue_run_block",
        "create_agent",
        "customize_agent",
        "delete_workspace_file",
        "edit_agent",
        "move_agents_to_folder",
        "post_to_chat_platform",
        "run_agent",
        "run_block",
        "run_mcp_tool",
        "run_sub_session",
        "schedule_followup",
        "setup_agent_webhook_trigger",
        "store_skill",
        "update_preset",
        "write_workspace_file",
    }
)


def test_every_effectful_tool_is_escalated_or_gated_another_way():
    """Plain JUDGED is not enough for a tool whose effects outlive the chat:
    it must escalate once the session has read untrusted content, be
    always-ask, or defer to a gate that sees more than a classifier can."""
    for tool in _EFFECTFUL:
        assert (
            tool in TAINT_ESCALATES
            or tier_for(tool) is Tier.ALWAYS_ASK
            or tier_for(tool) is Tier.DEFER
        ), tool


def test_read_tier_has_no_effectful_members():
    assert not READ_TOOLS & _EFFECTFUL
