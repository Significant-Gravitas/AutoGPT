from datetime import datetime, timezone

from backend.copilot.briefing.generate import (
    AgentInfo,
    compose_briefing,
    render_briefing_markdown,
)

NOW = datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)


def make_expert(id="exp-1", name="Ana", avatar="https://a/x.png"):
    # Build a minimal backend.api.features.experts.models.Expert
    from backend.api.features.experts.models import Expert

    return Expert(
        id=id,
        name=name,
        avatar_url=avatar,
        role="Researcher",
        tagline=None,
        bio=None,
        skills=[],
        identity="",
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )


def make_exec(
    id="run-1",
    graph_id="g-1",
    expert_id="exp-1",
    status="COMPLETED",
    summary="Found 3 leads",
):
    from unittest.mock import MagicMock

    m = MagicMock()
    m.id, m.graph_id, m.expert_id = id, graph_id, expert_id
    m.status = status
    m.stats = {"activity_status": summary} if summary else {}
    return m


def make_review(
    node_exec_id="ne-1",
    graph_exec_id="run-1",
    graph_id="g-1",
    instructions="Approve outreach email",
):
    from unittest.mock import MagicMock

    r = MagicMock()
    r.node_exec_id, r.graph_exec_id, r.instructions = (
        node_exec_id,
        graph_exec_id,
        instructions,
    )
    r.graph_id = graph_id
    return r


def test_nothing_to_say_returns_none():
    assert (
        compose_briefing(
            experts=[make_expert()],
            executions=[],
            reviews=[],
            agent_info_by_graph_id={},
            generated_at=NOW,
            tz_name="UTC",
        )
        is None
    )


def test_expert_runs_and_decisions():
    content = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec()],
        reviews=[make_review()],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content is not None
    assert content.zero_expert_fallback is False
    item = content.run_items[0]
    assert (item.expert_name, item.agent_name, item.summary) == (
        "Ana",
        "Lead Finder",
        "Found 3 leads",
    )
    assert item.link == "/library/agents/lib-1?executionId=run-1"
    decision = content.decision_items[0]
    assert decision.title == "Approve outreach email"
    assert decision.expert_name == "Ana"  # attributed via the run's expert_id
    assert decision.link == "/library/agents/lib-1?executionId=run-1"


def test_zero_experts_falls_back_to_plain_activity():
    content = compose_briefing(
        experts=[],
        executions=[make_exec(expert_id=None)],
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("My Agent", "lib-9")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content.zero_expert_fallback is True
    assert content.run_items[0].expert_id is None


def test_copilot_review_links_to_session():
    content = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec()],
        reviews=[make_review(graph_exec_id="copilot-session-abc123")],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content.decision_items[0].link == "/copilot?sessionId=abc123"


def test_markdown_has_three_sections_and_links():
    content = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec()],
        reviews=[make_review()],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    md = render_briefing_markdown(content)
    assert "What ran" in md and "What was found" in md and "Needs your decision" in md
    assert "(/library/agents/lib-1?executionId=run-1)" in md
