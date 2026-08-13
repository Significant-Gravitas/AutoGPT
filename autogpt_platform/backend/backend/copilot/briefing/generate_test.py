from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from backend.copilot.briefing import generate as generate_module
from backend.copilot.briefing.generate import AgentInfo, compose_briefing
from backend.copilot.briefing.render import render_briefing_markdown
from backend.data.execution import ExecutionStatus, GraphExecutionMeta

NOW = datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def stub_narrative(monkeypatch):
    """Keep the delivery tests off the network.

    Returns the mock so a test can assert on it or hand it a narrative; the
    default (`None`) is the production fallback, so every other test here
    exercises the template-only briefing it already asserted on.
    """
    mock = AsyncMock(return_value=None)
    monkeypatch.setattr(generate_module, "compose_narrative", mock)
    return mock


def make_expert(id="exp-1", name="Ana", avatar="https://a/x.png", workflows=None):
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
        voice_preferences="",
        boundaries="",
        protected_soul_rules=[],
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=workflows or [],
    )


def make_exec(
    id="run-1",
    graph_id="g-1",
    expert_id="exp-1",
    status=ExecutionStatus.COMPLETED,
    summary="Found 3 leads",
):
    return GraphExecutionMeta(
        id=id,
        user_id="user-1",
        graph_id=graph_id,
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=status,
        started_at=NOW - timedelta(minutes=5),
        ended_at=NOW,
        expert_id=expert_id,
        stats=GraphExecutionMeta.Stats(activity_status=summary),
    )


def make_review(
    node_exec_id="ne-1",
    graph_exec_id="run-1",
    graph_id="g-1",
    instructions="Approve outreach email",
    expert_id=None,
    expert_name=None,
    expert_avatar_url=None,
):
    from unittest.mock import MagicMock

    r = MagicMock()
    r.node_exec_id, r.graph_exec_id, r.instructions = (
        node_exec_id,
        graph_exec_id,
        instructions,
    )
    r.graph_id = graph_id
    # Explicit None defaults matter: MagicMock auto-attributes are truthy and
    # would short-circuit compose_briefing's enriched-attribution preference.
    r.expert_id = expert_id
    r.expert_name = expert_name
    r.expert_avatar_url = expert_avatar_url
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
    assert item.link == "/library/agents/lib-1?activeTab=runs&activeItem=run-1"
    decision = content.decision_items[0]
    assert decision.title == "Approve outreach email"
    assert decision.expert_name == "Ana"  # attributed via the run's expert_id
    assert decision.link == "/library/agents/lib-1?activeTab=runs&activeItem=run-1"


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


def test_decision_prefers_enriched_expert_attribution():
    """A review raised inside an expert's copilot session carries attribution
    resolved by _enrich_pending_reviews; compose must not discard it."""
    content = compose_briefing(
        experts=[make_expert()],  # exp-1 "Ana" — not the enriched expert
        executions=[],
        reviews=[
            make_review(
                graph_exec_id="copilot-session-abc123",
                expert_id="exp-2",
                expert_name="Bob",
                expert_avatar_url="https://a/b.png",
            )
        ],
        agent_info_by_graph_id={},
        generated_at=NOW,
        tz_name="UTC",
    )
    decision = content.decision_items[0]
    assert (decision.expert_id, decision.expert_name) == ("exp-2", "Bob")
    assert decision.expert_avatar_url == "https://a/b.png"


def test_decision_backfills_display_fields_from_expert_id():
    """Enriched expert_id without name/avatar falls back to the local expert
    lookup for the display fields."""
    content = compose_briefing(
        experts=[make_expert()],
        executions=[],
        reviews=[make_review(expert_id="exp-1")],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    decision = content.decision_items[0]
    assert (decision.expert_id, decision.expert_name) == ("exp-1", "Ana")


def test_failed_runs_sort_before_completed():
    content = compose_briefing(
        experts=[make_expert()],
        executions=[
            make_exec(id="run-1", status=ExecutionStatus.COMPLETED),
            make_exec(id="run-2", status=ExecutionStatus.FAILED),
        ],
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content is not None
    assert [i.status for i in content.run_items] == ["FAILED", "COMPLETED"]


def test_run_items_capped_at_ten():
    executions = [
        make_exec(id=f"run-{i}", status=ExecutionStatus.COMPLETED) for i in range(12)
    ]
    content = compose_briefing(
        experts=[make_expert()],
        executions=executions,
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content is not None
    assert len(content.run_items) == 10


def test_review_without_agent_info_falls_back_to_library_link():
    content = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec()],
        reviews=[make_review(graph_id="g-unknown")],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content is not None
    assert content.decision_items[0].link == "/library"


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
    assert "(/library/agents/lib-1?activeTab=runs&activeItem=run-1)" in md


@pytest.mark.asyncio
async def test_generate_skips_when_flag_off(monkeypatch):
    from backend.copilot.briefing import generate

    async def flag_off(*a, **kw):
        return False

    monkeypatch.setattr(generate, "is_feature_enabled", flag_off)
    result = await generate.generate_and_deliver_briefing("user-1")
    assert result == {"status": "skipped", "reason": "flag_disabled"}


@pytest.mark.asyncio
async def test_generate_skips_when_already_delivered(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    monkeypatch.setattr(generate, "is_feature_enabled", AsyncMock(return_value=True))
    user = MagicMock()
    user.timezone = "UTC"
    monkeypatch.setattr(
        generate,
        "user_db",
        lambda: MagicMock(get_user_by_id=AsyncMock(return_value=user)),
    )
    delivered_record = MagicMock(delivered_at=datetime(2026, 8, 7, 9, 5))
    client = MagicMock(get_briefing_for_date=AsyncMock(return_value=delivered_record))
    monkeypatch.setattr(generate, "get_database_manager_async_client", lambda: client)

    result = await generate.generate_and_deliver_briefing("user-1")
    assert result == {"status": "skipped", "reason": "already_delivered"}


@pytest.mark.asyncio
async def test_generate_delivers_and_composes_briefing(monkeypatch):
    import uuid
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    fixed_now = datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)
    fake_datetime = MagicMock(wraps=datetime)
    fake_datetime.now.return_value = fixed_now
    monkeypatch.setattr(generate, "datetime", fake_datetime)

    monkeypatch.setattr(generate, "is_feature_enabled", AsyncMock(return_value=True))

    user = MagicMock()
    user.timezone = "UTC"
    monkeypatch.setattr(
        generate,
        "user_db",
        lambda: MagicMock(get_user_by_id=AsyncMock(return_value=user)),
    )

    briefing_record = MagicMock(id="briefing-1")
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=None),
        create_briefing=AsyncMock(return_value=briefing_record),
        append_plain_session_message=AsyncMock(return_value="session-1"),
        mark_briefing_delivered=AsyncMock(),
    )
    monkeypatch.setattr(generate, "get_database_manager_async_client", lambda: client)

    expert = make_expert()
    execution = make_exec()
    review = make_review()
    monkeypatch.setattr(
        generate,
        "experts_db",
        lambda: MagicMock(list_experts=AsyncMock(return_value=[expert])),
    )
    monkeypatch.setattr(
        generate,
        "execution_db",
        lambda: MagicMock(get_graph_executions=AsyncMock(return_value=[execution])),
    )
    monkeypatch.setattr(
        generate,
        "review_db",
        lambda: MagicMock(
            get_pending_reviews_for_user=AsyncMock(return_value=[review])
        ),
    )
    library_agent = MagicMock(graph_id="g-1", id="lib-1")
    library_agent.name = "Lead Finder"
    monkeypatch.setattr(
        generate,
        "library_db",
        lambda: MagicMock(
            get_library_agent_refs_by_graph_ids=AsyncMock(return_value=[library_agent])
        ),
    )

    result = await generate.generate_and_deliver_briefing("user-1")

    assert result == {
        "status": "delivered",
        "briefing_id": "briefing-1",
        "session_id": "session-1",
    }

    expected_content = compose_briefing(
        experts=[expert],
        executions=[execution],
        reviews=[review],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=fixed_now,
        tz_name="UTC",
    )
    assert expected_content is not None

    create_call = client.create_briefing.await_args
    assert create_call.args == (
        "user-1",
        fixed_now.date(),
        expected_content.model_dump(mode="json"),
    )

    expected_message_id = str(
        uuid.uuid5(
            generate._BRIEFING_NAMESPACE,
            f"morning-briefing:user-1:{fixed_now.date().isoformat()}",
        )
    )
    append_call = client.append_plain_session_message.await_args
    assert append_call.kwargs == {
        "user_id": "user-1",
        "content": render_briefing_markdown(expected_content),
        "message_id": expected_message_id,
        "metadata": {"kind": "morning_briefing", "briefing_id": "briefing-1"},
    }
    client.mark_briefing_delivered.assert_awaited_once_with("user-1", "briefing-1")


def _patch_generate_env(monkeypatch, generate, client):
    """Shared plumbing for the delivery-retry tests: fixed clock, flag on,
    UTC user, and the given database-manager client."""
    from unittest.mock import AsyncMock, MagicMock

    fixed_now = datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)
    fake_datetime = MagicMock(wraps=datetime)
    fake_datetime.now.return_value = fixed_now
    monkeypatch.setattr(generate, "datetime", fake_datetime)
    monkeypatch.setattr(generate, "is_feature_enabled", AsyncMock(return_value=True))
    user = MagicMock()
    user.timezone = "UTC"
    monkeypatch.setattr(
        generate,
        "user_db",
        lambda: MagicMock(get_user_by_id=AsyncMock(return_value=user)),
    )
    monkeypatch.setattr(generate, "get_database_manager_async_client", lambda: client)


@pytest.mark.asyncio
async def test_generate_retries_undelivered_briefing_without_recomposing(monkeypatch):
    """A stored-but-undelivered record (prior session post failed) is
    redelivered from its stored content — no data gathering, no recompose,
    no duplicate create."""
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    stored = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec()],
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert stored is not None
    record = MagicMock(
        id="briefing-1", delivered_at=None, content=stored.model_dump(mode="json")
    )
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=record),
        create_briefing=AsyncMock(),
        append_plain_session_message=AsyncMock(return_value="session-1"),
        mark_briefing_delivered=AsyncMock(),
    )
    _patch_generate_env(monkeypatch, generate, client)
    gather_mock = MagicMock(list_experts=AsyncMock(return_value=[]))
    monkeypatch.setattr(generate, "experts_db", lambda: gather_mock)

    result = await generate.generate_and_deliver_briefing("user-1")

    assert result == {
        "status": "delivered",
        "briefing_id": "briefing-1",
        "session_id": "session-1",
    }
    client.create_briefing.assert_not_awaited()
    gather_mock.list_experts.assert_not_awaited()
    append_call = client.append_plain_session_message.await_args
    assert append_call.kwargs["content"] == render_briefing_markdown(stored)
    client.mark_briefing_delivered.assert_awaited_once_with("user-1", "briefing-1")


@pytest.mark.asyncio
async def test_generate_failed_post_leaves_briefing_retryable(monkeypatch):
    """When the session post raises, the record must stay undelivered so the
    next run retries instead of skipping with already_delivered."""
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=None),
        create_briefing=AsyncMock(return_value=MagicMock(id="briefing-1")),
        append_plain_session_message=AsyncMock(
            side_effect=RuntimeError("session post failed")
        ),
        mark_briefing_delivered=AsyncMock(),
    )
    _patch_generate_env(monkeypatch, generate, client)
    monkeypatch.setattr(
        generate,
        "experts_db",
        lambda: MagicMock(list_experts=AsyncMock(return_value=[make_expert()])),
    )
    monkeypatch.setattr(
        generate,
        "execution_db",
        lambda: MagicMock(get_graph_executions=AsyncMock(return_value=[make_exec()])),
    )
    monkeypatch.setattr(
        generate,
        "review_db",
        lambda: MagicMock(get_pending_reviews_for_user=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        generate,
        "library_db",
        lambda: MagicMock(
            get_library_agent_refs_by_graph_ids=AsyncMock(return_value=[])
        ),
    )

    with pytest.raises(RuntimeError, match="session post failed"):
        await generate.generate_and_deliver_briefing("user-1")

    client.mark_briefing_delivered.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_keeps_library_link_when_workflow_has_no_library_agent_id(
    monkeypatch,
):
    from unittest.mock import AsyncMock, MagicMock

    from backend.api.features.experts.models import ExpertWorkflowRef
    from backend.copilot.briefing import generate

    fixed_now = datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)
    fake_datetime = MagicMock(wraps=datetime)
    fake_datetime.now.return_value = fixed_now
    monkeypatch.setattr(generate, "datetime", fake_datetime)

    monkeypatch.setattr(generate, "is_feature_enabled", AsyncMock(return_value=True))

    user = MagicMock()
    user.timezone = "UTC"
    monkeypatch.setattr(
        generate,
        "user_db",
        lambda: MagicMock(get_user_by_id=AsyncMock(return_value=user)),
    )

    briefing_record = MagicMock(id="briefing-1")
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=None),
        create_briefing=AsyncMock(return_value=briefing_record),
        append_plain_session_message=AsyncMock(return_value="session-1"),
        mark_briefing_delivered=AsyncMock(),
    )
    monkeypatch.setattr(generate, "get_database_manager_async_client", lambda: client)

    # The workflow shares "g-1" with the library agent below but carries no
    # library_agent_id of its own — the merge must not clobber the
    # library-derived id with None.
    workflow = ExpertWorkflowRef(
        id="wf-1",
        store_listing_version_id=None,
        library_agent_id=None,
        graph_id="g-1",
        name="Lead Finder Workflow",
        description=None,
    )
    expert = make_expert(workflows=[workflow])
    execution = make_exec()
    monkeypatch.setattr(
        generate,
        "experts_db",
        lambda: MagicMock(list_experts=AsyncMock(return_value=[expert])),
    )
    monkeypatch.setattr(
        generate,
        "execution_db",
        lambda: MagicMock(get_graph_executions=AsyncMock(return_value=[execution])),
    )
    monkeypatch.setattr(
        generate,
        "review_db",
        lambda: MagicMock(get_pending_reviews_for_user=AsyncMock(return_value=[])),
    )
    library_agent = MagicMock(graph_id="g-1", id="lib-1")
    library_agent.name = "Lead Finder"
    monkeypatch.setattr(
        generate,
        "library_db",
        lambda: MagicMock(
            get_library_agent_refs_by_graph_ids=AsyncMock(return_value=[library_agent])
        ),
    )

    result = await generate.generate_and_deliver_briefing("user-1")
    assert result["status"] == "delivered"

    content_dict = client.create_briefing.await_args.args[2]
    run_item = content_dict["run_items"][0]
    assert run_item["library_agent_id"] == "lib-1"
    assert run_item["link"] == "/library/agents/lib-1?activeTab=runs&activeItem=run-1"
    assert run_item["agent_name"] == "Lead Finder Workflow"


def test_run_items_carry_the_card_fields_home_reads():
    """The stored row is what /home anchors its card on, so the shared composer
    has to persist the split headline, timings and cost — not just the raw
    summary the markdown renderer needs."""
    execution = make_exec(summary="Filed 2 tickets. All were duplicates.")
    execution.stats = GraphExecutionMeta.Stats(
        activity_status="Filed 2 tickets. All were duplicates.", duration=12.5, cost=3
    )

    content = compose_briefing(
        experts=[make_expert()],
        executions=[execution],
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )

    assert content is not None
    item = content.run_items[0]
    assert item.summary == "Filed 2 tickets. All were duplicates."
    assert (item.title, item.detail) == ("Filed 2 tickets.", "All were duplicates.")
    assert (item.duration_seconds, item.cost_cents) == (12.5, 3)
    assert item.expert_role == "Researcher"


def test_markdown_escapes_untrusted_text_in_link_labels():
    """Agent names and review instructions can come from a marketplace agent;
    they must not be able to break out of the markdown link they sit in."""
    review = make_review(instructions="Approve [Trusted](https://evil.example) now")

    content = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec(summary="Found [a](https://evil.example)")],
        reviews=[review],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead [Finder]", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content is not None

    markdown = render_briefing_markdown(content)

    assert "[Trusted](https://evil.example)" not in markdown
    assert "Approve \\[Trusted\\]\\(https://evil.example\\) now" in markdown
    assert (
        "[Lead \\[Finder\\]](/library/agents/lib-1?activeTab=runs&activeItem=run-1)"
        in markdown
    )


def test_run_totals_survive_the_run_item_cap():
    """Home reports "N completed / N failed" off the stored row, so the counts
    have to describe the whole night rather than the 10 runs that fit."""
    executions = [
        make_exec(id=f"run-{i}", status=ExecutionStatus.COMPLETED) for i in range(12)
    ] + [make_exec(id="broke", status=ExecutionStatus.FAILED)]

    content = compose_briefing(
        experts=[make_expert()],
        executions=executions,
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )

    assert content is not None
    assert len(content.run_items) == 10
    assert (content.completed_total, content.failed_total) == (12, 1)


def test_decision_items_are_capped_and_the_overflow_is_summarized():
    """100 pending reviews must not become a 100-line assistant message —
    that message also rides along in every later LLM turn of the session."""
    reviews = [
        make_review(node_exec_id=f"ne-{i}", graph_exec_id=f"run-{i}") for i in range(25)
    ]

    content = compose_briefing(
        experts=[make_expert()],
        executions=[],
        reviews=reviews,
        agent_info_by_graph_id={},
        generated_at=NOW,
        tz_name="UTC",
    )

    assert content is not None
    assert len(content.decision_items) == 10
    assert content.decision_total == 25

    markdown = render_briefing_markdown(content)
    assert "**Needs your decision (25)**" in markdown
    assert "…and 15 more on your home page" in markdown


def test_markdown_renders_non_relative_link_targets_as_plain_text():
    """Targets are server-composed relative paths; anything else must not
    become a clickable link in the user's thread."""
    content = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec()],
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content is not None
    content.run_items[0].link = "javascript:alert(1)"

    markdown = render_briefing_markdown(content)

    assert "javascript:alert(1)" not in markdown
    assert "Lead Finder" in markdown


@pytest.mark.asyncio
async def test_generate_writes_recomposed_content_back_to_an_unreadable_row(
    monkeypatch,
):
    """The stored row and the posted message must not diverge: a row whose
    content no longer validates is rewritten, not just re-posted."""
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    fake_datetime = MagicMock(wraps=datetime)
    fake_datetime.now.return_value = NOW
    monkeypatch.setattr(generate, "datetime", fake_datetime)
    monkeypatch.setattr(generate, "is_feature_enabled", AsyncMock(return_value=True))

    user = MagicMock()
    user.timezone = "UTC"
    monkeypatch.setattr(
        generate,
        "user_db",
        lambda: MagicMock(get_user_by_id=AsyncMock(return_value=user)),
    )

    stale_record = MagicMock(id="briefing-1", delivered_at=None)
    stale_record.content = {"unexpected": "shape"}
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=stale_record),
        create_briefing=AsyncMock(),
        update_briefing_content=AsyncMock(),
        append_plain_session_message=AsyncMock(return_value="session-1"),
        mark_briefing_delivered=AsyncMock(),
    )
    monkeypatch.setattr(generate, "get_database_manager_async_client", lambda: client)

    monkeypatch.setattr(
        generate,
        "experts_db",
        lambda: MagicMock(list_experts=AsyncMock(return_value=[make_expert()])),
    )
    monkeypatch.setattr(
        generate,
        "execution_db",
        lambda: MagicMock(get_graph_executions=AsyncMock(return_value=[make_exec()])),
    )
    monkeypatch.setattr(
        generate,
        "review_db",
        lambda: MagicMock(get_pending_reviews_for_user=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        generate,
        "library_db",
        lambda: MagicMock(
            get_library_agent_refs_by_graph_ids=AsyncMock(return_value=[])
        ),
    )

    result = await generate.generate_and_deliver_briefing("user-1")

    assert result["status"] == "delivered"
    client.create_briefing.assert_not_awaited()
    update_call = client.update_briefing_content.await_args
    assert update_call.args[:2] == ("user-1", "briefing-1")
    assert update_call.args[2]["run_items"][0]["agent_name"] == "Agent task"


@pytest.mark.asyncio
async def test_generate_stops_reprocessing_an_unreadable_row_with_nothing_to_say(
    monkeypatch,
):
    """Otherwise every future run re-gathers and re-composes the same row."""
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    fake_datetime = MagicMock(wraps=datetime)
    fake_datetime.now.return_value = NOW
    monkeypatch.setattr(generate, "datetime", fake_datetime)
    monkeypatch.setattr(generate, "is_feature_enabled", AsyncMock(return_value=True))

    user = MagicMock()
    user.timezone = "UTC"
    monkeypatch.setattr(
        generate,
        "user_db",
        lambda: MagicMock(get_user_by_id=AsyncMock(return_value=user)),
    )

    stale_record = MagicMock(id="briefing-1", delivered_at=None)
    stale_record.content = {"unexpected": "shape"}
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=stale_record),
        append_plain_session_message=AsyncMock(),
        mark_briefing_delivered=AsyncMock(),
    )
    monkeypatch.setattr(generate, "get_database_manager_async_client", lambda: client)

    monkeypatch.setattr(
        generate,
        "experts_db",
        lambda: MagicMock(list_experts=AsyncMock(return_value=[make_expert()])),
    )
    monkeypatch.setattr(
        generate,
        "execution_db",
        lambda: MagicMock(get_graph_executions=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        generate,
        "review_db",
        lambda: MagicMock(get_pending_reviews_for_user=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        generate,
        "library_db",
        lambda: MagicMock(
            get_library_agent_refs_by_graph_ids=AsyncMock(return_value=[])
        ),
    )

    result = await generate.generate_and_deliver_briefing("user-1")

    assert result == {"status": "skipped", "reason": "nothing_to_say"}
    client.mark_briefing_delivered.assert_awaited_once_with("user-1", "briefing-1")
    client.append_plain_session_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_bounds_the_execution_query(monkeypatch):
    """The 24h read is filtered and limited in the query — a user running
    minute-crons would otherwise ship thousands of rows over the RPC."""
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    fake_datetime = MagicMock(wraps=datetime)
    fake_datetime.now.return_value = NOW
    monkeypatch.setattr(generate, "datetime", fake_datetime)
    monkeypatch.setattr(generate, "is_feature_enabled", AsyncMock(return_value=True))

    user = MagicMock()
    user.timezone = "UTC"
    monkeypatch.setattr(
        generate,
        "user_db",
        lambda: MagicMock(get_user_by_id=AsyncMock(return_value=user)),
    )
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=None),
        create_briefing=AsyncMock(return_value=MagicMock(id="briefing-1")),
        append_plain_session_message=AsyncMock(return_value="session-1"),
        mark_briefing_delivered=AsyncMock(),
    )
    monkeypatch.setattr(generate, "get_database_manager_async_client", lambda: client)

    get_graph_executions = AsyncMock(return_value=[make_exec()])
    monkeypatch.setattr(
        generate,
        "experts_db",
        lambda: MagicMock(list_experts=AsyncMock(return_value=[make_expert()])),
    )
    monkeypatch.setattr(
        generate,
        "execution_db",
        lambda: MagicMock(get_graph_executions=get_graph_executions),
    )
    monkeypatch.setattr(
        generate,
        "review_db",
        lambda: MagicMock(get_pending_reviews_for_user=AsyncMock(return_value=[])),
    )
    refs_lookup = AsyncMock(return_value=[])
    monkeypatch.setattr(
        generate,
        "library_db",
        lambda: MagicMock(get_library_agent_refs_by_graph_ids=refs_lookup),
    )

    await generate.generate_and_deliver_briefing("user-1")

    kwargs = get_graph_executions.await_args.kwargs
    assert kwargs["statuses"] == [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
    assert kwargs["limit"] == generate._EXECUTION_FETCH_LIMIT
    # Only the graphs actually referenced are resolved — no whole-library page.
    assert refs_lookup.await_args.args == ("user-1", ["g-1"])


def _patch_fresh_compose_env(monkeypatch, generate, client):
    """Delivery plumbing for a user with one expert, one run and no reviews."""
    from unittest.mock import AsyncMock, MagicMock

    _patch_generate_env(monkeypatch, generate, client)
    monkeypatch.setattr(
        generate,
        "experts_db",
        lambda: MagicMock(list_experts=AsyncMock(return_value=[make_expert()])),
    )
    monkeypatch.setattr(
        generate,
        "execution_db",
        lambda: MagicMock(get_graph_executions=AsyncMock(return_value=[make_exec()])),
    )
    monkeypatch.setattr(
        generate,
        "review_db",
        lambda: MagicMock(get_pending_reviews_for_user=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        generate,
        "library_db",
        lambda: MagicMock(
            get_library_agent_refs_by_graph_ids=AsyncMock(return_value=[])
        ),
    )


@pytest.mark.asyncio
async def test_narrative_is_persisted_and_posted(monkeypatch, stub_narrative):
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    stub_narrative.return_value = "I checked your leads overnight."
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=None),
        create_briefing=AsyncMock(return_value=MagicMock(id="briefing-1")),
        append_plain_session_message=AsyncMock(return_value="session-1"),
        mark_briefing_delivered=AsyncMock(),
    )
    _patch_fresh_compose_env(monkeypatch, generate, client)

    await generate.generate_and_deliver_briefing("user-1")

    stored = client.create_briefing.await_args.args[2]
    assert stored["narrative"] == "I checked your leads overnight."
    posted = client.append_plain_session_message.await_args.kwargs["content"]
    assert "I checked your leads overnight." in posted


@pytest.mark.asyncio
async def test_narrative_failure_still_delivers_the_template_briefing(
    monkeypatch, stub_narrative
):
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    stub_narrative.return_value = None
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=None),
        create_briefing=AsyncMock(return_value=MagicMock(id="briefing-1")),
        append_plain_session_message=AsyncMock(return_value="session-1"),
        mark_briefing_delivered=AsyncMock(),
    )
    _patch_fresh_compose_env(monkeypatch, generate, client)

    result = await generate.generate_and_deliver_briefing("user-1")

    assert result["status"] == "delivered"
    assert client.create_briefing.await_args.args[2]["narrative"] is None
    assert (
        "**What ran**"
        in client.append_plain_session_message.await_args.kwargs["content"]
    )


@pytest.mark.asyncio
async def test_narrative_is_not_regenerated_when_redelivering(
    monkeypatch, stub_narrative
):
    """A stored row is reposted verbatim — no second LLM call, same paragraph."""
    from unittest.mock import AsyncMock, MagicMock

    from backend.copilot.briefing import generate

    stored = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec()],
        reviews=[],
        agent_info_by_graph_id={},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert stored is not None
    stored = stored.model_copy(update={"narrative": "Yesterday's paragraph."})
    record = MagicMock(id="briefing-1", delivered_at=None)
    record.content = stored.model_dump(mode="json")
    client = MagicMock(
        get_briefing_for_date=AsyncMock(return_value=record),
        create_briefing=AsyncMock(),
        update_briefing_content=AsyncMock(),
        append_plain_session_message=AsyncMock(return_value="session-1"),
        mark_briefing_delivered=AsyncMock(),
    )
    _patch_generate_env(monkeypatch, generate, client)

    await generate.generate_and_deliver_briefing("user-1")

    stub_narrative.assert_not_awaited()
    posted = client.append_plain_session_message.await_args.kwargs["content"]
    assert "Yesterday's paragraph." in posted


def test_narrative_renders_above_the_outcomes_and_stays_escaped():
    content = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec()],
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content is not None
    content = content.model_copy(
        update={"narrative": "I found [3 leads](/evil) — see *below*."}
    )

    markdown = render_briefing_markdown(content)

    lines = markdown.splitlines()
    assert lines[2] == "I found \\[3 leads\\]\\(/evil\\) — see \\*below\\*."
    assert lines.index("**What ran**") > 2


def test_briefing_without_a_narrative_renders_unchanged():
    content = compose_briefing(
        experts=[make_expert()],
        executions=[make_exec()],
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("Lead Finder", "lib-1")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert content is not None
    assert content.narrative is None
    assert render_briefing_markdown(content).splitlines()[2] == "**What ran**"
