"""Tests for the memory_search tool and its tiered fan-out."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode

from backend.copilot.graphiti import tiers
from backend.copilot.graphiti.client import derive_memory_group_id
from backend.copilot.graphiti.memory_model import MemoryEnvelope, MemoryKind, SourceKind
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.graphiti.tiers import MemoryTier, TierTarget
from backend.copilot.model import ChatSession
from backend.copilot.tools import graphiti_search as search_mod
from backend.copilot.tools.graphiti_search import MemorySearchTool
from backend.copilot.tools.models import ErrorResponse, MemorySearchResponse

_MODULE = "backend.copilot.tools.graphiti_search"
_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _enable_shared_memory_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "true")


def _edge(uuid: str, fact: str) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id="user_user-1",
        source_node_uuid="a",
        target_node_uuid="b",
        created_at=_NOW,
        name="relates",
        fact=fact,
        valid_at=_NOW,
        attributes={"status": "active"},
    )


def _episode(content: str) -> EpisodicNode:
    return EpisodicNode(
        name="ep",
        group_id="user_user-1",
        source=EpisodeType.text,
        source_description="chat",
        content=content,
        created_at=_NOW,
        valid_at=_NOW,
    )


async def _search(
    edges: list[EntityEdge],
    episodes: list[EpisodicNode],
    *,
    expert_id: str | None = None,
    still: tuple[list[EntityEdge], list[EpisodicNode]] | None = None,
    **tool_kwargs,
):
    """Run the tool with both recall reads, their recheck (``still``, or
    everything read) and the hit recorder mocked."""
    session = ChatSession.new("user-1", dry_run=False, expert_id=expert_id)
    search_facts = AsyncMock(return_value=edges)
    recent_episodes = AsyncMock(return_value=episodes)
    recheck = AsyncMock(return_value=still or (edges, episodes))
    record_recall = MagicMock(return_value="hit-coroutine")
    spawn = MagicMock()
    with (
        patch(f"{_MODULE}.search_facts", search_facts),
        patch(f"{_MODULE}.recent_episodes", recent_episodes),
        patch(f"{_MODULE}.recheck", recheck),
        patch(f"{_MODULE}.record_recall", record_recall),
        patch(f"{_MODULE}.spawn_background_task", spawn),
    ):
        result = await MemorySearchTool()._execute(
            "user-1", session, query="private fact", **tool_kwargs
        )
    return result, search_facts, recent_episodes, record_recall, spawn


@pytest.mark.asyncio
async def test_expert_session_searches_only_expert_memory_group() -> None:
    result, search_facts, recent_episodes, _, _ = await _search(
        [], [], expert_id="expert-1"
    )

    assert isinstance(result, MemorySearchResponse)
    expert_scope = MemoryScope.for_expert("user-1", "expert-1")
    search_facts.assert_awaited_once_with(expert_scope, "private fact", limit=15)
    recent_episodes.assert_awaited_once_with(expert_scope, 5)


@pytest.mark.asyncio
async def test_renders_facts_and_episodes() -> None:
    result, *_ = await _search(
        [_edge("e1", "Alice works on Atlas")], [_episode("talked about Atlas")]
    )

    assert isinstance(result, MemorySearchResponse)
    assert result.facts == [
        "Alice works on Atlas (valid: 2025-01-01 00:00:00+00:00 — present)"
    ]
    assert result.recent_episodes == ["[2025-01-01 00:00:00+00:00] talked about Atlas"]


@pytest.mark.asyncio
async def test_only_what_the_recheck_still_finds_is_shown_and_counted() -> None:
    """A forget that answered while the search ran: the last read before
    rendering (``recall_recheck.recheck``) no longer finds the fact or its
    episode, so neither is shown or counted as a hit."""
    kept = _edge("e2", "Bob leads Atlas")
    edges = [_edge("e1", "Alice works on Atlas"), kept]

    result, _, _, record_recall, _ = await _search(
        edges, [_episode("Alice works on Atlas")], still=([kept], [])
    )

    assert isinstance(result, MemorySearchResponse)
    assert [fact.split(" (")[0] for fact in result.facts] == ["Bob leads Atlas"]
    assert result.recent_episodes == []
    record_recall.assert_called_once_with(MemoryScope.for_user("user-1"), ["e2"])


@pytest.mark.asyncio
async def test_returned_facts_are_counted_as_hits() -> None:
    """memory_search feeds the ratification sweep: a tentative fact the
    model retrieves here has been used, like one warm context surfaced."""
    edges = [_edge("e1", "fact one"), _edge("e2", "fact two")]

    _, _, _, record_recall, spawn = await _search(edges, [])

    record_recall.assert_called_once_with(MemoryScope.for_user("user-1"), ["e1", "e2"])
    spawn.assert_called_once()
    assert spawn.call_args.args == ("hit-coroutine",)


@pytest.mark.asyncio
async def test_no_facts_no_hit_task() -> None:
    _, _, _, record_recall, spawn = await _search([], [_episode("just chatting")])

    record_recall.assert_not_called()
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_scope_filter_drops_other_scopes_even_for_long_envelopes() -> None:
    """A MemoryEnvelope longer than the 500-char display cut is still scoped
    by its full body, so a project memory cannot leak into a global search."""
    long_project = MemoryEnvelope(
        content="x" * 600,
        source_kind=SourceKind.user_asserted,
        scope="project:crm",
        memory_kind=MemoryKind.fact,
    ).model_dump_json()
    episodes = [_episode(long_project), _episode("plain conversation")]

    result, *_ = await _search([], episodes, scope="real:global")

    assert isinstance(result, MemorySearchResponse)
    assert result.recent_episodes == ["[2025-01-01 00:00:00+00:00] plain conversation"]


@pytest.mark.asyncio
async def test_search_failure_is_reported_not_raised() -> None:
    session = ChatSession.new("user-1", dry_run=False)
    with (
        patch(f"{_MODULE}.search_facts", AsyncMock(side_effect=RuntimeError("down"))),
        patch(f"{_MODULE}.recent_episodes", AsyncMock(return_value=[])),
    ):
        result = await MemorySearchTool()._execute("user-1", session, query="x")

    assert "temporarily unavailable" in result.message


# ---------------------------------------------------------------------------
# Tiered search fan-out
# ---------------------------------------------------------------------------


def _fact_edge(fact: str) -> EntityEdge:
    return _edge(f"edge-{fact}", fact)


def _tier_client(edges: list[EntityEdge]) -> AsyncMock:
    """A shared tier's Graphiti client: ``edges`` from the search, all of
    them active and in scope for the lifecycle filters' queries."""
    client = AsyncMock()
    client.search_.return_value = SimpleNamespace(edges=edges)
    client.retrieve_episodes.return_value = []
    client.driver = SimpleNamespace(
        execute_query=AsyncMock(
            side_effect=lambda _query, **params: (
                [{"uuid": uuid} for uuid in params["edge_ids"]],
                None,
                None,
            )
        )
    )
    return client


@contextmanager
def _personal_recall(edges: list[EntityEdge]) -> Iterator[AsyncMock]:
    """The personal tier's recall reads (``search_facts`` returning
    ``edges``, no episodes, a recheck that keeps everything) with the hit
    recorder stubbed; yields the ``search_facts`` mock."""
    search_facts = AsyncMock(return_value=edges)
    with (
        patch.object(search_mod, "search_facts", search_facts),
        patch.object(search_mod, "recent_episodes", AsyncMock(return_value=[])),
        patch.object(
            search_mod,
            "recheck",
            AsyncMock(side_effect=lambda _scope, facts, episodes: (facts, episodes)),
        ),
        patch.object(search_mod, "record_recall", MagicMock(return_value="hit")),
        patch.object(search_mod, "spawn_background_task", MagicMock()),
    ):
        yield search_facts


def _org_session() -> ChatSession:
    return ChatSession(
        session_id="s",
        user_id="user-1",
        title=None,
        messages=[],
        usage=[],
        credentials={},
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        organization_id="org-1",
    )


class TestMemorySearchTiers:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("expert_id", [None, "expert-1"])
    async def test_rollout_flip_keeps_private_memory_without_shared_guidance(
        self, monkeypatch: pytest.MonkeyPatch, expert_id: str | None
    ) -> None:
        tool = MemorySearchTool()
        session = ChatSession.new(
            "user-1",
            dry_run=False,
            expert_id=expert_id,
            organization_id="org-1",
            team_id="team-1",
        )
        private_group = derive_memory_group_id("user-1", expert_id)
        clients = {
            "org_org-1": _tier_client([_fact_edge("org fact")]),
            "team_team-1": _tier_client([_fact_edge("team fact")]),
        }

        # Reuse the tool and populated session across flag flips to catch stale
        # targets or shared guidance without replacing the real tier resolver.
        for enabled in (True, False, True):
            monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", str(enabled).lower())
            with (
                _personal_recall([_fact_edge("private fact")]) as search_facts,
                patch.object(
                    tiers, "is_org_member", AsyncMock(return_value=True)
                ) as org_access,
                patch.object(
                    tiers, "get_user_team_ids", AsyncMock(return_value=["team-1"])
                ) as team_access,
                patch.object(
                    tiers,
                    "resolve_team_names",
                    AsyncMock(return_value={"team-1": "Platform"}),
                ),
                patch.object(
                    search_mod,
                    "get_graphiti_client",
                    AsyncMock(side_effect=lambda group_id: clients[group_id]),
                ) as get_client,
            ):
                result = await tool._execute("user-1", session, query="fact")

            assert isinstance(result, MemorySearchResponse)
            assert any(fact.startswith("private fact") for fact in result.facts)
            assert ("'org memory'" in result.message) is enabled
            assert ("'team memory (<name>)'" in result.message) is enabled
            # The private memory is read through the recall policy, in the
            # same group whichever way the flag points.
            search_facts.assert_awaited_once()
            assert search_facts.await_args.args[0].group_id == private_group
            expected_groups = set(clients) if enabled else set()
            assert {
                call.args[0] for call in get_client.await_args_list
            } == expected_groups
            if enabled:
                assert any("[org memory] org fact" in fact for fact in result.facts)
                assert any(
                    "[team memory (Platform)] team fact" in fact
                    for fact in result.facts
                )
                org_access.assert_awaited_once()
                team_access.assert_awaited_once()
            else:
                assert len(result.facts) == 1
                org_access.assert_not_awaited()
                team_access.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_tiers_label_shared_results(self) -> None:
        tool = MemorySearchTool()
        targets = [
            TierTarget(group_id="user_user-1", tier=MemoryTier.personal, label=None),
            TierTarget(group_id="org_org-1", tier=MemoryTier.org, label="org memory"),
            TierTarget(
                group_id="team_team-1",
                tier=MemoryTier.team,
                label="team memory (Platform)",
                team_id="team-1",
            ),
        ]
        clients = {
            "org_org-1": _tier_client([_fact_edge("org fact")]),
            "team_team-1": _tier_client([_fact_edge("team fact")]),
        }

        async def _fake_client(group_id: str):
            return clients[group_id]

        with (
            _personal_recall([_fact_edge("personal fact")]) as search_facts,
            patch.object(
                search_mod,
                "resolve_search_targets",
                new_callable=AsyncMock,
                return_value=targets,
            ),
            patch.object(search_mod, "get_graphiti_client", side_effect=_fake_client),
        ):
            result = await tool._execute(
                user_id="user-1", session=_org_session(), query="q", tier="all"
            )

        assert isinstance(result, MemorySearchResponse)
        assert any("[org memory] org fact" in f for f in result.facts)
        assert any("[team memory (Platform)] team fact" in f for f in result.facts)
        # Personal fact is unlabelled.
        assert any(f.startswith("personal fact") for f in result.facts)
        # Every resolved (active-membership) tier group was queried, and only those.
        search_facts.assert_awaited_once()
        assert search_facts.await_args.kwargs["limit"] == 15
        for client in clients.values():
            client.search_.assert_awaited_once()
        clients["org_org-1"].retrieve_episodes.assert_not_awaited()
        assert (
            clients["org_org-1"].search_.await_args.kwargs["search_filter"].expired_at
            is not None
        )
        assert clients["org_org-1"].search_.await_args.kwargs["config"].limit == 50

    @pytest.mark.asyncio
    async def test_shared_tier_drops_non_active_edges(self) -> None:
        target = TierTarget(
            group_id="org_org-1", tier=MemoryTier.org, label="org memory"
        )
        active = _fact_edge("approved fact")
        rejected = _fact_edge("rejected fact")
        client = _tier_client([active, rejected])
        client.driver.execute_query.side_effect = None
        client.driver.execute_query.return_value = (
            [{"uuid": active.uuid}],
            None,
            None,
        )

        with (
            patch.object(
                search_mod,
                "resolve_search_targets",
                new_callable=AsyncMock,
                return_value=[target],
            ),
            patch.object(
                search_mod,
                "get_graphiti_client",
                new_callable=AsyncMock,
                return_value=client,
            ),
        ):
            result = await MemorySearchTool()._execute(
                "user-1", _org_session(), query="fact", tier="org"
            )

        assert isinstance(result, MemorySearchResponse)
        assert any("approved fact" in fact for fact in result.facts)
        assert all("rejected fact" not in fact for fact in result.facts)
        assert result.recent_episodes == []

    @pytest.mark.asyncio
    async def test_scope_hard_filters_relationship_facts(self) -> None:
        target = TierTarget(
            group_id="org_org-1", tier=MemoryTier.org, label="org memory"
        )
        atlas = _fact_edge("atlas-only fact")
        beacon = _fact_edge("beacon-only fact")
        client = _tier_client([atlas, beacon])

        async def filter_rows(_query, **params):
            ids = params["edge_ids"]
            if "scope" in params:
                ids = [atlas.uuid] if params["scope"] == "project:atlas" else []
            return ([{"uuid": uuid} for uuid in ids], None, None)

        client.driver.execute_query.side_effect = filter_rows

        with (
            patch.object(
                search_mod,
                "resolve_search_targets",
                new_callable=AsyncMock,
                return_value=[target],
            ),
            patch.object(
                search_mod,
                "get_graphiti_client",
                new_callable=AsyncMock,
                return_value=client,
            ),
        ):
            result = await MemorySearchTool()._execute(
                "user-1",
                _org_session(),
                query="fact",
                tier="org",
                scope="project:atlas",
            )

        assert isinstance(result, MemorySearchResponse)
        assert any("atlas-only fact" in fact for fact in result.facts)
        assert all("beacon-only fact" not in fact for fact in result.facts)

    @pytest.mark.asyncio
    async def test_tier_all_queries_only_active_membership_targets(self) -> None:
        """The tool queries exactly the groups resolve_search_targets returns
        — which is the active-membership source — never an arbitrary team."""
        tool = MemorySearchTool()
        # resolve_search_targets returns ONLY personal (simulating a user with
        # no active team memberships in the org).
        targets = [
            TierTarget(group_id="user_user-1", tier=MemoryTier.personal, label=None)
        ]
        get_client = AsyncMock()

        with (
            _personal_recall([_fact_edge("personal fact")]) as search_facts,
            patch.object(
                search_mod,
                "resolve_search_targets",
                new_callable=AsyncMock,
                return_value=targets,
            ) as resolve,
            patch.object(search_mod, "get_graphiti_client", get_client),
        ):
            result = await tool._execute(
                user_id="user-1", session=_org_session(), query="q", tier="all"
            )

        resolve.assert_awaited_once_with(
            "user-1",
            "org-1",
            "all",
            session_team_id=None,
            expert_id=None,
        )
        assert isinstance(result, MemorySearchResponse)
        search_facts.assert_awaited_once()
        get_client.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_team_tier_with_no_active_memberships_returns_no_memories(
        self,
    ) -> None:
        tool = MemorySearchTool()
        with patch.object(
            search_mod,
            "resolve_search_targets",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await tool._execute(
                user_id="user-1", session=_org_session(), query="q", tier="team"
            )

        assert isinstance(result, MemorySearchResponse)
        assert result.facts == []
        assert "No memories found" in result.message

    @pytest.mark.asyncio
    async def test_org_tier_without_org_returns_polite_error(self) -> None:
        tool = MemorySearchTool()
        session = ChatSession(
            session_id="s",
            user_id="user-1",
            title=None,
            messages=[],
            usage=[],
            credentials={},
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        result = await tool._execute(
            user_id="user-1", session=session, query="q", tier="org"
        )

        assert isinstance(result, ErrorResponse)
        assert "organization" in result.message.lower()
