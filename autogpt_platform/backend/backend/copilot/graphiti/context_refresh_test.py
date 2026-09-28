"""Tests for the follow-up warm-context refresh (``context_refresh.py``)."""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchMethod,
    SearchResults,
)
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER

from . import context, context_refresh, recall
from .context import fetch_warm_context
from .context_refresh import (
    REFRESH_RECIPE,
    join_refresh,
    refresh_warm_context,
    should_refresh_warm_context,
    start_refresh,
)
from .scope import MemoryScope

_NOW = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _edge(uuid: str = "edge-a", fact: str = "user likes python") -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id="user_abc",
        source_node_uuid="user",
        target_node_uuid="python",
        created_at=_NOW,
        name="preference",
        fact=fact,
        valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        attributes={"status": "active"},
    )


def _episode(content: str) -> EpisodicNode:
    return EpisodicNode(
        name="ep",
        group_id="user_abc",
        source=EpisodeType.text,
        source_description="chat",
        content=content,
        created_at=_NOW,
        valid_at=_NOW,
    )


class TestShouldRefreshWarmContext:
    """Pure deterministic cost gate for follow-up refreshes."""

    def test_empty_message_is_skipped(self) -> None:
        assert should_refresh_warm_context("") is False
        assert should_refresh_warm_context(None) is False

    def test_short_acknowledgement_is_skipped(self) -> None:
        assert should_refresh_warm_context("ok") is False
        assert should_refresh_warm_context("yes thanks") is False

    def test_word_count_boundary(self) -> None:
        # Pin the exact threshold so an accidental off-by-one (2 or 4) fails.
        assert should_refresh_warm_context("one two") is False
        assert should_refresh_warm_context("one two three") is True

    def test_terse_task_starts_are_not_excluded(self) -> None:
        """Cases 3-5 in SECRT-2378 are task STARTS mid-session, and they are
        phrased tersely. A threshold that skips these skips the bug."""
        assert should_refresh_warm_context("restart the executor") is True
        assert should_refresh_warm_context("deploy prod now") is True
        assert should_refresh_warm_context("resume the migration") is True

    def test_substantive_message_triggers_refresh(self) -> None:
        assert should_refresh_warm_context("deploy the staging environment now") is True

    def test_cjk_message_without_whitespace_triggers_refresh(self) -> None:
        # Japanese/Chinese don't separate words with spaces — str.split() would
        # score 1 and never pass. Each ideograph counts as a signal unit.
        assert should_refresh_warm_context("会議の予定を教えて") is True  # >= 3 chars
        assert should_refresh_warm_context("明日の東京の天気") is True
        # A one-ideograph reply still reads as trivial.
        assert should_refresh_warm_context("はい") is False

    def test_thai_and_hangul_ranges_are_covered(self) -> None:
        """Thai and Hangul are in _is_unspaced_script's ranges but were only
        covered by the CJK/kana cases — a regression narrowing either range
        would silently disable refresh for those users and still pass CI."""
        assert should_refresh_warm_context("ประชุมพรุ่งนี้") is True
        assert should_refresh_warm_context("내일 회의 일정 알려줘") is True
        assert should_refresh_warm_context("네") is False

    def test_mixed_script_units_are_summed_not_double_counted(self) -> None:
        """The two counts are additive: ideographs individually, everything
        else by whitespace. A mixed message straddling the threshold is where
        an off-by-one or a double-count would show up, and neither
        single-script case above can catch it."""
        # 1 latin word + 1 ideograph = 2 units — just under.
        assert should_refresh_warm_context("restart 東") is False
        # 1 latin word + 2 ideographs = 3 units — just over.
        assert should_refresh_warm_context("restart 東京") is True
        # Double counting the ideographs (once individually, once as a
        # whitespace word) would push this to 4 and wrongly pass.
        assert should_refresh_warm_context("東") is False


class TestRefreshWarmContext:
    """Follow-up refresh uses the cheap RRF recipe and honours the gate."""

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_user_id(self) -> None:
        with patch.object(
            context_refresh, "fetch_warm_context", new_callable=AsyncMock
        ) as mock_fetch:
            result = await refresh_warm_context("", "deploy the staging environment")
        assert result is None
        mock_fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_trivial_message_skips_fetch(self) -> None:
        with patch.object(
            context_refresh, "fetch_warm_context", new_callable=AsyncMock
        ) as mock_fetch:
            result = await refresh_warm_context("user-abc", "ok")
        assert result is None
        mock_fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_substantive_message_fetches_without_cross_encoder(self) -> None:
        with patch.object(
            context_refresh,
            "fetch_warm_context",
            new_callable=AsyncMock,
            return_value="<temporal_context>x</temporal_context>",
        ) as mock_fetch:
            result = await refresh_warm_context(
                "user-abc", "deploy the staging environment now"
            )
        assert result == "<temporal_context>x</temporal_context>"
        mock_fetch.assert_awaited_once()
        kwargs = mock_fetch.await_args.kwargs
        assert kwargs["recipe"] is REFRESH_RECIPE
        # No ratification hits on a refresh: it has no classifier.
        assert kwargs["ratify"] is False
        # Refresh must use the tighter budget, not the 8s first-turn timeout.
        assert (
            kwargs["timeout"] == context_refresh.graphiti_config.context_refresh_timeout
        )

    @pytest.mark.asyncio
    async def test_force_bypasses_gate_for_trivial_message(self) -> None:
        """A post-compaction turn (force=True) refreshes even on a short message."""
        with patch.object(
            context_refresh,
            "fetch_warm_context",
            new_callable=AsyncMock,
            return_value="<temporal_context>x</temporal_context>",
        ) as mock_fetch:
            result = await refresh_warm_context("user-abc", "go on", force=True)
        assert result is not None
        mock_fetch.assert_awaited_once()
        assert mock_fetch.await_args.kwargs["recipe"] is REFRESH_RECIPE

    @pytest.mark.asyncio
    async def test_refresh_is_scoped_to_the_expert_graph(self) -> None:
        """An expert chat must refresh from the EXPERT's memory, not the
        user's personal graph — otherwise the follow-up turn recalls facts the
        first turn deliberately never saw."""
        with patch.object(
            context_refresh,
            "fetch_warm_context",
            new_callable=AsyncMock,
            return_value="<temporal_context>x</temporal_context>",
        ) as mock_fetch:
            await refresh_warm_context(
                "user-abc",
                "deploy the staging environment now",
                expert_id="expert-1",
            )
        assert mock_fetch.await_args.args[2] == "expert-1"

    @pytest.mark.asyncio
    async def test_expert_id_reaches_the_group_id_derivation(self) -> None:
        """End-to-end through the real ``fetch_warm_context`` and
        ``recall.search_facts``: the expert must select the graph, so a
        scoping regression can't hide behind a mock of the layer that builds
        the ``MemoryScope``."""
        mock_client = AsyncMock()
        mock_client.search_.return_value = SearchResults(edges=[])

        with (
            patch.object(
                recall,
                "get_graphiti_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ) as mock_get_client,
            patch.object(
                context, "recent_episodes", new_callable=AsyncMock, return_value=[]
            ) as mock_recent,
        ):
            await refresh_warm_context(
                "user-abc",
                "deploy the staging environment now",
                expert_id="expert-1",
            )

        expert_scope = MemoryScope.for_expert("user-abc", "expert-1")
        mock_get_client.assert_awaited_once_with(expert_scope.group_id)
        search = mock_client.search_.await_args
        assert search is not None
        assert search.kwargs["group_ids"] == [expert_scope.group_id]
        mock_recent.assert_awaited_once_with(expert_scope, 5)


class TestRefreshTimeoutIsApplied:
    """The refresh's tighter budget must be APPLIED, not merely passed."""

    @pytest.mark.asyncio
    async def test_refresh_budget_bounds_a_slow_fetch(self, monkeypatch) -> None:
        monkeypatch.setattr(context.graphiti_config, "context_refresh_timeout", 0.05)
        monkeypatch.setattr(context.graphiti_config, "context_timeout", 30.0)

        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(5)
            return "<temporal_context>too late</temporal_context>"

        with patch.object(context, "_fetch", slow_fetch):
            result = await refresh_warm_context(
                "user-abc", "deploy the staging environment now"
            )

        # The 30s first-turn budget would have hung here; the refresh budget
        # cuts it off and degrades to None like any other retrieval failure.
        assert result is None


_BLOCK = "<temporal_context>fresh</temporal_context>"
# How much later than the grace a join may return: the event loop's timer
# granularity (about 16 ms on Windows) and the task's cancellation.
_JOIN_SLACK_S = 0.25


def _refresh_taking(seconds: float):
    async def _refresh(*_args, **_kwargs):
        await asyncio.sleep(seconds)
        return _BLOCK

    return _refresh


class TestJoinGrace:
    """``warm_context_refresh_join_grace_ms`` is the most a follow-up refresh
    may add to time-to-first-token: the join waits at most that long."""

    @pytest.mark.asyncio
    async def test_start_skips_a_turn_the_refresh_would_skip(self) -> None:
        with patch.object(
            context_refresh, "refresh_warm_context", new=AsyncMock()
        ) as mock:
            assert start_refresh(None, "deploy the staging environment now") is None
            assert start_refresh("user-abc", "ok") is None
            forced = start_refresh("user-abc", "ok", force=True)
            assert forced is not None
            await forced.task

        mock.assert_awaited_once_with("user-abc", "ok", expert_id=None, force=True)

    @pytest.mark.asyncio
    async def test_a_refresh_done_within_the_grace_is_returned(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            context_refresh.graphiti_config, "warm_context_refresh_join_grace_ms", 500
        )
        with patch.object(
            context_refresh, "refresh_warm_context", new=_refresh_taking(0)
        ):
            pending = start_refresh("user-abc", "deploy the staging environment now")
            assert pending is not None
            assert await join_refresh(pending) == _BLOCK

    @pytest.mark.asyncio
    async def test_a_late_refresh_is_cancelled_and_logged_after_the_grace(
        self, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(
            context_refresh.graphiti_config, "warm_context_refresh_join_grace_ms", 100
        )
        caplog.set_level(logging.INFO, logger=context_refresh.__name__)
        with patch.object(
            context_refresh, "refresh_warm_context", new=_refresh_taking(60)
        ):
            pending = start_refresh("user-abc", "deploy the staging environment now")
            assert pending is not None
            loop = asyncio.get_running_loop()
            joined = loop.time()
            out = await join_refresh(pending)
            waited = loop.time() - joined
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending.task, timeout=1)

        assert out is None
        assert 0.05 <= waited <= 0.1 + _JOIN_SLACK_S
        late = [r for r in caplog.records if "refresh late, skipped" in r.message]
        assert len(late) == 1 and late[0].levelno == logging.INFO
        assert "join grace 100 ms" in late[0].message

    @pytest.mark.asyncio
    async def test_time_before_the_join_is_the_refreshs_own(self, monkeypatch) -> None:
        """A refresh started before the query build had the build's time as
        well as the grace: 300 ms of work, joined 250 ms in with a 150 ms
        grace, is on time. The same refresh started at the join is late."""
        monkeypatch.setattr(
            context_refresh.graphiti_config, "warm_context_refresh_join_grace_ms", 150
        )
        with patch.object(
            context_refresh, "refresh_warm_context", new=_refresh_taking(0.3)
        ):
            early = start_refresh("user-abc", "deploy the staging environment now")
            assert early is not None
            await asyncio.sleep(0.25)
            assert await join_refresh(early) == _BLOCK

            at_join = start_refresh("user-abc", "deploy the staging environment now")
            assert at_join is not None
            assert await join_refresh(at_join) is None

    @pytest.mark.asyncio
    async def test_a_cancelled_join_cancels_the_refresh(self, monkeypatch) -> None:
        """The turn can be cancelled while it waits (the client went away):
        the refresh must not outlive it."""
        monkeypatch.setattr(
            context_refresh.graphiti_config, "warm_context_refresh_join_grace_ms", 5000
        )
        with patch.object(
            context_refresh, "refresh_warm_context", new=_refresh_taking(60)
        ):
            pending = start_refresh("user-abc", "deploy the staging environment now")
            assert pending is not None
            join = asyncio.create_task(join_refresh(pending))
            await asyncio.sleep(0.05)
            join.cancel()
            with pytest.raises(asyncio.CancelledError):
                await join
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending.task, timeout=1)


class TestRefreshRecipe:
    """The refresh swaps the reranker to RRF but keeps the first turn's
    search methods."""

    @pytest.mark.asyncio
    async def test_the_refresh_searches_with_rrf_and_keeps_bfs(self) -> None:
        """Through the real ``fetch_warm_context`` and ``recall.search_facts``,
        so what is asserted is the config graphiti actually receives, limit
        included."""
        mock_client = AsyncMock()
        mock_client.search_.return_value = SearchResults(edges=[])

        with (
            patch.object(
                recall,
                "get_graphiti_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch.object(
                context, "recent_episodes", new_callable=AsyncMock, return_value=[]
            ),
        ):
            await refresh_warm_context("test-user", "deploy the staging environment")

        search = mock_client.search_.await_args
        assert search is not None
        cfg = search.kwargs["config"]
        assert cfg.edge_config is not None
        assert cfg.edge_config.reranker == EdgeReranker.rrf
        assert cfg.limit == context_refresh.graphiti_config.context_max_facts
        # BFS graph traversal must be preserved on the refresh path — dropping
        # it would narrow recall breadth on exactly the path added to fix
        # recall (SECRT-2378). RRF recipe alone lacks BFS; the refresh must
        # keep the cross-encoder recipe's search methods.
        assert EdgeSearchMethod.bfs in cfg.edge_config.search_methods

    def test_the_recipe_keeps_the_search_methods_and_swaps_the_reranker(
        self,
    ) -> None:
        first_turn = EDGE_HYBRID_SEARCH_CROSS_ENCODER.edge_config
        refresh = REFRESH_RECIPE.edge_config
        assert first_turn is not None and refresh is not None
        assert first_turn.reranker == EdgeReranker.cross_encoder
        assert refresh.reranker == EdgeReranker.rrf
        # Identical search methods (incl. BFS) — only the reranker differs.
        assert EdgeSearchMethod.bfs in first_turn.search_methods
        assert refresh.search_methods == first_turn.search_methods


class TestRefreshReadsThroughTheRecallPolicy:
    """The follow-up refresh reads memory exactly as the first turn does:
    live facts only (``search_facts``), recallable episodes only
    (``recent_episodes``), one last check of both by uuid right before
    rendering (``recheck``), then ``recall_render``. So a fact forgotten
    between two turns, and the episode text it came from, cannot come back
    on the next turn's refresh. The live proof against FalkorDB is
    ``context_refresh_integration_test.py``; this pins the wiring."""

    @pytest.mark.asyncio
    async def test_a_fact_forgotten_since_the_last_turn_is_not_refreshed(
        self,
    ) -> None:
        scope = MemoryScope.for_expert("user-abc", "expert-1")
        forgotten, kept = _edge("e1", "Alice works on Atlas"), _edge("e2", "Bob")
        source = _episode("Remember that Alice works on Atlas")
        with (
            patch.object(
                context,
                "search_facts",
                new_callable=AsyncMock,
                return_value=[forgotten, kept],
            ) as search,
            patch.object(
                context,
                "recent_episodes",
                new_callable=AsyncMock,
                return_value=[source],
            ) as recent,
            # The last read no longer finds the fact, nor the episode that
            # cites it: the forget answered after the search had read them.
            patch.object(
                context,
                "recheck",
                new_callable=AsyncMock,
                return_value=([kept], []),
            ) as last_read,
            patch.object(context, "_spawn_ratification_hits") as ratify,
        ):
            block = await refresh_warm_context(
                "user-abc", "who works on Atlas now", expert_id="expert-1"
            )

        assert search.await_args is not None
        assert search.await_args.args == (scope, "who works on Atlas now")
        recipe = search.await_args.kwargs["recipe"]
        assert recipe.edge_config is not None
        assert recipe.edge_config.reranker == EdgeReranker.rrf
        recent.assert_awaited_once_with(scope, 5)
        last_read.assert_awaited_once_with(scope, [forgotten, kept], [source])
        assert block is not None and "Bob" in block
        assert "Alice" not in block, "the forgotten fact or its source text"
        ratify.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_turn_and_refresh_read_the_same_scope(self) -> None:
        """An expert chat's first turn reads the expert's memory, and so
        must every refresh after it."""
        scopes: list[MemoryScope] = []

        async def _fetch(scope, message, **_kwargs):
            scopes.append(scope)
            return None

        with patch.object(context, "_fetch", side_effect=_fetch):
            await fetch_warm_context("user-abc", "deploy staging", "expert-1")
            await refresh_warm_context(
                "user-abc", "deploy the staging stack", expert_id="expert-1"
            )

        assert scopes == [MemoryScope.for_expert("user-abc", "expert-1")] * 2
