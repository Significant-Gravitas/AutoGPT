"""Tests for the per-user community rebuild helper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .communities import (
    MAX_LABEL_PROP_ITERATIONS,
    _activity_since_last_rebuild,
    _bounded_label_propagation,
    _patch_upstream_label_propagation,
    _summarize_communities,
    rebuild_communities_for_user,
)


@pytest.fixture(autouse=True)
def _free_rebuild_lock():
    """``rebuild_communities_for_user`` acquires a per-user Redis lock before
    doing any work. Give every test a redis whose lock is free (SET NX → ok,
    release → ok); the contention test overrides ``set``."""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.eval = AsyncMock(return_value=1)
    with patch(
        "backend.copilot.graphiti.communities.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        yield redis


@pytest.fixture(autouse=True)
def _pin_openrouter_transport():
    """Pin the chat transport to OpenRouter so the flex-tier path is taken
    deterministically. ``rebuild_communities_for_user`` only takes the flex
    branch when ``chat_cfg.transport.supports_flex_tier`` is True, which holds
    only for the OpenRouter transport. That resolves from
    ``config.openrouter_active``, which needs a non-empty ``api_key`` — absent
    on fork-PR CI, where repo secrets (incl. ``OPENAI_API_KEY``) aren't exposed.
    Without this pin the transport silently drops to ``direct_anthropic`` (no
    flex), and tests that only patch ``make_flex_graphiti_client`` fall through
    to the real sync client. The flag-driven sync test patches
    ``community_rebuild_use_flex_tier`` itself, so it still takes the sync path.
    """
    with (
        patch("backend.copilot.sdk.env.config.use_openrouter", True),
        patch("backend.copilot.sdk.env.config.api_key", "or-key"),
        patch(
            "backend.copilot.sdk.env.config.base_url",
            "https://openrouter.ai/api/v1",
        ),
    ):
        yield


def _neighbor(uuid: str, edge_count: int = 1):
    """Mimic graphiti's Neighbor namedtuple — only attributes touched by LP."""
    return SimpleNamespace(node_uuid=uuid, edge_count=edge_count)


class TestBoundedLabelPropagation:
    """Regression coverage for the synchronous-LP infinite-loop fix.

    Upstream graphiti_core.label_propagation has an unbounded ``while True:``
    that can oscillate forever on bipartite-ish subgraphs (synchronous LP
    flips labels in lock-step). Our bounded variant caps iterations and
    returns the current state if the cap is hit. These tests pin the
    contract.
    """

    def test_empty_projection_returns_empty(self) -> None:
        assert _bounded_label_propagation({}) == []

    def test_single_node_no_neighbors(self) -> None:
        # One node, no neighbors → one cluster of size 1
        result = _bounded_label_propagation({"a": []})
        assert result == [["a"]]

    def test_connected_pair_converges(self) -> None:
        # Two nodes pointing at each other — should converge fast
        projection = {
            "a": [_neighbor("b")],
            "b": [_neighbor("a")],
        }
        result = _bounded_label_propagation(projection)
        # Expect single cluster — both nodes ended up in the same community
        assert len(result) == 1
        assert sorted(result[0]) == ["a", "b"]

    def test_two_disjoint_pairs_form_two_clusters(self) -> None:
        # No edges between {a,b} and {c,d} — should form 2 clusters
        projection = {
            "a": [_neighbor("b")],
            "b": [_neighbor("a")],
            "c": [_neighbor("d")],
            "d": [_neighbor("c")],
        }
        clusters = sorted([sorted(c) for c in _bounded_label_propagation(projection)])
        assert clusters == [["a", "b"], ["c", "d"]]

    def test_oscillating_projection_returns_at_cap_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Construct a fixture that genuinely oscillates under synchronous
        LP, and assert we exit at the cap with a warning rather than
        spinning forever.

        Two-node swap with edge_count >= 2: each iteration each node's
        plurality neighbor-community wins (count > 1 → bypasses the
        tie-break), so they swap labels every step. After step 1
        a=label(b_orig), b=label(a_orig). After step 2 they swap back.
        Forever — without the bound.
        """
        import logging

        projection = {
            "a": [_neighbor("b", edge_count=2)],
            "b": [_neighbor("a", edge_count=2)],
        }

        caplog.set_level(logging.WARNING, logger="backend.copilot.graphiti.communities")
        result = _bounded_label_propagation(projection)

        # Must return — not hang — and must produce a valid clustering
        # (every node assigned somewhere, no node lost).
        assigned = sorted(node for cluster in result for node in cluster)
        assert assigned == ["a", "b"]

        # The warning text must mention the cap. Check caplog.text (the
        # full captured log) because record collection can vary with
        # pytest-logging configuration; the formatted text is reliable.
        assert (
            f"{MAX_LABEL_PROP_ITERATIONS}-iteration cap" in caplog.text
        ), f"expected cap-warning in caplog.text; got: {caplog.text!r}"


class TestUpstreamMonkeyPatch:
    """The monkey-patch hooks into graphiti_core's module-level
    ``label_propagation`` symbol so ``get_community_clusters`` picks it
    up automatically. Pin the contract so a future refactor can't break
    it silently.
    """

    def test_upstream_reference_replaced(self) -> None:
        from graphiti_core.utils.maintenance import community_operations

        # After importing our communities module, upstream's symbol
        # must point at our bounded variant.
        assert community_operations.label_propagation is _bounded_label_propagation

    def test_patch_is_idempotent(self) -> None:
        from graphiti_core.utils.maintenance import community_operations

        # Idempotency sentinel on the function itself
        assert (
            getattr(community_operations.label_propagation, "_autogpt_bounded", False)
            is True
        )

        # Calling the patch again must be a no-op (must not raise, must
        # not replace with a fresh function lacking the sentinel)
        _patch_upstream_label_propagation()
        _patch_upstream_label_propagation()
        assert community_operations.label_propagation is _bounded_label_propagation


class TestSummarizeCommunities:
    def test_none_passthrough(self) -> None:
        assert _summarize_communities(None) is None

    def test_list_returns_count(self) -> None:
        assert _summarize_communities([1, 2, 3]) == {"count": 3}

    def test_dict_passthrough(self) -> None:
        assert _summarize_communities({"a": 1}) == {"a": 1}

    def test_unknown_shape_coerces_to_str(self) -> None:
        result = _summarize_communities(object())
        assert isinstance(result, dict)
        assert "raw" in result


class TestRebuildCommunitiesForUser:
    @pytest.mark.asyncio
    async def test_invalid_user_id_returns_error_dict(self) -> None:
        result = await rebuild_communities_for_user("")
        assert result["error"] is not None
        assert result["communities_built"] is None

    @pytest.mark.asyncio
    async def test_success_path_calls_detach_delete_then_build(self) -> None:
        # Mock the Graphiti client + its driver
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)

        client = MagicMock()
        client.graph_driver = driver
        client.build_communities = AsyncMock(return_value=[{"name": "c1"}])

        # Bypass the activity gate (covered separately in
        # TestActivitySinceLastRebuild) so this test stays focused on the
        # DETACH DELETE → build_communities → summarize flow.
        with (
            patch(
                "backend.copilot.graphiti.communities.make_flex_graphiti_client",
                new=AsyncMock(return_value=client),
            ),
            patch(
                "backend.copilot.graphiti.communities.close_graphiti_client",
                new=AsyncMock(),
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(return_value=(True, "first_rebuild", {})),
            ),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["error"] is None
        assert result["communities_built"] == {"count": 1}
        assert result["elapsed_seconds"] is not None

        # Defensive DETACH DELETE must run before build_communities
        cleanup_query = driver.execute_query.call_args.args[0]
        assert "MATCH (c:Community" in cleanup_query
        assert "DETACH DELETE c" in cleanup_query
        client.build_communities.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failure_path_returns_error_in_result(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)

        client = MagicMock()
        client.graph_driver = driver
        # Stub activity gate as "rebuild needed" so we reach build_communities
        client.build_communities = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch(
                "backend.copilot.graphiti.communities.make_flex_graphiti_client",
                new=AsyncMock(return_value=client),
            ),
            patch(
                "backend.copilot.graphiti.communities.close_graphiti_client",
                new=AsyncMock(),
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(return_value=(True, "first_rebuild", {})),
            ),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["error"] is not None
        assert "RuntimeError" in result["error"]
        # Always returns elapsed_seconds even on failure
        assert result["elapsed_seconds"] is not None

    @pytest.mark.asyncio
    async def test_skips_when_lock_held(self, _free_rebuild_lock) -> None:
        """A second concurrent rebuild (SET NX fails) returns skipped and
        never touches the graph — no client, no activity gate, no DETACH."""
        _free_rebuild_lock.set = AsyncMock(return_value=None)

        with (
            patch(
                "backend.copilot.graphiti.communities.make_flex_graphiti_client",
                new=AsyncMock(),
            ) as flex,
            patch(
                "backend.copilot.graphiti.communities.get_graphiti_client",
                new=AsyncMock(),
            ) as plain,
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(),
            ) as gate,
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["skipped"] is True
        assert result["skip_reason"] == "rebuild already in progress"
        # Short-circuited before any work — the held lock is the whole point.
        gate.assert_not_awaited()
        flex.assert_not_awaited()
        plain.assert_not_awaited()
        # We don't own the lock, so we must NOT release it.
        _free_rebuild_lock.eval.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_releases_lock_after_run(self, _free_rebuild_lock) -> None:
        """A normal run releases via single-key compare-and-delete Lua
        (cluster-routable) using the same token it acquired with."""
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)
        client = MagicMock()
        client.graph_driver = driver
        client.build_communities = AsyncMock(return_value=[{"name": "c1"}])

        with (
            patch(
                "backend.copilot.graphiti.communities.make_flex_graphiti_client",
                new=AsyncMock(return_value=client),
            ),
            patch(
                "backend.copilot.graphiti.communities.close_graphiti_client",
                new=AsyncMock(),
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(return_value=(True, "first_rebuild", {})),
            ),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["skipped"] is False
        # Acquire: single-key SET NX EX with a token value + TTL backstop.
        set_args, set_kwargs = _free_rebuild_lock.set.call_args
        assert set_kwargs.get("nx") is True
        assert set_kwargs.get("ex") == 1800 + 120
        # Release: single-key Lua compare-and-delete (numkeys=1 → routes on
        # the cluster) using the same token we acquired with.
        _free_rebuild_lock.eval.assert_awaited_once()
        eval_args = _free_rebuild_lock.eval.call_args.args
        assert eval_args[1] == 1
        assert eval_args[2].startswith("graphiti:community_rebuild_lock:")
        assert eval_args[3] == set_args[1]


def _ep_query_result(latest: str | None, total: int) -> tuple:
    """Build an execute_query return tuple shaped like an episode/community count."""
    rows = [{"latest": latest, "total": total}] if latest is not None or total else []
    if not rows:
        rows = [{"latest": None, "total": 0}]
    return (rows, None, None)


class TestActivitySinceLastRebuild:
    """Gate that decides whether ``rebuild_communities_for_user`` actually
    pays the LLM-summarization cost or short-circuits as a no-op."""

    @pytest.mark.asyncio
    async def test_skips_when_no_episodes(self) -> None:
        driver = AsyncMock()
        # First query: episodes — empty. Gate should return early.
        driver.execute_query.side_effect = [
            _ep_query_result(None, 0),  # episodes
            _ep_query_result(None, 0),  # communities
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is False
        assert reason == "no_episodes"
        assert stats["episodes_total"] == 0

    @pytest.mark.asyncio
    async def test_rebuilds_on_first_run(self) -> None:
        """Episodes exist but no communities yet — first rebuild always runs."""
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _ep_query_result("2026-05-20T01:00:00Z", 10),  # episodes
            _ep_query_result(None, 0),  # communities
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is True
        assert reason == "first_rebuild"
        assert stats["new_episodes_since_last_rebuild"] == 10

    @pytest.mark.asyncio
    async def test_skips_when_no_new_episodes_since_last_rebuild(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _ep_query_result("2026-05-20T01:00:00Z", 10),  # episodes
            _ep_query_result("2026-05-20T02:00:00Z", 3),  # communities — newer
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is False
        assert reason == "no_new_episodes_since_last_rebuild"
        assert stats["new_episodes_since_last_rebuild"] == 0

    @pytest.mark.asyncio
    async def test_skips_when_new_episodes_below_threshold(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _ep_query_result("2026-05-20T03:00:00Z", 13),  # episodes
            _ep_query_result("2026-05-20T02:00:00Z", 3),  # communities older
            ([{"c": 3}], None, None),  # count of new episodes — below threshold 5
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is False
        assert reason == "below_activity_threshold"
        assert stats["new_episodes_since_last_rebuild"] == 3
        assert stats["min_new_episodes_threshold"] == 5

    @pytest.mark.asyncio
    async def test_rebuilds_when_above_threshold(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _ep_query_result("2026-05-20T03:00:00Z", 20),  # episodes
            _ep_query_result("2026-05-20T02:00:00Z", 3),  # communities older
            ([{"c": 8}], None, None),  # 8 new episodes >= threshold 5
        ]
        should, reason, stats = await _activity_since_last_rebuild(
            driver, "user_x", min_new_episodes=5
        )
        assert should is True
        assert reason == "activity_above_threshold"
        assert stats["new_episodes_since_last_rebuild"] == 8


class TestRebuildForUserActivityGate:
    """End-to-end behavior of ``rebuild_communities_for_user`` with the gate."""

    @pytest.mark.asyncio
    async def test_skip_path_records_reason_and_does_not_call_build(self) -> None:
        driver = AsyncMock()
        client = MagicMock()
        client.graph_driver = driver
        client.build_communities = AsyncMock()

        with (
            patch(
                "backend.copilot.graphiti.communities.make_flex_graphiti_client",
                new=AsyncMock(return_value=client),
            ),
            patch(
                "backend.copilot.graphiti.communities.close_graphiti_client",
                new=AsyncMock(),
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(
                    return_value=(False, "below_activity_threshold", {"x": 1})
                ),
            ),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["skipped"] is True
        assert result["skip_reason"] == "below_activity_threshold"
        assert result["activity"] == {"x": 1}
        assert result["communities_built"] is None
        assert result["error"] is None
        # Critical: build_communities NOT called, no DETACH DELETE issued
        client.build_communities.assert_not_awaited()
        driver.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_bypasses_gate(self) -> None:
        """``force=True`` skips activity check; always rebuilds."""
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)
        client = MagicMock()
        client.graph_driver = driver
        client.build_communities = AsyncMock(return_value=[])

        with (
            patch(
                "backend.copilot.graphiti.communities.make_flex_graphiti_client",
                new=AsyncMock(return_value=client),
            ),
            patch(
                "backend.copilot.graphiti.communities.close_graphiti_client",
                new=AsyncMock(),
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(),
            ) as mock_gate,
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef", force=True
            )

        assert result["forced"] is True
        assert result["skipped"] is False
        # The activity gate must not be consulted when force=True
        mock_gate.assert_not_called()
        # build_communities WAS called
        client.build_communities.assert_awaited_once()


class TestRebuildPathSelection:
    """Which Graphiti client (flex vs sync) the rebuild actually uses."""

    @pytest.mark.asyncio
    async def test_uses_flex_client_by_default(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)
        client = MagicMock()
        client.graph_driver = driver
        client.build_communities = AsyncMock(return_value=[])

        flex_factory = AsyncMock(return_value=client)
        close_mock = AsyncMock()
        sync_factory = AsyncMock(return_value=client)

        with (
            patch(
                "backend.copilot.graphiti.communities.make_flex_graphiti_client",
                new=flex_factory,
            ),
            patch(
                "backend.copilot.graphiti.communities.close_graphiti_client",
                new=close_mock,
            ),
            patch(
                "backend.copilot.graphiti.communities.get_graphiti_client",
                new=sync_factory,
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(return_value=(True, "first_rebuild", {})),
            ),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["execution_path"] == "flex"
        flex_factory.assert_awaited_once()
        sync_factory.assert_not_called()
        # finally block must release the one-shot driver
        close_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uses_sync_client_when_flex_disabled(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)
        client = MagicMock()
        client.graph_driver = driver
        client.build_communities = AsyncMock(return_value=[])

        flex_factory = AsyncMock(return_value=client)
        close_mock = AsyncMock()
        sync_factory = AsyncMock(return_value=client)

        with (
            patch(
                "backend.copilot.graphiti.communities.graphiti_config.community_rebuild_use_flex_tier",
                new=False,
            ),
            patch(
                "backend.copilot.graphiti.communities.make_flex_graphiti_client",
                new=flex_factory,
            ),
            patch(
                "backend.copilot.graphiti.communities.close_graphiti_client",
                new=close_mock,
            ),
            patch(
                "backend.copilot.graphiti.communities.get_graphiti_client",
                new=sync_factory,
            ),
            patch(
                "backend.copilot.graphiti.communities._activity_since_last_rebuild",
                new=AsyncMock(return_value=(True, "first_rebuild", {})),
            ),
        ):
            result = await rebuild_communities_for_user(
                "883cc9da-fe37-4863-839b-acba022bf3ef"
            )

        assert result["execution_path"] == "sync"
        sync_factory.assert_awaited_once()
        flex_factory.assert_not_called()
        # No flex client → no close call
        close_mock.assert_not_awaited()
