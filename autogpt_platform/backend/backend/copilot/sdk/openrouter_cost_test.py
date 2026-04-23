"""Unit tests for SDK-path OpenRouter cost recording."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.copilot.model import ChatSession
from backend.copilot.sdk.openrouter_cost import record_turn_cost_from_openrouter


def _session() -> ChatSession:
    now = datetime.now(UTC)
    return ChatSession(
        session_id="sess-1",
        user_id="user-1",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=[],
    )


def _mock_generation_response(cost: float) -> dict:
    return {
        "data": {
            "total_cost": cost,
            "native_tokens_prompt": 1000,
            "native_tokens_completion": 50,
            "tokens_prompt": 1100,
            "tokens_completion": 60,
        }
    }


class TestRecordTurnCostFromOpenRouter:
    """Single-write semantics: the cost + rate-limit counter is updated
    exactly once per turn via this background task.  The sync path at
    the call site is already skipped for non-Anthropic OpenRouter turns,
    so there's no double-counting path even on partial failure."""

    @pytest.mark.asyncio
    async def test_empty_generation_ids_no_op(self):
        """Direct-Anthropic turn produces no gen-IDs — task is a no-op."""
        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="anthropic/claude-sonnet-4.6",
                prompt_tokens=10,
                completion_tokens=5,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=[],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.05,
                api_key="sk-or-test",
                log_prefix="[test]",
            )
        mock_persist.assert_not_called()
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_api_key_no_op(self):
        """Without an OpenRouter API key we can't query the endpoint — skip."""
        with patch(
            "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
            new_callable=AsyncMock,
        ) as mock_persist:
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=10,
                completion_tokens=5,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-1"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.02,
                api_key=None,
                log_prefix="[test]",
            )
        mock_persist.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_generation_records_real_cost(self):
        """Authoritative cost from OpenRouter is the value recorded — no
        reliance on the fallback estimate."""
        real_cost = 0.02900595

        async def _get(self, url, **kwargs):  # noqa: ARG001
            return httpx.Response(200, json=_mock_generation_response(real_cost))

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=29669,
                completion_tokens=280,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-1776842410"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.01858,  # rate-card estimate, deliberately wrong
                api_key="sk-or-test",
                log_prefix="[test]",
            )
        mock_persist.assert_called_once()
        kwargs = mock_persist.call_args.kwargs
        assert kwargs["cost_usd"] == pytest.approx(real_cost, rel=1e-9)
        assert kwargs["prompt_tokens"] == 29669
        assert kwargs["completion_tokens"] == 280
        assert kwargs["provider"] == "open_router"
        assert kwargs["model"] == "moonshotai/kimi-k2.6"

    @pytest.mark.asyncio
    async def test_multi_round_turn_sums_costs(self):
        """Tool-use turn has N generation IDs; the real cost is the sum
        of ``total_cost`` across all rounds — recorded in a single row."""
        costs_by_id = {"gen-a": 0.029, "gen-b": 0.030}

        async def _get(self, url, **kwargs):  # noqa: ARG001
            gen_id = kwargs.get("params", {}).get("id")
            return httpx.Response(
                200, json=_mock_generation_response(costs_by_id[gen_id])
            )

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=60000,
                completion_tokens=600,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-a", "gen-b"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.037,
                api_key="sk-or-test",
                log_prefix="[test]",
            )
        mock_persist.assert_called_once()
        cost = mock_persist.call_args.kwargs["cost_usd"]
        assert cost == pytest.approx(sum(costs_by_id.values()), rel=1e-9)

    @pytest.mark.asyncio
    async def test_partial_lookup_falls_back_to_estimate(self):
        """If only some gen-IDs resolve, summing them would under-report.
        Fall back to the caller's estimate and log — the rate-limit
        counter stays populated with the best available number."""
        fallback = 0.05
        seq = iter(
            [
                httpx.Response(200, json=_mock_generation_response(0.03)),
                httpx.Response(404, text="not found"),
                httpx.Response(404, text="not found"),
                httpx.Response(404, text="not found"),
                httpx.Response(404, text="not found"),
            ]
        )

        async def _get(self, *args, **kwargs):  # noqa: ARG001
            return next(seq)

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=1000,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-a", "gen-b"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=fallback,
                api_key="sk-or-test",
                log_prefix="[test]",
            )
        mock_persist.assert_called_once()
        assert mock_persist.call_args.kwargs["cost_usd"] == fallback

    @pytest.mark.asyncio
    async def test_fast_fail_on_401_no_retries(self):
        """Permanent client errors (401/403/400) must not retry — burning
        the 30s retry window on an unauthenticated request wastes API
        quota and delays the fallback."""
        call_count = {"n": 0}

        async def _get(self, *args, **kwargs):  # noqa: ARG001
            call_count["n"] += 1
            return httpx.Response(401, text="unauthorized")

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=1000,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-a"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.02,
                api_key="sk-bad",
                log_prefix="[test]",
            )
        # Only one call — no retries.
        assert call_count["n"] == 1
        # Fallback was recorded (lookup failed → keep rate-limit counter live).
        mock_persist.assert_called_once()
        assert mock_persist.call_args.kwargs["cost_usd"] == 0.02

    @pytest.mark.asyncio
    async def test_retries_on_404_then_succeeds(self):
        """Indexing lag: endpoint returns 404 initially, then 200 once the
        billing row is indexed.  Retry budget should exhaust transient
        states rather than giving up on first 404."""
        seq = iter(
            [
                httpx.Response(404, text="not found"),
                httpx.Response(200, json=_mock_generation_response(0.025)),
            ]
        )

        async def _get(self, *args, **kwargs):  # noqa: ARG001
            return next(seq)

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=1000,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-a"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.05,
                api_key="sk-or-test",
                log_prefix="[test]",
            )
        mock_persist.assert_called_once()
        assert mock_persist.call_args.kwargs["cost_usd"] == pytest.approx(0.025)

    @pytest.mark.asyncio
    async def test_complete_lookup_failure_falls_back_to_estimate(self):
        """Every lookup fails → record the estimate so the rate-limit
        counter isn't left empty.  At-worst parity with the pre-task
        behaviour."""
        fallback = 0.02

        async def _get(self, *args, **kwargs):  # noqa: ARG001
            raise httpx.ConnectError("no network")

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=1000,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-a"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=fallback,
                api_key="sk-or-test",
                log_prefix="[test]",
            )
        mock_persist.assert_called_once()
        assert mock_persist.call_args.kwargs["cost_usd"] == fallback

    @pytest.mark.asyncio
    async def test_compaction_subagent_gen_ids_are_swept(self, tmp_path):
        """CLI-internal compaction spawns a subagent JSONL under
        ``<project_dir>/<session_id>/subagents/agent-acompact-*.jsonl``
        whose gen-IDs the live adapter never surfaces.  When
        ``cli_project_dir`` + ``cli_session_id`` + ``turn_start_ts``
        are supplied the reconcile walks only THIS session's subagents
        and discovers the compaction IDs."""
        session_id = "sess-abc"
        sub_dir = tmp_path / session_id / "subagents"
        sub_dir.mkdir(parents=True)
        (sub_dir / "agent-acompact-xyz.jsonl").write_text(
            '{"type":"assistant","message":{"id":"gen-compact-1","content":[]}}\n'
            '{"type":"assistant","message":{"id":"gen-compact-2","content":[]}}\n'
        )

        costs_by_id = {
            "gen-main-1": 0.020,
            "gen-compact-1": 0.005,
            "gen-compact-2": 0.003,
        }

        async def _get(self, *args, **kwargs):  # noqa: ARG001
            gen_id = kwargs.get("params", {}).get("id")
            return httpx.Response(
                200, json=_mock_generation_response(costs_by_id[gen_id])
            )

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="anthropic/claude-opus-4.7",
                prompt_tokens=1000,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-main-1"],
                cli_project_dir=str(tmp_path),
                cli_session_id=session_id,
                turn_start_ts=0.0,
                fallback_cost_usd=0.05,
                api_key="sk-or-test",
                log_prefix="[test]",
            )
        mock_persist.assert_called_once()
        assert mock_persist.call_args.kwargs["cost_usd"] == pytest.approx(
            sum(costs_by_id.values()), rel=1e-9
        )

    @pytest.mark.asyncio
    async def test_compaction_sweep_no_subagents_is_noop(self, tmp_path):
        """No compaction happened → reconcile uses only the caller's
        gen-IDs, same as when cli_project_dir is None."""
        session_id = "sess-none"
        (tmp_path / session_id).mkdir()

        async def _get(self, *args, **kwargs):  # noqa: ARG001
            return httpx.Response(200, json=_mock_generation_response(0.02))

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=1000,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-main-1"],
                cli_project_dir=str(tmp_path),
                cli_session_id=session_id,
                turn_start_ts=0.0,
                fallback_cost_usd=0.05,
                api_key="sk-or-test",
                log_prefix="[test]",
            )
        mock_persist.assert_called_once()
        assert mock_persist.call_args.kwargs["cost_usd"] == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_compaction_sweep_ignores_prior_turn_and_foreign_sessions(
        self, tmp_path
    ):
        """Scoping guards the sweep from double-billing: a stale subagent
        file from a prior turn (mtime before ``turn_start_ts``) and any
        subagent from a foreign session (different session_id folder)
        must BOTH be skipped.  Without either guard, a long-running
        session with past compactions would re-bill every prior turn,
        and a second chat session in the same cwd would inherit the
        first session's compaction cost."""
        import os
        import time

        this_session = "sess-current"
        other_session = "sess-other"

        this_subagents = tmp_path / this_session / "subagents"
        this_subagents.mkdir(parents=True)
        other_subagents = tmp_path / other_session / "subagents"
        other_subagents.mkdir(parents=True)

        # Prior-turn compaction file — same session, stale mtime.
        stale_file = this_subagents / "agent-acompact-stale.jsonl"
        stale_file.write_text(
            '{"type":"assistant","message":{"id":"gen-stale-1","content":[]}}\n'
        )
        # Foreign session's compaction file.
        foreign_file = other_subagents / "agent-acompact-foreign.jsonl"
        foreign_file.write_text(
            '{"type":"assistant","message":{"id":"gen-foreign-1","content":[]}}\n'
        )
        # Current-turn compaction file — fresh.
        fresh_file = this_subagents / "agent-acompact-fresh.jsonl"
        fresh_file.write_text(
            '{"type":"assistant","message":{"id":"gen-fresh-1","content":[]}}\n'
        )

        # turn_start_ts lies between the stale and fresh mtimes.
        past = time.time() - 3600
        os.utime(stale_file, (past, past))
        os.utime(foreign_file, (past, past))
        turn_start_ts = time.time() - 60  # 1 min ago
        fresh_now = time.time()
        os.utime(fresh_file, (fresh_now, fresh_now))

        costs_by_id = {
            "gen-main-1": 0.010,
            "gen-fresh-1": 0.004,
        }

        async def _get(self, *args, **kwargs):  # noqa: ARG001
            gen_id = kwargs.get("params", {}).get("id")
            # If the sweep leaks a stale/foreign ID, the test fails here
            # with a KeyError rather than silently over-billing.
            assert gen_id in costs_by_id, f"sweep leaked out-of-scope gen_id {gen_id}"
            return httpx.Response(
                200, json=_mock_generation_response(costs_by_id[gen_id])
            )

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=1000,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-main-1"],
                cli_project_dir=str(tmp_path),
                cli_session_id=this_session,
                turn_start_ts=turn_start_ts,
                fallback_cost_usd=0.05,
                api_key="sk-or-test",
                log_prefix="[test]",
            )
        mock_persist.assert_called_once()
        # Exactly the current-turn main + fresh compaction — no stale,
        # no foreign.
        assert mock_persist.call_args.kwargs["cost_usd"] == pytest.approx(
            sum(costs_by_id.values()), rel=1e-9
        )


class TestLangfuseTraceBackfill:
    """Reconciled cost is mirrored back onto the Langfuse trace as a
    child event so operators see the real number alongside the SDK-CLI
    rate-card estimate the OTel bridge already wrote."""

    @pytest.mark.asyncio
    async def test_event_emitted_with_real_cost_when_trace_id_supplied(self):
        real_cost = 0.025
        mock_lf = MagicMock()

        async def _get(_self, _url, **_kwargs):
            return httpx.Response(200, json=_mock_generation_response(real_cost))

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ),
            patch("httpx.AsyncClient.get", new=_get),
            patch(
                "backend.copilot.sdk.openrouter_cost.get_client",
                return_value=mock_lf,
            ),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=29669,
                completion_tokens=280,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-1"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.018,
                api_key="sk-or-test",
                log_prefix="[test]",
                langfuse_trace_id="trace-abc",
            )

        mock_lf.create_event.assert_called_once()
        kwargs = mock_lf.create_event.call_args.kwargs
        assert kwargs["trace_context"] == {"trace_id": "trace-abc"}
        assert kwargs["name"] == "openrouter-cost-reconcile"
        meta = kwargs["metadata"]
        assert meta["cost_usd"] == pytest.approx(real_cost, rel=1e-9)
        assert meta["cost_source"] == "openrouter"
        assert meta["fallback_cost_usd"] == pytest.approx(0.018, rel=1e-9)
        assert meta["resolved_generation_id_count"] == 1
        assert meta["generation_id_count"] == 1
        assert meta["prompt_tokens"] == 29669
        assert meta["completion_tokens"] == 280
        assert meta["model"] == "moonshotai/kimi-k2.6"
        assert meta["provider"] == "open_router"

    @pytest.mark.asyncio
    async def test_event_marks_fallback_when_lookup_partial(self):
        """When some gen-ID lookups fail, real_cost falls back to the
        rate-card estimate.  The Langfuse event must mark cost_source as
        ``"fallback"`` so operators don't mistake it for an authoritative
        OpenRouter reconciliation.
        """
        mock_lf = MagicMock()
        call_count = {"n": 0}

        async def _get(_self, _url, **_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=_mock_generation_response(0.012))
            return httpx.Response(404, json={"error": "not found"})

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ),
            patch("httpx.AsyncClient.get", new=_get),
            patch(
                "backend.copilot.sdk.openrouter_cost.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.copilot.sdk.openrouter_cost.get_client",
                return_value=mock_lf,
            ),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=100,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-1", "gen-2"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.018,
                api_key="sk-or-test",
                log_prefix="[test]",
                langfuse_trace_id="trace-partial",
            )

        mock_lf.create_event.assert_called_once()
        meta = mock_lf.create_event.call_args.kwargs["metadata"]
        assert meta["cost_source"] == "fallback"
        assert meta["cost_usd"] == pytest.approx(0.018, rel=1e-9)
        assert meta["resolved_generation_id_count"] == 1
        assert meta["generation_id_count"] == 2

    @pytest.mark.asyncio
    async def test_no_event_when_trace_id_missing(self):
        """Anthropic-direct turns and any path without an active Langfuse
        OTel context don't have a trace_id — backfill is a no-op."""
        mock_lf = MagicMock()

        async def _get(_self, _url, **_kwargs):
            return httpx.Response(200, json=_mock_generation_response(0.01))

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ),
            patch("httpx.AsyncClient.get", new=_get),
            patch(
                "backend.copilot.sdk.openrouter_cost.get_client",
                return_value=mock_lf,
            ),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=100,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-1"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.005,
                api_key="sk-or-test",
                log_prefix="[test]",
                langfuse_trace_id=None,
            )

        mock_lf.create_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_backfill_failure_swallowed_does_not_break_persist(self):
        """If Langfuse is unavailable, the reconcile still persists the
        cost row — backfill is best-effort and must not raise."""
        mock_lf = MagicMock()
        mock_lf.create_event = MagicMock(side_effect=RuntimeError("network down"))

        async def _get(_self, _url, **_kwargs):
            return httpx.Response(200, json=_mock_generation_response(0.02))

        with (
            patch(
                "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                new_callable=AsyncMock,
            ) as mock_persist,
            patch("httpx.AsyncClient.get", new=_get),
            patch(
                "backend.copilot.sdk.openrouter_cost.get_client",
                return_value=mock_lf,
            ),
        ):
            await record_turn_cost_from_openrouter(
                session=_session(),
                user_id="u1",
                model="moonshotai/kimi-k2.6",
                prompt_tokens=100,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                generation_ids=["gen-1"],
                cli_project_dir=None,
                cli_session_id=None,
                turn_start_ts=None,
                fallback_cost_usd=0.018,
                api_key="sk-or-test",
                log_prefix="[test]",
                langfuse_trace_id="trace-xyz",
            )

        mock_persist.assert_called_once()
