"""Tests for the workflow-recording MCP tool handlers.

The handlers are thin over recording_skill + the shim's _RecordingProxy;
these tests cover the gating (capability + consent), the start/stop/list
lifecycle, the §8 confirmation gate surfaced through generate/dry_run, and
the multi-row dry-run requirement.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.copilot.sdk.recording_tools as rt
from backend.api.features.local_executor.state import RecordingState
from backend.copilot.tools.local_pc_shim import (
    LocalPCShim,
    ShimConnectionGuard,
    ShimRecordingError,
)
from backend.copilot.tools.recording_models import (
    RecordingReviewApplied,
    RecordingSummary,
    WorkflowRecording,
)


def _parse(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


def _shim(*, capabilities=("recording", "computer_use")):
    # spec=LocalPCShim so the handler's isinstance(sb, LocalPCShim) gate passes.
    shim = MagicMock(spec=LocalPCShim)
    shim.sandbox_id = "sess-1"
    shim.machine_id = "machine-1"
    shim.capabilities = list(capabilities)
    shim.capability_set = frozenset(capabilities)
    guard = ShimConnectionGuard(
        generation=1,
        machine_id="machine-1",
        computer_use_driver="cua-driver 0.38.0",
        computer_use_tools=("click", "get_window_state", "type_text"),
    )
    shim.capture_connection_guard.return_value = guard
    shim.connection_guard_matches.return_value = True
    shim.recording = MagicMock()
    shim.recording.effective_route_for = MagicMock(side_effect=lambda _id, route: route)
    shim.computer = MagicMock()
    shim.computer.call = AsyncMock(return_value={"is_error": False, "content": []})
    shim.close_recording = MagicMock()
    return shim


def _install(monkeypatch, shim):
    monkeypatch.setattr(rt, "get_current_sandbox", lambda: shim)
    monkeypatch.setattr(
        rt, "_chat_config", SimpleNamespace(allow_computer_use=True), raising=False
    )
    monkeypatch.setattr(
        rt,
        "get_execution_context",
        lambda: ("user-1", SimpleNamespace(session_id=shim.sandbox_id)),
    )
    monkeypatch.setattr(
        rt,
        "is_computer_use_approved",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        rt,
        "get_recording_state",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        rt,
        "list_recording_states",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(rt, "mark_recording_started", AsyncMock())
    monkeypatch.setattr(rt, "mark_recording_stopped", AsyncMock())
    # Fresh per-session state for each test.
    rt._SESSION_STATE.clear()


def _mark_reviewed(shim, recording_id: str) -> None:
    rt.register_recording_reviewed(
        shim,
        recording_id,
        RecordingReviewApplied(recording_id=recording_id, step_count=2),
    )


def _single_row_recording():
    return WorkflowRecording.from_payload(
        {
            "recording_id": "rec_1",
            "steps": [
                {
                    "seq": 1,
                    "action": "fill",
                    "screenshot_ref": "s1",
                    "cursor": [10, 20],
                    "enrichment": {
                        "kind": "dom",
                        "label": "First Name",
                        "selectors": [{"strategy": "id", "value": "#fn"}],
                    },
                    "value": {"raw": "John", "type": "text", "is_parameter": None},
                },
                {
                    "seq": 2,
                    "action": "submit",
                    "screenshot_ref": "s2",
                    "cursor": [30, 40],
                    "enrichment": {"kind": "none", "label": "Save"},
                },
            ],
        }
    )


def _two_row_recording():
    def row(seq_off, fn):
        return [
            {
                "seq": seq_off + 1,
                "action": "fill",
                "screenshot_ref": f"s{seq_off}",
                "cursor": [10, 20],
                "enrichment": {
                    "kind": "dom",
                    "label": "First Name",
                    "selectors": [{"strategy": "id", "value": "#fn"}],
                },
                "value": {"raw": fn, "type": "text", "is_parameter": None},
            },
            {
                "seq": seq_off + 2,
                "action": "submit",
                "screenshot_ref": f"s{seq_off}b",
                "cursor": [30, 40],
                "enrichment": {"kind": "none", "label": "Save"},
            },
        ]

    return WorkflowRecording.from_payload(
        {"recording_id": "rec_2", "steps": [*row(0, "John"), *row(2, "Jane")]}
    )


class TestGating:
    @pytest.mark.asyncio
    async def test_no_shim_refuses(self, monkeypatch):
        monkeypatch.setattr(rt, "get_current_sandbox", lambda: None)
        out = await rt._h_record_workflow({"action": "start"})
        assert out["isError"] is True
        assert _parse(out)["code"] == "NO_LOCAL_PC_EXECUTOR"

    @pytest.mark.asyncio
    async def test_missing_capability_refuses(self, monkeypatch):
        shim = _shim(capabilities=("files",))
        _install(monkeypatch, shim)
        out = await rt._h_list_recordings({})
        assert out["isError"] is True
        assert _parse(out)["code"] == "CAPABILITY_NOT_GRANTED"


class TestRecordWorkflowStart:
    @pytest.mark.asyncio
    async def test_start_requests_native_consent(self, monkeypatch):
        shim = _shim()
        shim.recording.start_with_consent = AsyncMock(return_value="rec_1")
        _install(monkeypatch, shim)
        out = await rt._h_record_workflow({"action": "start"})
        assert out["isError"] is False
        shim.recording.start_with_consent.assert_awaited_once_with(
            mode="copilot",
            interpretation_route="extract_then_cloud",
            channels=["floor"],
        )

    @pytest.mark.asyncio
    async def test_start_passes_token_and_tracks_recording(self, monkeypatch):
        shim = _shim()
        shim.recording.start_with_consent = AsyncMock(return_value="rec_1")
        _install(monkeypatch, shim)
        out = await rt._h_record_workflow(
            {
                "action": "start",
                "mode": "copilot",
                "channels": ["floor", "browser"],
            }
        )
        assert out["isError"] is False
        assert _parse(out)["recording_id"] == "rec_1"
        shim.recording.start_with_consent.assert_awaited_once()
        # Tracked in session state for list_recordings.
        listed = _parse(await rt._h_list_recordings({}))
        assert listed["recordings"][0]["recording_id"] == "rec_1"
        assert listed["recordings"][0]["status"] == "recording"

    @pytest.mark.asyncio
    async def test_start_surfaces_shim_consent_error(self, monkeypatch):
        shim = _shim()
        shim.recording.start_with_consent = AsyncMock(
            side_effect=ShimRecordingError("CONSENT_REQUIRED", "bad token")
        )
        _install(monkeypatch, shim)
        out = await rt._h_record_workflow({"action": "start"})
        assert out["isError"] is True
        assert _parse(out)["code"] == "CONSENT_REQUIRED"


def test_session_recording_state_is_lru_bounded():
    rt._SESSION_STATE.clear()
    for index in range(rt._MAX_SESSION_STATES + 1):
        shim = _shim()
        shim.sandbox_id = f"session-{index}"
        rt._state_for(shim)

    assert len(rt._SESSION_STATE) == rt._MAX_SESSION_STATES
    assert "session-0" not in rt._SESSION_STATE


class TestRecordWorkflowStop:
    @pytest.mark.asyncio
    async def test_stop_returns_summary_and_closes_stream(self, monkeypatch):
        shim = _shim()
        shim.recording.stop = AsyncMock(
            return_value=RecordingSummary(
                recording_id="rec_1", step_count=2, duration_seconds=5.0
            )
        )
        _install(monkeypatch, shim)
        out = await rt._h_record_workflow({"action": "stop", "recording_id": "rec_1"})
        assert out["isError"] is False
        assert _parse(out)["summary"]["step_count"] == 2
        shim.close_recording.assert_called_once_with("rec_1")


class TestRecordingSharedState:
    """A recording started or stopped through the tool is saved where every
    worker reads it, as the HTTP routes save theirs."""

    @pytest.mark.asyncio
    async def test_a_start_is_saved_where_every_worker_reads_it(self, monkeypatch):
        """The start lived only in this worker's memory, so another worker's
        list_recordings left it out."""
        shim = _shim()
        shim.recording.start_with_consent = AsyncMock(return_value="rec_1")
        shim.recording.effective_route_for = MagicMock(return_value="local_vlm")
        _install(monkeypatch, shim)

        out = await rt._h_record_workflow({"action": "start", "channels": ["floor"]})

        assert out["isError"] is False
        rt.mark_recording_started.assert_awaited_once_with(
            "sess-1",
            "rec_1",
            mode="copilot",
            interpretation_route="local_vlm",
            channels=["floor"],
        )

    @pytest.mark.asyncio
    async def test_a_start_whose_state_cannot_be_saved_is_stopped(self, monkeypatch):
        shim = _shim()
        shim.recording.start_with_consent = AsyncMock(return_value="rec_1")
        shim.recording.stop = AsyncMock(
            return_value=RecordingSummary(recording_id="rec_1")
        )
        _install(monkeypatch, shim)
        monkeypatch.setattr(
            rt, "mark_recording_started", AsyncMock(side_effect=RuntimeError("down"))
        )

        out = await rt._h_record_workflow({"action": "start"})

        assert out["isError"] is True
        assert _parse(out)["code"] == "RECORDING_STATE_UNAVAILABLE"
        shim.recording.stop.assert_awaited_once_with("rec_1")
        assert _parse(await rt._h_list_recordings({}))["recordings"] == []

    @pytest.mark.asyncio
    async def test_a_stop_is_saved_and_never_sent_twice(self, monkeypatch):
        """A stop saved only here left the shared state saying recording, and
        the machine can't take a second STOP for the same recording."""
        summary = RecordingSummary(recording_id="rec_1", step_count=2)
        shim = _shim()
        shim.recording.stop = AsyncMock(return_value=summary)
        _install(monkeypatch, shim)

        first = await rt._h_record_workflow({"action": "stop", "recording_id": "rec_1"})
        monkeypatch.setattr(
            rt,
            "get_recording_state",
            AsyncMock(
                return_value=RecordingState(
                    recording_id="rec_1", status="stopped", summary=summary.to_dict()
                )
            ),
        )
        again = await rt._h_record_workflow({"action": "stop", "recording_id": "rec_1"})

        assert first["isError"] is False and again["isError"] is False
        rt.mark_recording_stopped.assert_awaited_once_with(
            "sess-1", "rec_1", summary=summary.to_dict()
        )
        shim.recording.stop.assert_awaited_once_with("rec_1")
        assert _parse(again)["summary"]["step_count"] == 2


class TestConcurrentStops:
    @pytest.mark.asyncio
    async def test_a_stop_that_lost_the_race_returns_the_saved_summary(
        self, monkeypatch
    ):
        """Two stops both found no saved summary and both sent STOP; the
        machine refused the second, so that request failed although the
        first had saved the summary."""
        summary = RecordingSummary(recording_id="rec_1", step_count=2)
        shim = _shim()
        shim.recording.stop = AsyncMock(
            side_effect=ShimRecordingError("RECORDING_NOT_FOUND", "no such recording")
        )
        _install(monkeypatch, shim)
        saved = RecordingState(
            recording_id="rec_1", status="stopped", summary=summary.to_dict()
        )
        monkeypatch.setattr(
            rt, "get_recording_state", AsyncMock(side_effect=[None, saved])
        )

        out = await rt._h_record_workflow({"action": "stop", "recording_id": "rec_1"})

        assert out["isError"] is False
        assert _parse(out)["summary"]["step_count"] == 2
        rt.mark_recording_stopped.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_recording_nobody_stopped_is_still_not_found(self, monkeypatch):
        shim = _shim()
        shim.recording.stop = AsyncMock(
            side_effect=ShimRecordingError("RECORDING_NOT_FOUND", "no such recording")
        )
        _install(monkeypatch, shim)

        out = await rt._h_record_workflow({"action": "stop", "recording_id": "rec_1"})

        assert out["isError"] is True
        assert _parse(out)["code"] == "RECORDING_NOT_FOUND"


class TestGenerateSkill:
    @pytest.mark.asyncio
    async def test_shared_review_state_unblocks_separate_executor_process(
        self, monkeypatch
    ):
        shim = _shim()
        shim.recording.fetch = AsyncMock(return_value=_single_row_recording())
        _install(monkeypatch, shim)
        monkeypatch.setattr(
            rt,
            "get_recording_state",
            AsyncMock(
                return_value=RecordingState(
                    recording_id="rec_1",
                    status="reviewed",
                    step_count=2,
                )
            ),
        )

        out = await rt._h_generate_skill({"recording_id": "rec_1"})

        assert out["isError"] is False
        assert _parse(out)["recording_id"] == "rec_1"
        shim.recording.fetch.assert_awaited_once_with("rec_1")

    @pytest.mark.asyncio
    async def test_stopped_recording_requires_authoritative_review(self, monkeypatch):
        shim = _shim()
        shim.recording.fetch = AsyncMock(return_value=_single_row_recording())
        _install(monkeypatch, shim)
        rt.register_recording_stopped(
            shim,
            "rec_1",
            RecordingSummary(recording_id="rec_1", step_count=2),
        )

        out = await rt._h_generate_skill({"recording_id": "rec_1"})

        assert out["isError"] is True
        assert _parse(out)["code"] == "RECORDING_REVIEW_REQUIRED"
        shim.recording.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_row_needs_confirmation(self, monkeypatch):
        shim = _shim()
        shim.recording.fetch = AsyncMock(return_value=_single_row_recording())
        _install(monkeypatch, shim)
        _mark_reviewed(shim, "rec_1")
        out = await rt._h_generate_skill({"recording_id": "rec_1"})
        body = _parse(out)
        assert body["needs_confirmation"] is True
        assert body["questions"]
        # Proposed clarifications surfaced for the trust loop.
        assert body["proposed_clarifications"]

    @pytest.mark.asyncio
    async def test_confirmation_unblocks(self, monkeypatch):
        shim = _shim()
        shim.recording.fetch = AsyncMock(return_value=_single_row_recording())
        _install(monkeypatch, shim)
        _mark_reviewed(shim, "rec_1")
        out = await rt._h_generate_skill(
            {
                "recording_id": "rec_1",
                "clarifications": {"confirmed_parameters": {"First Name": True}},
            }
        )
        body = _parse(out)
        assert body["needs_confirmation"] is False


class TestDryRunSkill:
    @pytest.mark.asyncio
    async def test_dry_run_reconnect_during_consent_check_blocks_replay(
        self, monkeypatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)
        consent_started = asyncio.Event()
        release_consent = asyncio.Event()

        async def delayed_approval(*_args, **_kwargs):
            consent_started.set()
            await release_consent.wait()
            return True

        monkeypatch.setattr(rt, "is_computer_use_approved", delayed_approval)
        task = asyncio.create_task(
            rt._h_dry_run_skill({"recording_id": "rec_x", "data_rows": [{"a": 1}]})
        )
        await consent_started.wait()
        shim.connection_guard_matches.return_value = False
        release_consent.set()

        out = await task

        assert out["isError"] is True
        assert _parse(out)["code"] == "COMPUTER_USE_CONNECTION_CHANGED"
        shim.computer.call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dry_run_rechecks_revoked_computer_use_consent(self, monkeypatch):
        shim = _shim()
        _install(monkeypatch, shim)
        monkeypatch.setattr(
            rt,
            "is_computer_use_approved",
            AsyncMock(return_value=False),
        )

        out = await rt._h_dry_run_skill(
            {"recording_id": "rec_x", "data_rows": [{"a": 1}]}
        )

        assert out["isError"] is True
        assert _parse(out)["code"] == "COMPUTER_USE_CONSENT_REQUIRED"

    @pytest.mark.asyncio
    async def test_dry_run_refuses_when_the_deployment_turns_computer_use_off(
        self, monkeypatch
    ):
        """Replay drives the mouse and keyboard, but it checked only the
        session's consent, so it ran with computer use turned off."""
        shim = _shim()
        _install(monkeypatch, shim)
        monkeypatch.setattr(
            rt, "_chat_config", SimpleNamespace(allow_computer_use=False), raising=False
        )

        out = await rt._h_dry_run_skill(
            {"recording_id": "rec_x", "data_rows": [{"a": 1}]}
        )

        assert out["isError"] is True
        assert _parse(out)["code"] == "COMPUTER_USE_DISABLED"

    @pytest.mark.asyncio
    async def test_dry_run_refuses_a_shim_without_computer_use(self, monkeypatch):
        shim = _shim(capabilities=("recording",))
        _install(monkeypatch, shim)

        out = await rt._h_dry_run_skill(
            {"recording_id": "rec_x", "data_rows": [{"a": 1}]}
        )

        assert out["isError"] is True
        assert _parse(out)["code"] == "CAPABILITY_NOT_GRANTED"

    @pytest.mark.asyncio
    async def test_dry_run_refuses_without_the_input_it_replays(self, monkeypatch):
        """Otto's computer-use calls check the tools the shim offers; replay
        calls the shim directly, so it checks for the ones it replays."""
        shim = _shim()
        shim.capture_connection_guard.return_value = ShimConnectionGuard(
            generation=1,
            machine_id="machine-1",
            computer_use_driver="cua-driver 0.38.0",
            computer_use_tools=("get_window_state",),
        )
        _install(monkeypatch, shim)

        out = await rt._h_dry_run_skill(
            {"recording_id": "rec_x", "data_rows": [{"a": 1}]}
        )

        body = _parse(out)
        assert body["code"] == "CAPABILITY_NOT_GRANTED"
        assert body["details"]["missing"] == ["click", "type_text"]

    @pytest.mark.asyncio
    async def test_dry_run_requires_generated_skill(self, monkeypatch):
        shim = _shim()
        _install(monkeypatch, shim)
        out = await rt._h_dry_run_skill(
            {"recording_id": "rec_x", "data_rows": [{"a": 1}]}
        )
        assert out["isError"] is True
        assert _parse(out)["code"] == "SKILL_NOT_GENERATED"

    @pytest.mark.asyncio
    async def test_dry_run_blocks_unconfirmed_skill(self, monkeypatch):
        shim = _shim()
        shim.recording.fetch = AsyncMock(return_value=_single_row_recording())
        _install(monkeypatch, shim)
        _mark_reviewed(shim, "rec_1")
        await rt._h_generate_skill({"recording_id": "rec_1"})  # unconfirmed
        out = await rt._h_dry_run_skill(
            {"recording_id": "rec_1", "data_rows": [{"first_name": "A"}]}
        )
        assert out["isError"] is True
        assert _parse(out)["code"] == "SKILL_NEEDS_CONFIRMATION"

    @pytest.mark.asyncio
    async def test_dry_run_requires_rows(self, monkeypatch):
        shim = _shim()
        shim.recording.fetch = AsyncMock(return_value=_two_row_recording())
        _install(monkeypatch, shim)
        _mark_reviewed(shim, "rec_2")
        await rt._h_generate_skill({"recording_id": "rec_2"})  # confirmed (2 rows)
        out = await rt._h_dry_run_skill({"recording_id": "rec_2", "data_rows": []})
        assert out["isError"] is True
        assert _parse(out)["code"] == "INVALID_ARGUMENT"

    @pytest.mark.asyncio
    async def test_dry_run_executes_over_rows(self, monkeypatch):
        shim = _shim()
        shim.recording.fetch = AsyncMock(return_value=_two_row_recording())
        _install(monkeypatch, shim)
        _mark_reviewed(shim, "rec_2")
        await rt._h_generate_skill({"recording_id": "rec_2"})
        out = await rt._h_dry_run_skill(
            {
                "recording_id": "rec_2",
                "data_rows": [{"first_name": "Alice"}, {"first_name": "Bob"}],
            }
        )
        body = _parse(out)
        assert out["isError"] is False
        assert body["rows_attempted"] == 2
        assert body["destructive_blocked"] is True  # submit skipped by default
        guard = shim.capture_connection_guard.return_value
        typed = [
            call
            for call in shim.computer.call.await_args_list
            if call.args[0] == "type_text"
        ]
        assert typed
        assert all(call.kwargs["_guard"] == guard for call in typed)
        assert all(
            call.args[1]["target"] == {"kind": "desktop", "display_id": "primary"}
            for call in typed
        )
