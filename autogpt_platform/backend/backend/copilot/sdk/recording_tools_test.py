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


def _shim(*, capabilities=("recording",)):
    # spec=LocalPCShim so the handler's isinstance(sb, LocalPCShim) gate passes.
    shim = MagicMock(spec=LocalPCShim)
    shim.sandbox_id = "sess-1"
    shim.machine_id = "machine-1"
    shim.capabilities = list(capabilities)
    shim.computer_use_features_coarse = ["screenshot", "input"]
    shim.computer_use_features = ["screenshot.capture", "input.click"]
    guard = ShimConnectionGuard(
        generation=1,
        machine_id="machine-1",
        computer_use_features_coarse=("screenshot", "input"),
        computer_use_features=("screenshot.capture", "input.click"),
    )
    shim.capture_connection_guard.return_value = guard
    shim.connection_guard_matches.return_value = True
    shim.recording = MagicMock()
    shim.computer = MagicMock()
    shim.computer.type = AsyncMock()
    shim.computer.click = AsyncMock()
    shim.computer.clipboard_read = AsyncMock(return_value=None)
    shim.close_recording = MagicMock()
    return shim


def _install(monkeypatch, shim):
    monkeypatch.setattr(rt, "get_current_sandbox", lambda: shim)
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
        shim.computer.type.assert_not_awaited()
        shim.computer.click.assert_not_awaited()

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
        assert shim.computer.type.await_args_list
        assert all(
            call.kwargs["_guard"] == guard
            for call in shim.computer.type.await_args_list
        )
