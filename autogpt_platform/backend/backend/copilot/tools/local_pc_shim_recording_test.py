"""Tests for the LocalPCShim workflow-recording surface (_RecordingProxy +
RECORDING_STEP buffering). See WORKFLOW_RECORDING.md §6."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from .local_pc_shim import LocalPCShim, ShimRecordingError, _RecordingProxy
from .recording_models import RecordingSummary, TrajectoryStep, WorkflowRecording


def _make_recording_shim(rpc_return: dict | None = None) -> LocalPCShim:
    """Build a LocalPCShim with _rpc stubbed and the recording surface wired."""
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "test-session"
    shim.allowed_root = "/Users/test/workspace"
    shim.machine_id = "machine-uuid"
    shim.platform = "darwin"
    shim.arch = "arm64"
    shim.capabilities = ["shell", "files", "recording"]
    shim.computer_use_features = []
    shim._recording_steps = {}
    if rpc_return is not None:
        shim._rpc = AsyncMock(return_value=rpc_return)
    shim.recording = _RecordingProxy(shim)
    return shim


class TestRecordingStart:
    @pytest.mark.asyncio
    async def test_start_sends_consent_token_and_returns_id(self):
        shim = _make_recording_shim(
            {"type": "RECORDING_STARTED", "payload": {"recording_id": "rec_abc"}}
        )
        rec_id = await shim.recording.start(
            mode="copilot",
            interpretation_route="extract_then_cloud",
            channels=["floor", "browser"],
            consent_token="tok-123",
        )
        assert rec_id == "rec_abc"
        shim._rpc.assert_awaited_once_with(
            "START_RECORDING",
            {
                "mode": "copilot",
                "interpretation_route": "extract_then_cloud",
                "channels": ["floor", "browser"],
                "consent_token": "tok-123",
            },
        )

    @pytest.mark.asyncio
    async def test_start_precreates_step_buffer(self):
        """A fast first RECORDING_STEP can't race the START response."""
        shim = _make_recording_shim(
            {"type": "RECORDING_STARTED", "payload": {"recording_id": "rec_abc"}}
        )
        await shim.recording.start(
            mode="copilot",
            interpretation_route="extract_then_cloud",
            channels=["floor"],
            consent_token="tok",
        )
        assert "rec_abc" in shim._recording_steps

    @pytest.mark.asyncio
    async def test_start_consent_required_raises_typed_error(self):
        shim = _make_recording_shim(
            {
                "type": "ERROR",
                "payload": {"code": "CONSENT_REQUIRED", "message": "no token"},
            }
        )
        with pytest.raises(ShimRecordingError) as exc:
            await shim.recording.start(
                mode="copilot",
                interpretation_route="extract_then_cloud",
                channels=["floor"],
                consent_token="",
            )
        assert exc.value.code == "CONSENT_REQUIRED"
        # Translated to LLM-friendly English, not the raw enum.
        assert "consent token" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_start_already_active_raises_typed_error(self):
        shim = _make_recording_shim(
            {
                "type": "ERROR",
                "payload": {
                    "code": "RECORDING_ALREADY_ACTIVE",
                    "message": "busy",
                    "details": {"recording_id": "rec_existing"},
                },
            }
        )
        with pytest.raises(ShimRecordingError) as exc:
            await shim.recording.start(
                mode="demonstration",
                interpretation_route="extract_then_cloud",
                channels=["floor"],
                consent_token="tok",
            )
        assert exc.value.code == "RECORDING_ALREADY_ACTIVE"
        assert exc.value.details["recording_id"] == "rec_existing"


class TestRecordingStop:
    @pytest.mark.asyncio
    async def test_stop_returns_summary(self):
        shim = _make_recording_shim(
            {
                "type": "RECORDING_SUMMARY",
                "payload": {
                    "recording_id": "rec_abc",
                    "step_count": 14,
                    "enrichment_coverage": {"dom": 11, "ax": 0, "none": 3},
                    "duration_seconds": 47.2,
                },
            }
        )
        summary = await shim.recording.stop("rec_abc")
        assert isinstance(summary, RecordingSummary)
        assert summary.step_count == 14
        assert summary.enrichment_coverage == {"dom": 11, "ax": 0, "none": 3}
        assert summary.duration_seconds == 47.2

    @pytest.mark.asyncio
    async def test_stop_not_found_raises(self):
        shim = _make_recording_shim(
            {
                "type": "ERROR",
                "payload": {
                    "code": "RECORDING_NOT_FOUND",
                    "message": "gone",
                    "details": {"recording_id": "rec_missing"},
                },
            }
        )
        with pytest.raises(ShimRecordingError) as exc:
            await shim.recording.stop("rec_missing")
        assert exc.value.code == "RECORDING_NOT_FOUND"


class TestRecordingFetch:
    @pytest.mark.asyncio
    async def test_fetch_parses_full_recording(self):
        shim = _make_recording_shim(
            {
                "type": "RECORDING_DATA",
                "payload": {
                    "recording_id": "rec_abc",
                    "version": "1.0",
                    "created_at": 1712345678.0,
                    "machine_id": "m1",
                    "interpretation_route": "extract_then_cloud",
                    "redaction_applied": True,
                    "steps": [
                        {
                            "seq": 1,
                            "action": "fill",
                            "screenshot_ref": "stub_1",
                            "cursor": [840, 314],
                            "active_app": "Google Chrome",
                            "enrichment": {
                                "kind": "dom",
                                "selectors": [{"strategy": "id", "value": "#fn"}],
                                "label": "First Name",
                            },
                            "value": {
                                "raw": "John",
                                "type": "text",
                                "is_parameter": None,
                            },
                        }
                    ],
                },
            }
        )
        rec = await shim.recording.fetch("rec_abc")
        assert isinstance(rec, WorkflowRecording)
        assert rec.recording_id == "rec_abc"
        assert rec.redaction_applied is True
        assert len(rec.steps) == 1
        step = rec.steps[0]
        assert step.action == "fill"
        assert step.enrichment.kind == "dom"
        assert step.enrichment.selectors[0]["value"] == "#fn"
        assert step.value is not None
        assert step.value.raw == "John"
        assert step.value.is_parameter is None  # not yet confirmed (§8)


class TestRecordingStepStream:
    @pytest.mark.asyncio
    async def test_recv_step_buffers_per_recording(self):
        shim = _make_recording_shim()
        shim._handle_recording_step(
            {
                "type": "RECORDING_STEP",
                "payload": {
                    "recording_id": "rec_abc",
                    "step": {"seq": 1, "action": "click", "screenshot_ref": "s1"},
                },
            }
        )
        queue = shim._recording_steps["rec_abc"]
        step = queue.get_nowait()
        assert isinstance(step, TrajectoryStep)
        assert step.action == "click"

    @pytest.mark.asyncio
    async def test_stream_steps_yields_then_stops_on_close(self):
        shim = _make_recording_shim()
        rec_id = "rec_abc"

        async def produce():
            await asyncio.sleep(0)
            shim._handle_recording_step(
                {
                    "type": "RECORDING_STEP",
                    "payload": {
                        "recording_id": rec_id,
                        "step": {"seq": 1, "action": "fill"},
                    },
                }
            )
            shim._handle_recording_step(
                {
                    "type": "RECORDING_STEP",
                    "payload": {
                        "recording_id": rec_id,
                        "step": {"seq": 2, "action": "submit"},
                    },
                }
            )
            shim.close_recording(rec_id)

        collected: list[TrajectoryStep] = []

        async def consume():
            async for step in shim.recording.stream_steps(rec_id):
                collected.append(step)

        await asyncio.gather(produce(), asyncio.wait_for(consume(), timeout=2))
        assert [s.action for s in collected] == ["fill", "submit"]
        # Buffer is dropped after the iterator finishes.
        assert rec_id not in shim._recording_steps

    @pytest.mark.asyncio
    async def test_step_without_recording_id_is_dropped(self):
        shim = _make_recording_shim()
        shim._handle_recording_step(
            {"type": "RECORDING_STEP", "payload": {"step": {"action": "click"}}}
        )
        assert shim._recording_steps == {}

    @pytest.mark.asyncio
    async def test_step_without_body_is_dropped(self):
        shim = _make_recording_shim()
        # Pre-create the buffer as START would, so we can assert nothing
        # was enqueued for a malformed (body-less) frame.
        shim._ensure_recording_buffer("rec_abc")
        shim._handle_recording_step(
            {"type": "RECORDING_STEP", "payload": {"recording_id": "rec_abc"}}
        )
        assert shim._recording_steps["rec_abc"].empty()
