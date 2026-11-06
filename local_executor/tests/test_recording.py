"""
Tests for the workflow-recording subsystem (docs/WORKFLOW_RECORDING.md).

Mock-only. The OS-specific input-observation hooks are NOT exercised here —
they live behind the CaptureSource seam and are driven by MockCaptureSource.
Covers: protocol round-trips, RecordingSession lifecycle, consent tokens, route
probing, redaction (incl. the documented PII hygiene gap), one-at-a-time,
content-free audit, demonstration vs co-pilot streaming, and an end-to-end
scripted recording.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import RecordingHandler
from autogpt_local_executor.protocol import (
    ApplyRecordingReviewMessage,
    ApplyRecordingReviewPayload,
    EnrichmentCoverage,
    ErrorCode,
    ErrorMessage,
    RecordingConsentResultMessage,
    RecordingDataMessage,
    RecordingFetchMessage,
    RecordingFetchPayload,
    RecordingReviewAppliedMessage,
    RecordingStartedMessage,
    RecordingStepMessage,
    RecordingSummaryMessage,
    RequestRecordingConsentMessage,
    RequestRecordingConsentPayload,
    StartRecordingMessage,
    StartRecordingPayload,
    StepEnrichment,
    StepValue,
    StopRecordingMessage,
    StopRecordingPayload,
    TrajectoryStep,
    WorkflowRecording,
    dump_message,
    new_id,
    now_ts,
    parse_message,
)
from autogpt_local_executor.recording import (
    ConsentBroker,
    ConsentError,
    MockCaptureSource,
    RecordingSession,
    ScreenshotActionFloor,
    probe_interpretation_route,
    redact_step,
)
from autogpt_local_executor.recording.crypto import RecordingCipher

# Deterministic at-rest key so the session's encrypted buffer never touches the
# keychain or the passphrase fallback during tests.
_TEST_CIPHER = RecordingCipher(b"\x01" * 32)
_AUDIT_KEY = b"\x00" * 32


# ── Helpers ───────────────────────────────────────────────────────────────────


def _step(
    seq: int,
    *,
    action: str = "fill",
    raw: str | None = "John",
    value_type: str = "text",
    kind: str = "none",
    role: str | None = None,
    label: str | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        seq=seq,
        ts=float(seq),
        action=action,  # type: ignore[arg-type]
        screenshot_ref=f"stub_{seq}",
        cursor=(10 * seq, 20 * seq),
        active_app="Google Chrome",
        active_window="New customer — Acme",
        enrichment=StepEnrichment(kind=kind, role=role, label=label),  # type: ignore[arg-type]
        value=StepValue(raw=raw, type=value_type),  # type: ignore[arg-type]
    )


def _make_config(tmp_path: Path, *, enable_recording: bool = True) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        recording_buffer_dir=tmp_path / "recordings",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        enable_recording=enable_recording,
    )


def _make_session(tmp_path: Path, *, mode: str = "demonstration") -> RecordingSession:
    return RecordingSession(
        machine_id="test-machine",
        mode=mode,
        interpretation_route="extract_then_cloud",
        channels=["floor"],
        buffer_dir=tmp_path / "recordings",
        cipher=_TEST_CIPHER,
    )


def _audit(tmp_path: Path) -> AuditWriter:
    return AuditWriter(
        path=tmp_path / "audit.log",
        audit_key=_AUDIT_KEY,
        shim_version="test-0.0",
        machine_id="test-machine",
        session_id="test-session",
    )


def _capture_factory(steps: list[TrajectoryStep]):
    def factory(_session: RecordingSession) -> MockCaptureSource:
        return MockCaptureSource(steps)

    return factory


def _consent(broker: ConsentBroker, mode: str = "copilot") -> str:
    return broker.issue_consent_token(mode=mode, interpretation_route="extract_then_cloud")


# With only the floor channel (no structured text, no OCR, no local model),
# screenshots_to_cloud is the one interpretation route the probe can offer.
FLOOR_ROUTE = "screenshots_to_cloud"


def _floor_consent(broker: ConsentBroker, mode: str = "copilot") -> str:
    return broker.issue_consent_token(mode=mode, interpretation_route=FLOOR_ROUTE)


def _approve_consent(broker: ConsentBroker, *, mode: str, interpretation_route: str) -> str:
    return broker.issue_consent_token(mode=mode, interpretation_route=interpretation_route)


def _deny_consent(broker: ConsentBroker, *, mode: str, interpretation_route: str) -> None:
    return None


class _OsProxy(types.ModuleType):
    """The real `os` module with some functions replaced, for one module under test."""

    def __init__(self, **overrides: object) -> None:
        super().__init__("os")
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)


def test_production_capture_factory_wires_macos_input_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autogpt_local_executor import platform_info
    from autogpt_local_executor.recording import A11yEnricher, ScreenshotActionFloor
    from autogpt_local_executor.recording.macos_capture import MacInputCaptureSource

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setattr(platform_info, "available_recording_channels", lambda: ["floor"])
    monkeypatch.setattr(MacInputCaptureSource, "start", lambda self: True)
    handler = RecordingHandler(_make_config(tmp_path), recording_cipher=_TEST_CIPHER)

    source = handler._default_capture_factory(_make_session(tmp_path))

    assert isinstance(source, A11yEnricher)
    assert isinstance(source._floor, ScreenshotActionFloor)
    assert isinstance(source._floor._input_events, MacInputCaptureSource)


# ── Protocol round-trips (each new message type) ──────────────────────────────


def test_start_recording_round_trip() -> None:
    msg = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            mode="copilot",
            interpretation_route="extract_then_cloud",
            channels=["floor", "browser"],
            consent_token="tok",
        ),
    )
    again = parse_message(dump_message(msg))
    assert isinstance(again, StartRecordingMessage)
    assert again.payload.mode == "copilot"
    assert again.payload.channels == ["floor", "browser"]


def test_request_recording_consent_round_trip() -> None:
    msg = RequestRecordingConsentMessage(
        id=new_id(),
        ts=now_ts(),
        payload=RequestRecordingConsentPayload(
            mode="copilot",
            interpretation_route="screenshots_to_cloud",
            channels=["floor"],
        ),
    )
    again = parse_message(dump_message(msg))
    assert isinstance(again, RequestRecordingConsentMessage)
    assert again.payload.interpretation_route == "screenshots_to_cloud"


def test_start_recording_requires_consent_token() -> None:
    """consent_token has no default — the platform must supply it (§9)."""
    with pytest.raises(Exception):
        StartRecordingPayload(mode="copilot")  # type: ignore[call-arg]


def test_recording_started_round_trip() -> None:
    msg = RecordingStartedMessage(
        id=new_id(),
        ts=now_ts(),
        payload={"recording_id": "rec_1"},  # type: ignore[arg-type]
    )
    again = parse_message(dump_message(msg))
    assert isinstance(again, RecordingStartedMessage)
    assert again.payload.recording_id == "rec_1"


def test_stop_recording_round_trip() -> None:
    msg = StopRecordingMessage(
        id=new_id(), ts=now_ts(), payload=StopRecordingPayload(recording_id="rec_1")
    )
    again = parse_message(dump_message(msg))
    assert isinstance(again, StopRecordingMessage)
    assert again.payload.recording_id == "rec_1"


def test_recording_summary_round_trip() -> None:
    msg = RecordingSummaryMessage(
        id=new_id(),
        ts=now_ts(),
        payload={  # type: ignore[arg-type]
            "recording_id": "rec_1",
            "step_count": 14,
            "enrichment_coverage": {"dom": 11, "ax": 0, "none": 3},
            "duration_seconds": 47.2,
        },
    )
    again = parse_message(dump_message(msg))
    assert isinstance(again, RecordingSummaryMessage)
    assert again.payload.step_count == 14
    assert again.payload.enrichment_coverage == EnrichmentCoverage(dom=11, ax=0, none=3)


def test_recording_fetch_round_trip() -> None:
    msg = RecordingFetchMessage(
        id=new_id(), ts=now_ts(), payload=RecordingFetchPayload(recording_id="rec_1")
    )
    again = parse_message(dump_message(msg))
    assert isinstance(again, RecordingFetchMessage)
    assert again.payload.recording_id == "rec_1"


def test_apply_recording_review_round_trip() -> None:
    msg = ApplyRecordingReviewMessage(
        id=new_id(),
        ts=now_ts(),
        payload=ApplyRecordingReviewPayload(
            recording_id="rec_1",
            removed_step_seqs=[2],
            redacted_step_seqs=[1],
        ),
    )
    again = parse_message(dump_message(msg))
    assert isinstance(again, ApplyRecordingReviewMessage)
    assert again.payload.removed_step_seqs == [2]
    assert again.payload.redacted_step_seqs == [1]


def test_recording_data_round_trip() -> None:
    rec = WorkflowRecording(
        recording_id="rec_1",
        created_at=1.0,
        machine_id="m",
        interpretation_route="extract_then_cloud",
        steps=[_step(1, kind="dom")],
    )
    msg = RecordingDataMessage(id=new_id(), ts=now_ts(), payload={"recording": rec})  # type: ignore[arg-type]
    again = parse_message(dump_message(msg))
    assert isinstance(again, RecordingDataMessage)
    assert again.payload.recording.steps[0].action == "fill"
    assert again.payload.recording.steps[0].screenshot_ref == "stub_1"


def test_recording_step_round_trip() -> None:
    msg = RecordingStepMessage(
        id=new_id(),
        ts=now_ts(),
        payload={"recording_id": "rec_1", "step": _step(7)},  # type: ignore[arg-type]
    )
    again = parse_message(dump_message(msg))
    assert isinstance(again, RecordingStepMessage)
    assert again.payload.step.seq == 7
    assert again.payload.step.cursor == (70, 140)


def test_trajectory_step_floor_fields_always_present() -> None:
    """The floor fields are required — a step without them won't parse."""
    raw = json.dumps(
        {"seq": 1, "ts": 1.0, "action": "click", "screenshot_ref": "s"}
    )  # missing cursor/active_app/active_window
    with pytest.raises(Exception):
        TrajectoryStep.model_validate_json(raw)


def test_new_recording_error_codes_exist() -> None:
    for code in (
        "RECORDING_NOT_FOUND",
        "RECORDING_CHANNEL_UNAVAILABLE",
        "RECORDING_ALREADY_ACTIVE",
        "CONSENT_REQUIRED",
        "INTERPRETATION_UNAVAILABLE",
    ):
        assert hasattr(ErrorCode, code)


# ── RecordingSession lifecycle ────────────────────────────────────────────────


def test_session_lifecycle_start_steps_stop_summary_fetch(tmp_path: Path) -> None:
    sess = _make_session(tmp_path)
    sess.start()
    sess.append(_step(1, kind="dom"))
    sess.append(_step(2, kind="ax"))
    sess.append(_step(3, kind="none"))
    sess.stop()

    assert sess.step_count == 3
    cov = sess.enrichment_coverage()
    assert cov == EnrichmentCoverage(dom=1, ax=1, none=1)
    assert sess.duration_seconds() >= 0.0

    rec = sess.to_recording()
    assert isinstance(rec, WorkflowRecording)
    assert rec.recording_id == sess.recording_id
    assert rec.redaction_applied is True
    assert len(rec.steps) == 3


def test_session_recording_id_minted_and_unique(tmp_path: Path) -> None:
    ids = {_make_session(tmp_path).recording_id for _ in range(50)}
    assert len(ids) == 50
    assert all(rid.startswith("rec_") for rid in ids)


def test_session_append_after_stop_raises(tmp_path: Path) -> None:
    from autogpt_local_executor.recording import RecordingError

    sess = _make_session(tmp_path)
    sess.start()
    sess.stop()
    with pytest.raises(RecordingError):
        sess.append(_step(1))


def test_session_review_removes_and_redacts_authoritative_steps(tmp_path: Path) -> None:
    sess = _make_session(tmp_path)
    sess.start()
    sess.append(_step(1, raw="keep but hide"))
    sess.append(_step(2, raw="remove"))
    sess.append(_step(3, raw="keep"))
    sess.stop()

    count = sess.apply_review(removed_step_seqs=[2], redacted_step_seqs=[1])

    recording = sess.to_recording()
    assert count == 2
    assert [step.seq for step in recording.steps] == [1, 3]
    assert recording.steps[0].value.raw is None
    assert recording.steps[0].value.type == "secret"
    assert recording.steps[0].redacted is True


def test_session_exact_repeated_review_is_idempotent(tmp_path: Path) -> None:
    sess = _make_session(tmp_path)
    sess.start()
    sess.append(_step(1, raw="hide"))
    sess.append(_step(2, raw="remove"))
    sess.stop()

    first = sess.apply_review(removed_step_seqs=[2], redacted_step_seqs=[1])
    second = sess.apply_review(removed_step_seqs=[2], redacted_step_seqs=[1])

    assert first == second == 1
    assert [step.seq for step in sess.to_recording().steps] == [1]


def test_session_review_retry_after_a_later_review_is_idempotent(tmp_path: Path) -> None:
    sess = _make_session(tmp_path)
    sess.start()
    sess.append(_step(1, raw="hide"))
    sess.append(_step(2, raw="remove first"))
    sess.append(_step(3, raw="remove second"))
    sess.stop()

    first = sess.apply_review(removed_step_seqs=[2], redacted_step_seqs=[1])
    second = sess.apply_review(removed_step_seqs=[3], redacted_step_seqs=[])
    retry = sess.apply_review(removed_step_seqs=[2], redacted_step_seqs=[1])
    redact_removed = sess.apply_review(removed_step_seqs=[], redacted_step_seqs=[2, 3])

    assert (first, second, retry, redact_removed) == (2, 1, 1, 1)
    steps = sess.to_recording().steps
    assert [step.seq for step in steps] == [1]
    assert steps[0].value.raw is None
    assert steps[0].redacted is True


def test_session_review_rejects_unknown_step(tmp_path: Path) -> None:
    from autogpt_local_executor.recording import RecordingError

    sess = _make_session(tmp_path)
    sess.start()
    sess.append(_step(1))
    sess.stop()
    with pytest.raises(RecordingError):
        sess.apply_review(removed_step_seqs=[99], redacted_step_seqs=[])


@pytest.mark.parametrize("pin", [False, True])
def test_close_erases_a_snapshot_left_by_a_failed_buffer_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: bool
) -> None:
    import autogpt_local_executor.recording.session as session_mod

    def refuse_replace(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("the buffer file is locked")

    buffer_dir = tmp_path / "recordings"
    sess = _make_session(tmp_path)
    sess.start()
    sess.append(_step(1))
    published = set(buffer_dir.iterdir())
    with monkeypatch.context() as patch:
        patch.setattr(session_mod, "os", _OsProxy(replace=refuse_replace))
        sess.append(_step(2))
    assert len(set(buffer_dir.iterdir()) - published) == 1, "expected a leftover snapshot"
    sess.stop()

    sess.close(pin=pin)

    assert set(buffer_dir.iterdir()) == (published if pin else set())


# ── Screenshot floor cursor (§4) ──────────────────────────────────────────────


class _PointerBackend:
    def cursor_position(self) -> tuple[int, int, int]:
        return (500, 400, 0)


def _click_at(seq: int, cursor: tuple[int, int]) -> TrajectoryStep:
    return _step(seq, action="click").model_copy(update={"cursor": cursor})


async def _floor_cursors(tmp_path: Path, source: MockCaptureSource) -> list[tuple[int, int]]:
    floor = ScreenshotActionFloor(
        input_events=source, config=_make_config(tmp_path), backend=_PointerBackend()
    )
    return [step.cursor async for step in floor.steps()]


async def test_floor_keeps_a_click_at_the_display_origin(tmp_path: Path) -> None:
    source = MockCaptureSource([_click_at(1, (0, 0)), _click_at(2, (840, 314))])

    assert await _floor_cursors(tmp_path, source) == [(0, 0), (840, 314)]


async def test_floor_fills_cursor_for_a_source_that_cannot_see_the_pointer(
    tmp_path: Path,
) -> None:
    source = MockCaptureSource([_click_at(1, (0, 0)), _click_at(2, (840, 314))])
    source.reports_cursor = False

    assert await _floor_cursors(tmp_path, source) == [(500, 400), (500, 400)]


# ── Consent tokens (§9) ───────────────────────────────────────────────────────


def test_consent_token_issue_then_validate() -> None:
    broker = ConsentBroker()
    tok = broker.issue_consent_token(mode="copilot", interpretation_route="extract_then_cloud")
    body = broker.validate_consent_token(tok, mode="copilot")
    assert body["mode"] == "copilot"


def test_consent_token_single_use() -> None:
    broker = ConsentBroker()
    tok = _consent(broker)
    broker.validate_consent_token(tok)
    with pytest.raises(ConsentError):
        broker.validate_consent_token(tok)


def test_consent_token_platform_cannot_forge() -> None:
    """A token a different process minted (different key) is rejected — the
    platform cannot self-assert consent (§9)."""
    minter = ConsentBroker()
    validator = ConsentBroker()  # different process / different key
    tok = _consent(minter)
    with pytest.raises(ConsentError):
        validator.validate_consent_token(tok)


def test_consent_token_missing_or_garbage() -> None:
    broker = ConsentBroker()
    with pytest.raises(ConsentError):
        broker.validate_consent_token(None)
    with pytest.raises(ConsentError):
        broker.validate_consent_token("not-a-token")


def test_consent_token_mode_mismatch() -> None:
    broker = ConsentBroker()
    tok = broker.issue_consent_token(mode="demonstration", interpretation_route="local_vlm")
    with pytest.raises(ConsentError):
        broker.validate_consent_token(tok, mode="copilot")


# ── Route probing (§3.1) ──────────────────────────────────────────────────────


def test_route_probe_structured_channels_to_extract_then_cloud() -> None:
    d = probe_interpretation_route(channels=["floor", "browser"])
    assert d.route == "extract_then_cloud"
    assert d.requires_consent is False


def test_route_probe_ocr_branch(monkeypatch) -> None:
    import autogpt_local_executor.recording.route as route_mod

    monkeypatch.setattr(route_mod, "_ocr_available", lambda: True)
    d = route_mod.probe_interpretation_route(channels=["floor"])
    assert d.route == "extract_then_cloud"
    assert d.requires_consent is False


def test_route_probe_local_vlm_branch(monkeypatch) -> None:
    import autogpt_local_executor.recording.route as route_mod

    monkeypatch.setattr(route_mod, "_ocr_available", lambda: False)
    d = route_mod.probe_interpretation_route(channels=["floor"], local_llm_models=["llava:13b"])
    assert d.route == "local_vlm"
    assert d.requires_consent is False


def test_route_probe_screenshots_to_cloud_requires_consent(monkeypatch) -> None:
    import autogpt_local_executor.recording.route as route_mod

    monkeypatch.setattr(route_mod, "_ocr_available", lambda: False)
    d = route_mod.probe_interpretation_route(channels=["floor"], local_llm_models=[])
    assert d.route == "screenshots_to_cloud"
    assert d.requires_consent is True


# ── Redaction (§9) — hygiene, with the documented PII gap ─────────────────────


def test_redaction_strips_secret_field() -> None:
    step = _step(1, raw="hunter2", kind="ax", label="Password", role="textbox")
    out = redact_step(step)
    assert out.value.raw is None
    assert out.value.type == "secret"
    assert out.redacted is True


def test_redaction_leaves_non_secret_pii_intact() -> None:
    """KNOWN HYGIENE GAP (§9): a non-secret PII field (e.g. an email, or an SSN
    typed into a generic 'Number' field) is NOT stripped. Redaction is
    best-effort hygiene, not a privacy control — interpretation_route is. This
    test pins that documented behavior so a future change to it is deliberate.
    """
    email = _step(1, raw="alice@example.com", value_type="email", kind="dom", label="Email")
    out = redact_step(email)
    assert out.value.raw == "alice@example.com"  # NOT stripped — the gap
    assert out.redacted is False

    ssn_in_generic = _step(2, raw="123-45-6789", value_type="text", kind="ax", label="Number")
    out2 = redact_step(ssn_in_generic)
    assert out2.value.raw == "123-45-6789"  # slips the deny-list — the gap
    assert out2.redacted is False


def test_redaction_detects_secret_by_value_type() -> None:
    step = _step(1, raw="topsecret", value_type="secret")
    out = redact_step(step)
    assert out.value.raw is None
    assert out.redacted is True


def test_session_to_recording_applies_redaction(tmp_path: Path) -> None:
    sess = _make_session(tmp_path)
    sess.start()
    sess.append(_step(1, raw="John", kind="dom", label="First Name"))
    sess.append(_step(2, raw="hunter2", kind="dom", label="Password", role="textbox"))
    sess.stop()
    rec = sess.to_recording()
    assert rec.steps[0].value.raw == "John"
    assert rec.steps[1].value.raw is None
    assert rec.steps[1].redacted is True


# ── One-at-a-time (RECORDING_ALREADY_ACTIVE) ──────────────────────────────────


async def test_one_recording_at_a_time(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    broker = ConsentBroker()
    # Long-lived capture so the first session stays active across the 2nd START.
    handler = RecordingHandler(
        cfg,
        consent_broker=broker,
        capture_factory=_capture_factory([]),  # empty → session stays active until STOP
        recording_cipher=_TEST_CIPHER,
    )

    start1 = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            interpretation_route=FLOOR_ROUTE,
            mode="copilot",
            channels=["floor"],
            consent_token=_floor_consent(broker),
        ),
    )
    r1 = await handler.handle(start1, send=None)
    assert isinstance(r1, RecordingStartedMessage)

    start2 = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            interpretation_route=FLOOR_ROUTE,
            mode="copilot",
            channels=["floor"],
            consent_token=_floor_consent(broker),
        ),
    )
    r2 = await handler.handle(start2, send=None)
    assert isinstance(r2, ErrorMessage)
    assert r2.payload.code == ErrorCode.RECORDING_ALREADY_ACTIVE


# ── START consent enforcement via the handler ─────────────────────────────────


async def test_request_consent_approved_token_starts_recording(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    broker = ConsentBroker()
    handler = RecordingHandler(
        cfg,
        consent_broker=broker,
        consent_prompt=_approve_consent,
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    request = RequestRecordingConsentMessage(
        id=new_id(),
        ts=now_ts(),
        payload=RequestRecordingConsentPayload(mode="copilot", channels=["floor"]),
    )
    consent = await handler.handle(request, send=None)
    assert isinstance(consent, RecordingConsentResultMessage)
    assert consent.payload.approved is True
    assert consent.payload.consent_token
    assert consent.payload.expires_at

    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            mode=consent.payload.mode,
            interpretation_route=consent.payload.interpretation_route,
            channels=["floor"],
            consent_token=consent.payload.consent_token,
        ),
    )
    response = await handler.handle(start, send=None)
    assert isinstance(response, RecordingStartedMessage)


async def test_request_consent_denial_returns_no_token(tmp_path: Path) -> None:
    handler = RecordingHandler(
        _make_config(tmp_path),
        consent_broker=ConsentBroker(),
        consent_prompt=_deny_consent,
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    request = RequestRecordingConsentMessage(
        id=new_id(),
        ts=now_ts(),
        payload=RequestRecordingConsentPayload(mode="copilot", channels=["floor"]),
    )
    consent = await handler.handle(request, send=None)
    assert isinstance(consent, RecordingConsentResultMessage)
    assert consent.payload.approved is False
    assert consent.payload.consent_token is None
    assert consent.payload.expires_at is None


async def test_request_consent_rejects_unavailable_channel(tmp_path: Path) -> None:
    prompted = False

    def prompt(*_args, **_kwargs):
        nonlocal prompted
        prompted = True
        return None

    handler = RecordingHandler(
        _make_config(tmp_path),
        consent_prompt=prompt,
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    response = await handler.handle(
        RequestRecordingConsentMessage(
            id=new_id(),
            ts=now_ts(),
            payload=RequestRecordingConsentPayload(mode="copilot", channels=["browser"]),
        ),
        send=None,
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.RECORDING_CHANNEL_UNAVAILABLE
    assert prompted is False


async def test_handler_uses_daemon_local_model_list_for_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogpt_local_executor.recording.route as route_mod

    monkeypatch.setattr(route_mod, "_ocr_available", lambda: False)
    handler = RecordingHandler(
        _make_config(tmp_path),
        consent_prompt=_approve_consent,
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    handler.set_local_llm_models(["llava:13b"])

    response = await handler.handle(
        RequestRecordingConsentMessage(
            id=new_id(),
            ts=now_ts(),
            payload=RequestRecordingConsentPayload(mode="copilot", channels=["floor"]),
        ),
        send=None,
    )

    assert isinstance(response, RecordingConsentResultMessage)
    assert response.payload.interpretation_route == "local_vlm"


async def test_start_without_valid_token_returns_consent_required(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    handler = RecordingHandler(
        cfg,
        consent_broker=ConsentBroker(),
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(mode="copilot", channels=["floor"], consent_token="forged"),
    )
    resp = await handler.handle(start, send=None)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.CONSENT_REQUIRED


async def test_start_rejects_token_for_different_route(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    broker = ConsentBroker()
    handler = RecordingHandler(
        cfg,
        consent_broker=broker,
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            mode="copilot",
            interpretation_route="screenshots_to_cloud",
            channels=["floor"],
            consent_token=_consent(broker),
        ),
    )
    response = await handler.handle(start, send=None)
    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.CONSENT_REQUIRED


async def test_start_with_valid_token_succeeds(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    broker = ConsentBroker()
    handler = RecordingHandler(
        cfg,
        consent_broker=broker,
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            interpretation_route=FLOOR_ROUTE,
            mode="copilot",
            channels=["floor"],
            consent_token=_floor_consent(broker),
        ),
    )
    resp = await handler.handle(start, send=None)
    assert isinstance(resp, RecordingStartedMessage)
    assert resp.payload.recording_id.startswith("rec_")


async def test_recording_disabled_returns_capability_not_granted(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, enable_recording=False)
    handler = RecordingHandler(cfg, consent_broker=ConsentBroker(), recording_cipher=_TEST_CIPHER)
    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(mode="copilot", channels=["floor"], consent_token="x"),
    )
    resp = await handler.handle(start, send=None)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.CAPABILITY_NOT_GRANTED


# ── Audit is content-free (§9) ────────────────────────────────────────────────


async def test_audit_entry_is_content_free(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    audit = _audit(tmp_path)
    broker = ConsentBroker()
    secret_value = "super-secret-form-value-12345"
    steps = [
        _step(1, raw=secret_value, kind="dom", label="First Name"),
        _step(2, action="submit", raw=None, kind="none"),
    ]
    handler = RecordingHandler(
        cfg,
        audit=audit,
        consent_broker=broker,
        capture_factory=_capture_factory(steps),
        recording_cipher=_TEST_CIPHER,
    )

    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            mode="demonstration",
            interpretation_route=FLOOR_ROUTE,
            channels=["floor"],
            consent_token=_floor_consent(broker, mode="demonstration"),
        ),
    )
    started = await handler.handle(start, send=None)
    assert isinstance(started, RecordingStartedMessage)
    rid = started.payload.recording_id

    stop = StopRecordingMessage(
        id=new_id(), ts=now_ts(), payload=StopRecordingPayload(recording_id=rid)
    )
    await handler.handle(stop, send=None)

    fetch = RecordingFetchMessage(
        id=new_id(), ts=now_ts(), payload=RecordingFetchPayload(recording_id=rid)
    )
    await handler.handle(fetch, send=None)

    # The audit log must contain lifecycle records but NEVER step content.
    log_text = (tmp_path / "audit.log").read_text(encoding="utf-8")
    assert "RECORDING_STARTED" in log_text
    assert "RECORDING_STOPPED" in log_text
    assert "RECORDING_FETCHED" in log_text
    # No step content of any kind.
    assert secret_value not in log_text
    assert "First Name" not in log_text
    assert "stub_1" not in log_text  # no screenshot refs
    assert "Google Chrome" not in log_text  # no active_app
    # But content-free metadata IS present.
    records = [json.loads(line) for line in log_text.splitlines() if line.strip()]
    rec_records = [r for r in records if r["op"].startswith("RECORDING_")]
    assert rec_records, "expected recording lifecycle records"
    for r in rec_records:
        d = r["details"]
        assert d["recording_id"] == rid
        assert d["channels"] == ["floor"]
        assert d["interpretation_route"] in (
            "extract_then_cloud",
            "local_vlm",
            "screenshots_to_cloud",
        )
        assert "step_count" in d
        # Defensive: the details dict carries no step-shaped keys.
        assert "steps" not in d
        assert "action" not in d
        assert "value" not in d


# ── Demonstration buffers (no stream) vs co-pilot streams ─────────────────────


async def test_demonstration_mode_does_not_stream(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    broker = ConsentBroker()
    steps = [_step(1, kind="dom"), _step(2, kind="dom")]
    handler = RecordingHandler(
        cfg,
        consent_broker=broker,
        capture_factory=_capture_factory(steps),
        recording_cipher=_TEST_CIPHER,
    )

    sent: list[object] = []

    async def _send(frame: object) -> None:
        sent.append(frame)

    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            mode="demonstration",
            interpretation_route=FLOOR_ROUTE,
            channels=["floor"],
            consent_token=_floor_consent(broker, mode="demonstration"),
        ),
    )
    started = await handler.handle(start, send=_send)
    rid = started.payload.recording_id  # type: ignore[union-attr]

    stop = StopRecordingMessage(
        id=new_id(), ts=now_ts(), payload=StopRecordingPayload(recording_id=rid)
    )
    summary = await handler.handle(stop, send=None)

    # Demonstration: NO RECORDING_STEP frames streamed.
    assert sent == []
    # But the steps were buffered and are fetchable.
    assert isinstance(summary, RecordingSummaryMessage)
    assert summary.payload.step_count == 2

    fetch = RecordingFetchMessage(
        id=new_id(), ts=now_ts(), payload=RecordingFetchPayload(recording_id=rid)
    )
    data = await handler.handle(fetch, send=None)
    assert isinstance(data, RecordingDataMessage)
    assert len(data.payload.recording.steps) == 2


async def test_copilot_mode_streams_recording_steps(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    broker = ConsentBroker()
    steps = [_step(1, kind="dom"), _step(2, kind="ax"), _step(3, kind="none")]
    handler = RecordingHandler(
        cfg,
        consent_broker=broker,
        capture_factory=_capture_factory(steps),
        recording_cipher=_TEST_CIPHER,
    )

    sent: list[RecordingStepMessage] = []

    async def _send(frame: object) -> None:
        if isinstance(frame, RecordingStepMessage):
            sent.append(frame)

    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            interpretation_route=FLOOR_ROUTE,
            mode="copilot",
            channels=["floor"],
            consent_token=_floor_consent(broker),
        ),
    )
    started = await handler.handle(start, send=_send)
    rid = started.payload.recording_id  # type: ignore[union-attr]

    stop = StopRecordingMessage(
        id=new_id(), ts=now_ts(), payload=StopRecordingPayload(recording_id=rid)
    )
    summary = await handler.handle(stop, send=None)

    # Co-pilot: one RECORDING_STEP per captured step.
    assert len(sent) == 3
    assert [f.payload.step.seq for f in sent] == [1, 2, 3]
    assert all(f.payload.recording_id == rid for f in sent)
    assert isinstance(summary, RecordingSummaryMessage)
    assert summary.payload.step_count == 3


async def test_handler_review_changes_later_fetch(tmp_path: Path) -> None:
    broker = ConsentBroker()
    handler = RecordingHandler(
        _make_config(tmp_path),
        consent_broker=broker,
        capture_factory=_capture_factory([_step(1), _step(2), _step(3)]),
        recording_cipher=_TEST_CIPHER,
    )
    started = await handler.handle(
        StartRecordingMessage(
            id=new_id(),
            ts=now_ts(),
            payload=StartRecordingPayload(
                interpretation_route=FLOOR_ROUTE,
                mode="demonstration",
                channels=["floor"],
                consent_token=_floor_consent(broker, mode="demonstration"),
            ),
        ),
        send=None,
    )
    assert isinstance(started, RecordingStartedMessage)
    recording_id = started.payload.recording_id
    await handler.handle(
        StopRecordingMessage(
            id=new_id(),
            ts=now_ts(),
            payload=StopRecordingPayload(recording_id=recording_id),
        ),
        send=None,
    )

    reviewed = await handler.handle(
        ApplyRecordingReviewMessage(
            id=new_id(),
            ts=now_ts(),
            payload=ApplyRecordingReviewPayload(
                recording_id=recording_id,
                removed_step_seqs=[2],
                redacted_step_seqs=[1],
            ),
        ),
        send=None,
    )
    assert isinstance(reviewed, RecordingReviewAppliedMessage)
    assert reviewed.payload.step_count == 2

    fetched = await handler.handle(
        RecordingFetchMessage(
            id=new_id(),
            ts=now_ts(),
            payload=RecordingFetchPayload(recording_id=recording_id),
        ),
        send=None,
    )
    assert isinstance(fetched, RecordingDataMessage)
    assert [step.seq for step in fetched.payload.recording.steps] == [1, 3]
    assert fetched.payload.recording.steps[0].value.raw is None


# ── End-to-end: MockCaptureSource drives a scripted recording ─────────────────


async def test_mock_capture_source_end_to_end(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.recording_channels = ["floor", "browser"]
    broker = ConsentBroker()
    # Scripted browser-form-fill: 2 dom fills (one a password) + a submit.
    steps = [
        _step(1, action="fill", raw="John", kind="dom", label="First Name"),
        _step(2, action="fill", raw="hunter2", kind="dom", label="Password", role="textbox"),
        _step(3, action="submit", raw=None, kind="none"),
    ]
    handler = RecordingHandler(
        cfg,
        consent_broker=broker,
        capture_factory=_capture_factory(steps),
        recording_cipher=_TEST_CIPHER,
    )

    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            mode="demonstration",
            interpretation_route="extract_then_cloud",
            channels=["floor", "browser"],
            consent_token=broker.issue_consent_token(
                mode="demonstration", interpretation_route="extract_then_cloud"
            ),
        ),
    )
    started = await handler.handle(start, send=None)
    assert isinstance(started, RecordingStartedMessage)
    rid = started.payload.recording_id

    stop = StopRecordingMessage(
        id=new_id(), ts=now_ts(), payload=StopRecordingPayload(recording_id=rid)
    )
    summary = await handler.handle(stop, send=None)
    assert isinstance(summary, RecordingSummaryMessage)
    assert summary.payload.step_count == 3
    assert summary.payload.enrichment_coverage == EnrichmentCoverage(dom=2, ax=0, none=1)

    fetch = RecordingFetchMessage(
        id=new_id(), ts=now_ts(), payload=RecordingFetchPayload(recording_id=rid)
    )
    data = await handler.handle(fetch, send=None)
    assert isinstance(data, RecordingDataMessage)
    rec = data.payload.recording
    # Round-trips on the wire.
    again = parse_message(dump_message(data))
    assert isinstance(again, RecordingDataMessage)
    # Floor fields intact; password redacted; first name kept.
    assert rec.steps[0].value.raw == "John"
    assert rec.steps[1].value.raw is None and rec.steps[1].redacted is True
    assert rec.steps[2].action == "submit"
    assert all(s.screenshot_ref for s in rec.steps)


async def test_finalized_recordings_are_retained_by_id_with_bounded_eviction(
    tmp_path: Path,
) -> None:
    cfg = _make_config(tmp_path)
    cfg.recording_retention_limit = 2
    broker = ConsentBroker()
    handler = RecordingHandler(
        cfg,
        consent_broker=broker,
        capture_factory=_capture_factory([_step(1)]),
        recording_cipher=_TEST_CIPHER,
    )
    recording_ids: list[str] = []

    for _ in range(3):
        started = await handler.handle(
            StartRecordingMessage(
                id=new_id(),
                ts=now_ts(),
                payload=StartRecordingPayload(
                    interpretation_route=FLOOR_ROUTE,
                    mode="demonstration",
                    channels=["floor"],
                    consent_token=_floor_consent(broker, mode="demonstration"),
                ),
            ),
            send=None,
        )
        assert isinstance(started, RecordingStartedMessage)
        recording_ids.append(started.payload.recording_id)
        await handler.handle(
            StopRecordingMessage(
                id=new_id(),
                ts=now_ts(),
                payload=StopRecordingPayload(recording_id=started.payload.recording_id),
            ),
            send=None,
        )

    responses = []
    for recording_id in recording_ids:
        responses.append(
            await handler.handle(
                RecordingFetchMessage(
                    id=new_id(),
                    ts=now_ts(),
                    payload=RecordingFetchPayload(recording_id=recording_id),
                ),
                send=None,
            )
        )

    assert isinstance(responses[0], ErrorMessage)
    assert responses[0].payload.code == ErrorCode.RECORDING_NOT_FOUND
    assert all(isinstance(response, RecordingDataMessage) for response in responses[1:])


async def test_stop_unknown_recording_returns_not_found(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    handler = RecordingHandler(
        cfg,
        consent_broker=ConsentBroker(),
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    stop = StopRecordingMessage(
        id=new_id(), ts=now_ts(), payload=StopRecordingPayload(recording_id="rec_nope")
    )
    resp = await handler.handle(stop, send=None)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.RECORDING_NOT_FOUND


async def test_fetch_unknown_recording_returns_not_found(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    handler = RecordingHandler(
        cfg,
        consent_broker=ConsentBroker(),
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    fetch = RecordingFetchMessage(
        id=new_id(), ts=now_ts(), payload=RecordingFetchPayload(recording_id="rec_nope")
    )
    resp = await handler.handle(fetch, send=None)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.RECORDING_NOT_FOUND


async def test_start_refuses_a_route_other_than_the_consented_one(tmp_path: Path) -> None:
    """A token approves one interpretation route. A machine that can't honor
    it must not fall back to another under that approval: the fallback can be
    screenshots_to_cloud, which sends frames off the machine."""
    broker = ConsentBroker()
    handler = RecordingHandler(
        _make_config(tmp_path),
        consent_broker=broker,
        capture_factory=_capture_factory([]),
        recording_cipher=_TEST_CIPHER,
    )
    token = broker.issue_consent_token(mode="copilot", interpretation_route="local_vlm")
    start = StartRecordingMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StartRecordingPayload(
            mode="copilot",
            interpretation_route="local_vlm",
            channels=["floor"],
            consent_token=token,
        ),
    )

    response = await handler.handle(start, send=None)

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.CONSENT_REQUIRED

    # The refusal leaves the single-use token unspent: once the route can be
    # honored, the same token starts the recording.
    handler.set_local_llm_models(["llava:13b"])
    retry = await handler.handle(start.model_copy(update={"id": new_id()}), send=None)
    assert isinstance(retry, RecordingStartedMessage)


async def test_an_accessibility_result_with_no_fields_leaves_the_step_unenriched() -> None:
    """Any non-empty dict used to tag the step `ax`, so coverage counted
    steps the accessibility tree said nothing about."""
    from autogpt_local_executor.recording import A11yEnricher
    from autogpt_local_executor.recording.capture import MockCaptureSource

    def resolver(**_where):
        return {"role": None, "label": "", "unrelated": "value"}

    enricher = A11yEnricher(floor=MockCaptureSource([_step(1)]), resolver=resolver)

    [step] = [step async for step in enricher.steps()]

    assert step.enrichment.kind == "none"


async def test_an_accessibility_result_with_fields_tags_the_step_ax() -> None:
    """The test above alone passes for an enricher that never attaches
    anything."""
    from autogpt_local_executor.recording import A11yEnricher
    from autogpt_local_executor.recording.capture import MockCaptureSource

    def resolver(**_where):
        return {"role": "button", "label": "Save", "ax_path": None}

    enricher = A11yEnricher(floor=MockCaptureSource([_step(1)]), resolver=resolver)

    [step] = [step async for step in enricher.steps()]

    assert (step.enrichment.kind, step.enrichment.role, step.enrichment.label) == (
        "ax",
        "button",
        "Save",
    )


def _fake_quartz(monkeypatch, *, location_fails: bool = False):
    import sys
    from types import SimpleNamespace

    def location(_event):
        if location_fails:
            raise ValueError("the event has no location")
        return SimpleNamespace(x=1, y=2)

    monkeypatch.setitem(
        sys.modules,
        "Quartz",
        SimpleNamespace(
            CGEventGetLocation=location,
            CGEventGetIntegerValueField=lambda *_args: 0,
            kCGKeyboardEventKeycode=9,
        ),
    )


async def test_an_event_the_capture_could_not_read_counts_as_dropped(monkeypatch) -> None:
    """A marshal failure was only logged, so the recording went on without
    the action, as complete."""
    from autogpt_local_executor.recording import macos_capture

    _fake_quartz(monkeypatch, location_fails=True)
    source = macos_capture.MacInputCaptureSource()

    source._on_event(None, 1, object(), None)

    assert source.dropped_events == 1


async def test_the_macos_capture_reports_events_dropped_or_queued_at_the_stop(
    monkeypatch,
) -> None:
    """steps() checked for dropped events only at the top of each pass while
    running, so drops after the last check went unreported, and events still
    queued when the stop came were never yielded."""
    from autogpt_local_executor.recording import macos_capture
    from autogpt_local_executor.recording.session import RecordingError

    _fake_quartz(monkeypatch)
    source = macos_capture.MacInputCaptureSource(max_queued_events=2)
    source._tap = object()
    monkeypatch.setattr(source, "stop", lambda: None)
    source._stop.set()
    for _ in range(2):
        source._on_event(None, 1, object(), None)

    steps = [step async for step in source.steps()]
    assert len(steps) == 2

    source._stop.clear()
    source._stop.set()
    source._on_event(None, 1, object(), None)
    source._on_event(None, 1, object(), None)
    source._on_event(None, 1, object(), None)
    with pytest.raises(RecordingError, match="dropped 1"):
        async for _step_ in source.steps():
            pass


async def test_the_macos_capture_fails_a_recording_that_dropped_events(monkeypatch) -> None:
    """An unbounded queue grew without limit behind a stalled consumer; a
    bounded one must not then lose events silently."""
    import sys
    from types import SimpleNamespace

    from autogpt_local_executor.recording import macos_capture
    from autogpt_local_executor.recording.session import RecordingError

    quartz = SimpleNamespace(
        CGEventGetLocation=lambda _event: SimpleNamespace(x=1, y=2),
        CGEventGetIntegerValueField=lambda *_args: 0,
        kCGKeyboardEventKeycode=9,
    )
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    source = macos_capture.MacInputCaptureSource(max_queued_events=2)
    for _ in range(5):
        source._on_event(None, 1, object(), None)
    source._tap = object()
    monkeypatch.setattr(source, "stop", lambda: None)

    with pytest.raises(RecordingError, match="dropped 3"):
        async for _step_ in source.steps():
            pass
