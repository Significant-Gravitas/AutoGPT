"""Tests for the quality-gate half of the brain-dump eval.

The contract: a corpus with a ``gate_manifest.json`` gets every transcript
gated with the real classifier interface (mocked here), garbage files with
no reference are transcribed just for the gate, and one wrong verdict —
or a file the manifest forgot — fails the whole run.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from backend.api.features.onboarding_dump import (
    brain_dump_eval,
    brain_dump_gate_eval,
    quality,
    transcription,
)
from backend.api.features.onboarding_dump.transcription import TranscriptionResult

GOOD_TRANSCRIPT = "I run a bakery and want the weekly order emails handled."
GARBAGE_TRANSCRIPT = "Thank you for watching."


def _result(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, model="gpt-4o-transcribe")


def _write_manifest(directory: Path, entries: dict[str, str]) -> None:
    (directory / brain_dump_gate_eval.GATE_MANIFEST_NAME).write_text(
        json.dumps(entries), encoding="utf-8"
    )


def _gate_by_transcript(monkeypatch) -> None:
    """A stand-in gate: rejects the known garbage transcript, passes the rest."""

    async def check(transcript: str) -> str | None:
        return "insufficient_content" if transcript == GARBAGE_TRANSCRIPT else None

    monkeypatch.setattr(quality, "check_transcript_quality", check)


def test_load_manifest_returns_none_without_a_file(tmp_path: Path):
    assert brain_dump_gate_eval.load_manifest(tmp_path) is None


def test_load_manifest_rejects_unknown_expectations(tmp_path: Path):
    _write_manifest(tmp_path, {"a": "pass", "b": "reject", "c": "maybe"})
    with pytest.raises(ValidationError):
        brain_dump_gate_eval.load_manifest(tmp_path)


def test_load_manifest_accepts_valid_entries(tmp_path: Path):
    _write_manifest(tmp_path, {"a": "pass", "b": "reject"})
    assert brain_dump_gate_eval.load_manifest(tmp_path) == {"a": "pass", "b": "reject"}


async def test_run_eval_gates_scored_transcripts_and_garbage_audio(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text(GOOD_TRANSCRIPT, encoding="utf-8")
    (tmp_path / "garbage-silence.wav").write_bytes(b"audio")
    _write_manifest(tmp_path, {"dump01": "pass", "garbage-silence": "reject"})
    transcribe = AsyncMock(
        side_effect=[_result(GOOD_TRANSCRIPT), _result(GARBAGE_TRANSCRIPT)]
    )
    monkeypatch.setattr(transcription, "transcribe", transcribe)
    _gate_by_transcript(monkeypatch)

    report = await brain_dump_eval.run_eval(tmp_path)

    assert report.passed is True
    assert report.quality_gate is not None
    assert report.quality_gate.passed is True
    by_name = {result.name: result for result in report.quality_gate.results}
    assert by_name["dump01"].outcome == "pass"
    assert by_name["garbage-silence"].outcome == "reject"
    assert by_name["garbage-silence"].error_code == "insufficient_content"
    # The garbage clip is the gate's input, not a corpus mistake.
    assert report.audio_without_reference == []
    # Both files hit the pipeline: one for WER, one gate-only.
    assert transcribe.await_count == 2


async def test_run_eval_fails_when_a_good_dump_is_rejected(tmp_path: Path, monkeypatch):
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text(GARBAGE_TRANSCRIPT, encoding="utf-8")
    _write_manifest(tmp_path, {"dump01": "pass"})
    monkeypatch.setattr(
        transcription, "transcribe", AsyncMock(return_value=_result(GARBAGE_TRANSCRIPT))
    )
    _gate_by_transcript(monkeypatch)

    report = await brain_dump_eval.run_eval(tmp_path)

    # WER is 0% — only the gate verdict is wrong, and that alone fails.
    assert report.pooled.wer == 0.0
    assert report.quality_gate is not None
    assert report.quality_gate.passed is False
    assert report.passed is False
    assert "WRONG" in brain_dump_eval.render_report(report)


async def test_run_eval_fails_when_garbage_slips_through_the_gate(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "garbage-noise.wav").write_bytes(b"audio")
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text(GOOD_TRANSCRIPT, encoding="utf-8")
    _write_manifest(tmp_path, {"dump01": "pass", "garbage-noise": "reject"})
    # The STT hallucinates something the gate then (wrongly) accepts.
    transcribe = AsyncMock(
        side_effect=[_result(GOOD_TRANSCRIPT), _result(GOOD_TRANSCRIPT)]
    )
    monkeypatch.setattr(transcription, "transcribe", transcribe)
    _gate_by_transcript(monkeypatch)

    report = await brain_dump_eval.run_eval(tmp_path)

    assert report.quality_gate is not None
    assert report.quality_gate.passed is False
    assert report.passed is False


async def test_run_eval_flags_scored_files_missing_from_the_manifest(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text(GOOD_TRANSCRIPT, encoding="utf-8")
    _write_manifest(tmp_path, {"something-else": "pass"})
    monkeypatch.setattr(
        transcription, "transcribe", AsyncMock(return_value=_result(GOOD_TRANSCRIPT))
    )
    _gate_by_transcript(monkeypatch)

    report = await brain_dump_eval.run_eval(tmp_path)

    assert report.quality_gate is not None
    assert report.quality_gate.unlisted == ["dump01"]
    assert report.quality_gate.passed is False
    assert report.passed is False


async def test_run_eval_without_a_manifest_skips_the_gate(tmp_path: Path, monkeypatch):
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text(GOOD_TRANSCRIPT, encoding="utf-8")
    monkeypatch.setattr(
        transcription, "transcribe", AsyncMock(return_value=_result(GOOD_TRANSCRIPT))
    )
    gate = AsyncMock()
    monkeypatch.setattr(quality, "check_transcript_quality", gate)

    report = await brain_dump_eval.run_eval(tmp_path)

    assert report.quality_gate is None
    assert report.passed is True
    gate.assert_not_awaited()


async def test_gate_only_transcription_failure_counts_as_empty_transcript(
    tmp_path: Path, monkeypatch
):
    """An STT error on garbage audio still ends non-personalized in
    production, so the eval scores the gate on the empty transcript — but
    records the error, because the two findings are different."""
    audio = tmp_path / "garbage-silence.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        transcription,
        "transcribe",
        AsyncMock(
            side_effect=transcription.TranscriptionFailedError("audio too short")
        ),
    )

    async def check(transcript: str) -> str | None:
        return None if transcript else "no_usable_speech"

    monkeypatch.setattr(quality, "check_transcript_quality", check)

    result = await brain_dump_gate_eval.evaluate_gate_only(audio, "reject")

    assert result.outcome == "reject"
    assert result.correct is True
    assert result.transcription_error == "audio too short"


@pytest.mark.parametrize(
    "expected,outcome_correct", [("pass", True), ("reject", False)]
)
async def test_evaluate_transcript_compares_outcome_to_expectation(
    monkeypatch, expected: str, outcome_correct: bool
):
    _gate_by_transcript(monkeypatch)

    result = await brain_dump_gate_eval.evaluate_transcript(
        "dump01", expected, GOOD_TRANSCRIPT
    )

    assert result.outcome == "pass"
    assert result.correct is outcome_correct


# --- shipped corpus consistency (pure filesystem, no credentials) ---------

EVAL_DATA_DIR = Path(__file__).parent / "eval_data"


def test_shipped_corpus_matches_its_manifest():
    """Corpus drift (renamed/removed audio, a clip the manifest forgot,
    a reference file on a garbage clip) must fail CI without needing an
    LLM or STT credential."""
    manifest = brain_dump_gate_eval.load_manifest(EVAL_DATA_DIR)
    assert manifest, "shipped corpus must carry a gate manifest"

    audio = {path.stem for path in EVAL_DATA_DIR.glob("*.ogg")}
    references = {path.stem for path in EVAL_DATA_DIR.glob("*.txt")}

    assert set(manifest) == audio, "manifest keys and audio files must match 1:1"
    assert references <= audio, "reference files must match an audio file"
    for stem in sorted(audio):
        if stem.startswith("gate-reject-"):
            assert manifest[stem] == "reject"
            assert stem not in references, f"{stem} must not carry a reference"
        else:
            assert manifest[stem] == "pass"
            assert stem in references, f"{stem} needs a .txt reference"
