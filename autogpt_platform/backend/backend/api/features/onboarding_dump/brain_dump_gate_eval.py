"""Quality-gate scoring for the brain-dump eval corpus.

The WER harness answers "did the STT hear the words?"; this answers "did
the quality gate make the right call on the result?". A corpus opts in by
shipping a ``gate_manifest.json`` next to the audio:

    {"dump01-clean": "pass", "garbage-silence": "reject", ...}

Every scored WER file's transcript is pushed through the real
``quality.check_transcript_quality``, and manifest entries with no ``.txt``
reference (silence, noise) are transcribed just to see what the STT
hallucinates — the gate must reject it, whatever it is. One wrong verdict
fails the whole eval run: a garbage dump that personalizes, or a real dump
that bounces, are both release blockers.
"""

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from backend.api.features.onboarding_dump import quality, transcription

logger = logging.getLogger(__name__)

GATE_MANIFEST_NAME = "gate_manifest.json"
TRANSCRIPT_PREVIEW_CHARS = 80

GateExpectation = Literal["pass", "reject"]


class GateFileResult(BaseModel):
    name: str
    expected: GateExpectation
    outcome: GateExpectation
    error_code: str | None = None
    correct: bool
    transcript_preview: str = ""
    transcription_error: str | None = None


class GateReport(BaseModel):
    results: list[GateFileResult]
    passed: bool
    unlisted: list[str] = []
    """Scored audio files the manifest forgot — reported so a new clip
    can't silently skip the gate check."""


def load_manifest(directory: Path) -> dict[str, GateExpectation] | None:
    path = directory / GATE_MANIFEST_NAME
    if not path.exists():
        return None
    return TypeAdapter(dict[str, GateExpectation]).validate_json(
        path.read_text(encoding="utf-8")
    )


def build_report(
    results: list[GateFileResult],
    scored_names: list[str],
    manifest: dict[str, GateExpectation],
) -> GateReport:
    unlisted = sorted(set(scored_names) - set(manifest))
    return GateReport(
        results=results,
        unlisted=unlisted,
        passed=all(result.correct for result in results) and not unlisted,
    )


async def evaluate_transcript(
    name: str, expected: GateExpectation, transcript: str
) -> GateFileResult:
    error_code = await quality.check_transcript_quality(transcript)
    outcome: GateExpectation = "pass" if error_code is None else "reject"
    return GateFileResult(
        name=name,
        expected=expected,
        outcome=outcome,
        error_code=error_code,
        correct=outcome == expected,
        transcript_preview=transcript.strip()[:TRANSCRIPT_PREVIEW_CHARS],
    )


async def evaluate_gate_only(
    audio_path: Path, expected: GateExpectation
) -> GateFileResult:
    """Transcribe a reference-less file (silence, noise) and gate the result.

    A transcription failure counts as an empty transcript rather than an
    eval error: in production that request ends in a non-personalized
    ``failed`` status, which is the same protection the gate provides —
    but it is still recorded, because "the STT errored" and "the gate
    caught a hallucination" are different findings.
    """
    transcription_error: str | None = None
    try:
        result = await transcription.transcribe(
            audio_path.read_bytes(), audio_path.name
        )
        transcript = result.text
    except (
        transcription.TranscriptionUnavailableError,
        transcription.TranscriptionFailedError,
        OSError,
    ) as e:
        transcription_error = str(e)
        transcript = ""
        logger.warning("Gate-only transcription failed for %s: %s", audio_path.name, e)
    scored = await evaluate_transcript(audio_path.stem, expected, transcript)
    return scored.model_copy(update={"transcription_error": transcription_error})


def discover_gate_only(
    directory: Path,
    manifest: dict[str, GateExpectation],
    paired_names: set[str],
    audio_extensions: tuple[str, ...],
) -> list[Path]:
    """Manifest entries that have audio but no ``.txt`` reference."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in audio_extensions
        and path.stem in manifest
        and path.stem not in paired_names
    )


def render_gate_report(report: GateReport) -> list[str]:
    rows = [
        [
            result.name,
            result.expected,
            result.outcome,
            "ok" if result.correct else "WRONG",
            _gate_detail(result),
        ]
        for result in report.results
    ]
    lines = [
        "",
        f"Quality gate — {len(report.results)} file(s)",
        *render_table(["file", "expected", "outcome", "result", "detail"], rows),
    ]
    if report.unlisted:
        lines.append(f"Not in {GATE_MANIFEST_NAME}: {', '.join(report.unlisted)}")
    lines.append(
        "Gate verdicts must all match the manifest — "
        + ("PASS" if report.passed else "FAIL")
    )
    return lines


def _gate_detail(result: GateFileResult) -> str:
    detail = result.error_code or result.transcript_preview or "(empty transcript)"
    if result.transcription_error:
        detail += f"  [stt error: {result.transcription_error[:40]}]"
    return detail


def render_table(
    headers: list[str], rows: list[list[str]], right_align: frozenset[int] = frozenset()
) -> list[str]:
    """Box-drawn table with column widths sized to the content.

    Shared by the WER and quality-gate report sections so both stay
    readable however long the corpus filenames get.
    """
    widths = [
        max(len(header), *(len(row[column]) for row in rows)) if rows else len(header)
        for column, header in enumerate(headers)
    ]

    def line(cells: list[str]) -> str:
        padded = [
            cell.rjust(widths[i]) if i in right_align else cell.ljust(widths[i])
            for i, cell in enumerate(cells)
        ]
        return "│ " + " │ ".join(padded) + " │"

    def rule(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (width + 2) for width in widths) + right

    return [
        rule("┌", "┬", "┐"),
        line(headers),
        rule("├", "┼", "┤"),
        *[line(row) for row in rows],
        rule("└", "┴", "┘"),
    ]
