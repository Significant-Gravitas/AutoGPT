"""Word-error-rate harness for the onboarding brain-dump transcription.

Runs the real pipeline (``transcription.transcribe``) over a directory of
``<name>.<audio ext>`` + ``<name>.txt`` pairs and reports WER per file and
pooled over the corpus. Files big enough to take the chunked path are
transcribed segment by segment here so the stitch boundaries can be shown
— seam errors are invisible in an aggregate number.

See ``EVAL.md`` for the corpus requirements and the 5% release gate.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from backend.api.features.onboarding_dump import transcription
from backend.api.features.onboarding_dump.brain_dump_wer import (
    WordErrors,
    compute_word_errors,
    normalize_words,
)

WER_RELEASE_GATE = 0.05
SEAM_CONTEXT_WORDS = 10
AUDIO_EXTENSIONS = (".webm", ".mp4", ".m4a", ".mp3", ".wav", ".ogg")


class Seam(BaseModel):
    """One stitch boundary, between segment ``index`` and ``index + 1``."""

    index: int
    left_tail: str
    right_head: str
    dropped_words: int


class PipelineRun(BaseModel):
    """What one trip through the pipeline produced, before scoring."""

    transcript: str
    language: str | None = None
    model: str
    segment_count: int
    seams: list[Seam] = []


class FileResult(BaseModel):
    name: str
    audio_path: str
    errors: WordErrors
    latency_secs: float
    chunked: bool
    hypothesis_words: int = 0
    language: str | None = None
    model: str = ""
    """Which STT model actually produced this transcript.

    The pipeline falls back to ``FALLBACK_MODEL`` silently, so without
    this a run meant to gate ``gpt-4o-transcribe`` can pass on
    ``whisper-1``'s numbers with nothing in the report saying so.
    """
    segment_count: int = 0
    seams: list[Seam] = []
    error: str | None = None


class EvalReport(BaseModel):
    files: list[FileResult]
    pooled: WordErrors
    mean_wer: float
    median_wer: float
    total_secs: float
    gate: float
    passed: bool
    audio_without_reference: list[str]
    reference_without_audio: list[str]


class DumpPair(BaseModel):
    name: str
    audio_path: Path
    reference_path: Path


class PairDiscovery(BaseModel):
    pairs: list[DumpPair]
    audio_without_reference: list[str]
    reference_without_audio: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure WER of the brain-dump transcription pipeline."
    )
    parser.add_argument(
        "--dir", required=True, type=Path, help="directory of audio + .txt pairs"
    )
    parser.add_argument("--model", default=None, help="override the primary STT model")
    parser.add_argument(
        "--duration-secs",
        dest="duration_secs",
        default=None,
        type=float,
        help=(
            "fallback duration for chunked files whose container header "
            "reports none (an upper bound is safe)"
        ),
    )
    parser.add_argument(
        "--json", dest="json_path", default=None, type=Path, help="write results here"
    )
    args = parser.parse_args()

    if args.model:
        # The pipeline reads this module global per request, so assigning
        # it overrides the model for the run without touching the env.
        transcription.PRIMARY_MODEL = args.model

    report = asyncio.run(run_eval(args.dir, args.duration_secs))
    print(render_report(report))
    if args.json_path:
        args.json_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
    sys.exit(0 if report.passed else 1)


async def run_eval(directory: Path, duration_secs: float | None = None) -> EvalReport:
    discovery = discover_pairs(directory)
    started = time.monotonic()
    results = [await evaluate_pair(pair, duration_secs) for pair in discovery.pairs]
    scored = [result for result in results if result.error is None]
    pooled = WordErrors(
        substitutions=sum(r.errors.substitutions for r in scored),
        insertions=sum(r.errors.insertions for r in scored),
        deletions=sum(r.errors.deletions for r in scored),
        reference_words=sum(r.errors.reference_words for r in scored),
    )
    rates = [result.errors.wer for result in scored]
    return EvalReport(
        files=results,
        pooled=pooled,
        mean_wer=statistics.fmean(rates) if rates else 0.0,
        median_wer=statistics.median(rates) if rates else 0.0,
        total_secs=time.monotonic() - started,
        gate=WER_RELEASE_GATE,
        passed=bool(scored)
        and len(scored) == len(results)
        and pooled.wer < WER_RELEASE_GATE,
        audio_without_reference=discovery.audio_without_reference,
        reference_without_audio=discovery.reference_without_audio,
    )


async def evaluate_pair(
    pair: DumpPair, duration_secs: float | None = None
) -> FileResult:
    audio = pair.audio_path.read_bytes()
    reference = pair.reference_path.read_text(encoding="utf-8")
    # The harness has no duration metadata, so it takes the split decision
    # on the same byte cap `transcribe(audio, name, duration_secs=None)` uses.
    chunked = len(audio) > transcription.SINGLE_REQUEST_MAX_BYTES
    started = time.monotonic()
    try:
        run = await _run_pipeline(audio, pair.audio_path.name, chunked, duration_secs)
    except Exception as e:
        return FileResult(
            name=pair.name,
            audio_path=pair.audio_path.name,
            errors=_empty_errors(reference),
            latency_secs=time.monotonic() - started,
            chunked=chunked,
            error=str(e),
        )
    return FileResult(
        name=pair.name,
        audio_path=pair.audio_path.name,
        errors=compute_word_errors(reference, run.transcript),
        latency_secs=time.monotonic() - started,
        chunked=chunked,
        hypothesis_words=len(normalize_words(run.transcript)),
        language=run.language,
        model=run.model,
        segment_count=run.segment_count,
        seams=run.seams,
    )


def _empty_errors(reference: str) -> WordErrors:
    return WordErrors(
        substitutions=0,
        insertions=0,
        deletions=0,
        reference_words=len(normalize_words(reference)),
    )


async def _run_pipeline(
    audio: bytes,
    filename: str,
    chunked: bool,
    duration_secs: float | None = None,
) -> PipelineRun:
    if not chunked:
        result = await transcription.transcribe(audio, filename)
        return PipelineRun(
            transcript=result.text,
            language=result.language,
            model=result.model,
            segment_count=1,
        )

    segments = await transcription.split_audio(audio, filename, duration_secs)
    # ``split_audio`` re-encodes to ogg/opus, and the STT client infers the
    # format from the filename — naming a segment after the source (e.g.
    # ``0-dump01.webm``) gets the whole corpus rejected.
    results = [
        await transcription.transcribe(
            segment, f"segment-{index}{transcription.SEGMENT_SUFFIX}"
        )
        for index, segment in enumerate(segments)
    ]
    parts = [result.text for result in results]
    return PipelineRun(
        transcript=transcription.stitch_transcripts(parts),
        model=",".join(dict.fromkeys(result.model for result in results)),
        segment_count=len(parts),
        seams=describe_seams(parts),
    )


def describe_seams(parts: list[str]) -> list[Seam]:
    """Report what the overlap-dedup did at each stitch boundary."""
    seams: list[Seam] = []
    for index in range(len(parts) - 1):
        before = len(transcription.stitch_transcripts(parts[: index + 1]).split())
        added = len(parts[index + 1].split())
        after = len(transcription.stitch_transcripts(parts[: index + 2]).split())
        seams.append(
            Seam(
                index=index,
                left_tail=" ".join(parts[index].split()[-SEAM_CONTEXT_WORDS:]),
                right_head=" ".join(parts[index + 1].split()[:SEAM_CONTEXT_WORDS]),
                dropped_words=before + added - after,
            )
        )
    return seams


def discover_pairs(directory: Path) -> PairDiscovery:
    entries = sorted(directory.iterdir())
    audio_by_name: dict[str, Path] = {}
    for path in entries:
        if path.suffix.lower() in AUDIO_EXTENSIONS and path.stem not in audio_by_name:
            audio_by_name[path.stem] = path
    references = {path.stem: path for path in entries if path.suffix.lower() == ".txt"}
    return PairDiscovery(
        pairs=[
            DumpPair(name=name, audio_path=path, reference_path=references[name])
            for name, path in sorted(audio_by_name.items())
            if name in references
        ],
        audio_without_reference=sorted(set(audio_by_name) - set(references)),
        reference_without_audio=sorted(set(references) - set(audio_by_name)),
    )


def render_report(report: EvalReport) -> str:
    lines = [
        "",
        f"Brain-dump transcription WER — {len(report.files)} file(s)",
        "",
        f"{'file':<20}{'WER':>9}{'sub':>6}{'ins':>6}{'del':>6}"
        f"{'ref':>8}{'secs':>8}  {'segments':<10}model",
    ]
    lines.extend(_render_row(result) for result in report.files)
    lines.extend(_render_seams(report))
    lines.extend(_render_mismatches(report))
    lines.extend(
        [
            "",
            f"pooled WER   {report.pooled.wer:>8.2%}  "
            f"({report.pooled.reference_words} reference words)",
            f"mean WER     {report.mean_wer:>8.2%}",
            f"median WER   {report.median_wer:>8.2%}",
            f"wall clock   {report.total_secs:>8.1f}s",
            "",
            f"Release gate: aggregate WER must stay under {report.gate:.0%} — "
            + ("PASS (exit 0)" if report.passed else "FAIL (exit 1)"),
        ]
    )
    return "\n".join(lines)


def _render_row(result: FileResult) -> str:
    if result.error:
        return f"{result.name:<20}{'ERROR':>9}  {result.error[:60]}"
    return (
        f"{result.name:<20}{result.errors.wer:>9.2%}"
        f"{result.errors.substitutions:>6}{result.errors.insertions:>6}"
        f"{result.errors.deletions:>6}{result.errors.reference_words:>8}"
        f"{result.latency_secs:>8.1f}  {result.segment_count:<10}{result.model}"
    )


def _render_seams(report: EvalReport) -> list[str]:
    chunked = [result for result in report.files if result.seams]
    if not chunked:
        return []
    lines = ["", "Stitch boundaries (chunked files):"]
    for result in chunked:
        lines.append(f"  {result.name} — {result.segment_count} segments")
        for seam in result.seams:
            lines.append(
                f"    seam {seam.index}/{seam.index + 1}: "
                f"dedup dropped {seam.dropped_words} word(s)"
            )
            lines.append(f"      ...{seam.left_tail}")
            lines.append(f"      {seam.right_head}...")
    return lines


def _render_mismatches(report: EvalReport) -> list[str]:
    lines: list[str] = []
    if report.audio_without_reference:
        joined = ", ".join(report.audio_without_reference)
        lines.append(f"\nAudio with no .txt reference: {joined}")
    if report.reference_without_audio:
        joined = ", ".join(report.reference_without_audio)
        lines.append(f"\n.txt reference with no audio: {joined}")
    return lines


if __name__ == "__main__":
    main()
