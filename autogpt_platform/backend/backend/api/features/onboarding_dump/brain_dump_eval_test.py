from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from backend.api.features.onboarding_dump import brain_dump_eval, transcription
from backend.api.features.onboarding_dump.brain_dump_wer import (
    compute_word_errors,
    normalize_words,
)
from backend.api.features.onboarding_dump.transcription import TranscriptionResult


def test_identical_text_has_no_errors():
    errors = compute_word_errors("the quick brown fox", "the quick brown fox")

    assert errors.wer == 0.0
    assert (errors.substitutions, errors.insertions, errors.deletions) == (0, 0, 0)
    assert errors.reference_words == 4


def test_substitution_is_counted_once():
    errors = compute_word_errors("the quick brown fox", "the quick green fox")

    assert (errors.substitutions, errors.insertions, errors.deletions) == (1, 0, 0)
    assert errors.wer == pytest.approx(0.25)


def test_insertion_is_counted_once():
    errors = compute_word_errors("the quick brown fox", "the very quick brown fox")

    assert (errors.substitutions, errors.insertions, errors.deletions) == (0, 1, 0)
    assert errors.wer == pytest.approx(0.25)


def test_deletion_is_counted_once():
    errors = compute_word_errors("the quick brown fox", "the quick fox")

    assert (errors.substitutions, errors.insertions, errors.deletions) == (0, 0, 1)
    assert errors.wer == pytest.approx(0.25)


def test_mixed_operations_are_reported_separately():
    errors = compute_word_errors("a b c d", "a x c d e")

    assert (errors.substitutions, errors.insertions, errors.deletions) == (1, 1, 0)
    assert errors.wer == pytest.approx(0.5)


def test_empty_hypothesis_deletes_every_reference_word():
    errors = compute_word_errors("one two three", "")

    assert errors.deletions == 3
    assert errors.wer == pytest.approx(1.0)


def test_empty_reference_has_zero_wer():
    errors = compute_word_errors("", "hallucinated words here")

    assert errors.insertions == 3
    assert errors.wer == 0.0


def test_normalisation_ignores_case_punctuation_and_whitespace():
    reference = "So, I built AutoGPT — it's great!"
    hypothesis = "so i built autogpt   its great"

    assert normalize_words(reference) == normalize_words(hypothesis)
    assert compute_word_errors(reference, hypothesis).wer == 0.0


def test_normalisation_splits_on_hyphens_and_slashes():
    assert normalize_words("well-known and/or 24_7") == [
        "well",
        "known",
        "and",
        "or",
        "24",
        "7",
    ]


def test_normalisation_handles_non_english_punctuation():
    assert normalize_words("¿Qué tal? Muy bien.") == ["qué", "tal", "muy", "bien"]


def _result(text: str, language: str | None = None) -> TranscriptionResult:
    return TranscriptionResult(
        text=text, language=language, model=transcription.PRIMARY_MODEL
    )


def _write_pair(directory: Path, name: str, extension: str) -> None:
    (directory / f"{name}{extension}").write_bytes(b"audio-bytes")
    (directory / f"{name}.txt").write_text("hello world", encoding="utf-8")


def test_discover_pairs_matches_by_basename(tmp_path: Path):
    _write_pair(tmp_path, "dump01", ".webm")
    _write_pair(tmp_path, "dump02", ".mp3")

    discovery = brain_dump_eval.discover_pairs(tmp_path)

    assert [pair.name for pair in discovery.pairs] == ["dump01", "dump02"]
    assert discovery.pairs[0].audio_path == tmp_path / "dump01.webm"
    assert discovery.pairs[0].reference_path == tmp_path / "dump01.txt"


def test_discover_pairs_accepts_every_supported_extension(tmp_path: Path):
    for index, extension in enumerate(brain_dump_eval.AUDIO_EXTENSIONS):
        _write_pair(tmp_path, f"dump{index}", extension)

    discovery = brain_dump_eval.discover_pairs(tmp_path)

    assert len(discovery.pairs) == len(brain_dump_eval.AUDIO_EXTENSIONS)


def test_discover_pairs_reports_unmatched_files(tmp_path: Path):
    _write_pair(tmp_path, "dump01", ".webm")
    (tmp_path / "lonely_audio.wav").write_bytes(b"audio-bytes")
    (tmp_path / "lonely_reference.txt").write_text("text", encoding="utf-8")
    (tmp_path / "notes.pdf").write_bytes(b"ignored")

    discovery = brain_dump_eval.discover_pairs(tmp_path)

    assert [pair.name for pair in discovery.pairs] == ["dump01"]
    assert discovery.audio_without_reference == ["lonely_audio"]
    assert discovery.reference_without_audio == ["lonely_reference"]


async def test_run_eval_scores_every_pair(tmp_path: Path, monkeypatch):
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text("the quick brown fox", encoding="utf-8")
    (tmp_path / "dump02.mp3").write_bytes(b"audio")
    (tmp_path / "dump02.txt").write_text("hello there world", encoding="utf-8")
    transcribe = AsyncMock(
        side_effect=[
            _result("The quick brown fox!", "en"),
            _result("hello world", "en"),
        ]
    )
    monkeypatch.setattr(transcription, "transcribe", transcribe)

    report = await brain_dump_eval.run_eval(tmp_path)

    assert transcribe.await_count == 2
    assert [result.errors.wer for result in report.files] == [0.0, pytest.approx(1 / 3)]
    assert report.pooled.deletions == 1
    assert report.pooled.reference_words == 7
    assert report.passed is False
    assert "FAIL (exit 1)" in brain_dump_eval.render_report(report)


async def test_run_eval_passes_when_under_the_gate(tmp_path: Path, monkeypatch):
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text("the quick brown fox", encoding="utf-8")
    monkeypatch.setattr(
        transcription,
        "transcribe",
        AsyncMock(return_value=_result("the quick brown fox")),
    )

    report = await brain_dump_eval.run_eval(tmp_path)

    assert report.passed is True
    assert report.gate == brain_dump_eval.WER_RELEASE_GATE
    assert "PASS (exit 0)" in brain_dump_eval.render_report(report)


async def test_run_eval_reports_which_model_produced_the_transcript(
    tmp_path: Path, monkeypatch
):
    """A silent fallback must not be able to pass as the gated model."""
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text("the quick brown fox", encoding="utf-8")
    monkeypatch.setattr(
        transcription,
        "transcribe",
        AsyncMock(
            return_value=TranscriptionResult(
                text="the quick brown fox", model=transcription.FALLBACK_MODEL
            )
        ),
    )

    report = await brain_dump_eval.run_eval(tmp_path)

    assert report.files[0].model == transcription.FALLBACK_MODEL
    assert transcription.FALLBACK_MODEL in brain_dump_eval.render_report(report)


async def test_run_eval_records_pipeline_failures(tmp_path: Path, monkeypatch):
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text("some words here", encoding="utf-8")
    monkeypatch.setattr(
        transcription,
        "transcribe",
        AsyncMock(side_effect=transcription.TranscriptionFailedError("boom")),
    )

    report = await brain_dump_eval.run_eval(tmp_path)

    assert report.files[0].error == "boom"
    assert report.pooled.reference_words == 0
    assert report.passed is False


async def test_run_eval_reports_seams_for_chunked_files(tmp_path: Path, monkeypatch):
    (tmp_path / "dump01.webm").write_bytes(b"audio")
    (tmp_path / "dump01.txt").write_text("one two three four", encoding="utf-8")
    monkeypatch.setattr(transcription, "SINGLE_REQUEST_MAX_BYTES", 1)
    split = AsyncMock(return_value=[b"a", b"b"])
    monkeypatch.setattr(transcription, "split_audio", split)
    transcribe = AsyncMock(
        side_effect=[_result("one two three"), _result("two three four")]
    )
    monkeypatch.setattr(transcription, "transcribe", transcribe)

    report = await brain_dump_eval.run_eval(tmp_path, duration_secs=1800)
    result = report.files[0]

    assert result.chunked is True
    assert result.segment_count == 2
    # The split re-encodes to ogg/opus and the STT client infers the format
    # from the filename, so a segment must not keep the source extension.
    assert [call.args[1] for call in transcribe.await_args_list] == [
        "segment-0.ogg",
        "segment-1.ogg",
    ]
    # A `.webm` with no container duration cannot be split without a hint.
    assert split.await_args.args[2] == 1800
    assert result.errors.wer == 0.0
    assert result.seams == [
        brain_dump_eval.Seam(
            index=0,
            left_tail="one two three",
            right_head="two three four",
            dropped_words=2,
        )
    ]
    assert "seam 0/1: dedup dropped 2 word(s)" in brain_dump_eval.render_report(report)


def test_describe_seams_truncates_context_to_ten_words():
    left = " ".join(f"l{index}" for index in range(30))
    right = " ".join(f"r{index}" for index in range(30))

    seams = brain_dump_eval.describe_seams([left, right])

    assert len(seams[0].left_tail.split()) == brain_dump_eval.SEAM_CONTEXT_WORDS
    assert len(seams[0].right_head.split()) == brain_dump_eval.SEAM_CONTEXT_WORDS
    assert seams[0].left_tail.startswith("l20")
    assert seams[0].right_head.endswith("r9")
    assert seams[0].dropped_words == 0
