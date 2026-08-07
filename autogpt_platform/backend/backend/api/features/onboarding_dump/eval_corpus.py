"""Build a brain-dump eval corpus from public-domain speech.

Downloads Mini LibriSpeech (LibriVox recordings, public domain — exact
per-utterance reference transcripts included), stitches consecutive
utterances into ~1-minute clips, and writes the audio + ``.txt`` pairs the
WER harness (``brain_dump_eval.py``) consumes.

The corpus is deliberately mixed, per the eval plan:

- clean clips — the "very good" baseline;
- clips remixed with pink/white noise at moderate (15 dB SNR) and loud
  (5 dB SNR) levels, so the report shows how far accuracy degrades with a
  fan, a street, a café;
- garbage clips (silence, pure noise) with **no** reference text — these
  exist for the quality gate, which must reject whatever the STT
  hallucinates from them. Expectations land in ``gate_manifest.json``.

Read speech is not rambling speech — treat the numbers as a floor check,
not a substitute for the real-dump corpus EVAL.md asks for.

Usage (from ``autogpt_platform/backend``):

    poetry run python -m backend.api.features.onboarding_dump.eval_corpus \
        --out /tmp/brain-dump-eval-corpus
"""

import argparse
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from pydantic import BaseModel

SOURCE_URL = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
CACHE_DIR = Path.home() / ".cache" / "brain-dump-eval"

TOTAL_CLIPS = 25
# words per clip before it is closed out — ~110 words of read speech is
# roughly 45-70 seconds, comfortably past the gate's clear-pass threshold.
MIN_CLIP_WORDS = 110
MAX_CLIPS_PER_CHAPTER = 2

# clean / moderate-noise / loud-noise split over the 25 speech clips.
MODERATE_NOISE_CLIPS = 6
LOUD_NOISE_CLIPS = 6
MODERATE_SNR_DB = 15.0
LOUD_SNR_DB = 5.0

GATE_MANIFEST_NAME = "gate_manifest.json"

# Every filename states what the file tests. The speech clips measure STT
# accuracy (WER) at a given noise level and must pass the quality gate;
# the ``gate-reject-*`` clips contain no speech at all and exist purely to
# prove the gate rejects whatever the STT hallucinates from them.
CLEAN_LABEL = "stt-wer-clean"
MODERATE_LABEL = "stt-wer-moderate-noise-15db"
LOUD_LABEL = "stt-wer-loud-noise-5db"
GARBAGE_CLIPS = (
    ("gate-reject-silence", "anullsrc=r=16000:cl=mono", 8),
    (
        "gate-reject-white-noise",
        "anoisesrc=color=white:sample_rate=16000:amplitude=0.3",
        10,
    ),
    (
        "gate-reject-pink-noise",
        "anoisesrc=color=pink:sample_rate=16000:amplitude=0.4",
        12,
    ),
    (
        "gate-reject-brown-noise",
        "anoisesrc=color=brown:sample_rate=16000:amplitude=0.5",
        10,
    ),
)


class Utterance(BaseModel):
    audio_path: Path
    text: str


class Clip(BaseModel):
    name: str
    utterances: list[Utterance]
    snr_db: float | None = None

    @property
    def reference(self) -> str:
        return " ".join(u.text for u in self.utterances)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble a mixed clean/noisy WER + quality-gate corpus."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required on PATH")

    extracted = download_and_extract(args.source_url, args.cache)
    clips = plan_clips(collect_chapters(extracted))
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    for clip in clips:
        build_clip(clip, args.out)
        manifest[clip.name] = "pass"
        print(f"  {clip.name}.ogg  ({len(clip.reference.split())} ref words)")
    for name, source, seconds in GARBAGE_CLIPS:
        build_garbage_clip(args.out / f"{name}.ogg", source, seconds)
        manifest[name] = "reject"
        print(f"  {name}.ogg  (gate must reject)")

    (args.out / GATE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\n{len(clips)} speech clips + {len(GARBAGE_CLIPS)} garbage clips "
        f"in {args.out}\nRun: poetry run brain-dump-eval --dir {args.out}"
    )


def download_and_extract(url: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / url.rsplit("/", 1)[-1]
    if not tarball.exists():
        print(f"Downloading {url} ...")
        with urllib.request.urlopen(url) as response, open(tarball, "wb") as out:
            shutil.copyfileobj(response, out)
    root = cache / "extracted"
    if not root.exists():
        print(f"Extracting {tarball.name} ...")
        with tarfile.open(tarball) as archive:
            archive.extractall(root, filter="data")
    return root


def collect_chapters(root: Path) -> list[list[Utterance]]:
    """Every chapter's utterances, in spoken order.

    LibriSpeech lays a chapter out as ``<spk>/<ch>/<spk>-<ch>-NNNN.flac``
    plus one ``<spk>-<ch>.trans.txt`` of ``ID TEXT`` lines; consecutive
    utterance ids are consecutive audiobook speech, which is what makes
    concatenating them sound like one continuous take.
    """
    chapters = []
    for trans in sorted(root.rglob("*.trans.txt")):
        utterances = []
        for line in trans.read_text(encoding="utf-8").splitlines():
            utterance_id, _, text = line.partition(" ")
            flac = trans.parent / f"{utterance_id}.flac"
            if text.strip() and flac.exists():
                utterances.append(Utterance(audio_path=flac, text=text.strip()))
        if utterances:
            chapters.append(utterances)
    return chapters


def plan_clips(chapters: list[list[Utterance]]) -> list[Clip]:
    """Fold chapters into ``TOTAL_CLIPS`` clips and assign the noise mix.

    Chapters are interleaved (at most ``MAX_CLIPS_PER_CHAPTER`` each) so
    the corpus spans many speakers rather than exhausting the first one.
    """
    groups: list[list[Utterance]] = []
    for chapter in chapters:
        taken = 0
        group: list[Utterance] = []
        for utterance in chapter:
            group.append(utterance)
            if sum(len(u.text.split()) for u in group) >= MIN_CLIP_WORDS:
                groups.append(group)
                group = []
                taken += 1
                if taken == MAX_CLIPS_PER_CHAPTER:
                    break
        if len(groups) >= TOTAL_CLIPS:
            break
    if len(groups) < TOTAL_CLIPS:
        raise SystemExit(
            f"corpus source only yielded {len(groups)} clips of {TOTAL_CLIPS}"
        )

    clips = []
    counters: dict[str, int] = {}
    for index, group in enumerate(groups[:TOTAL_CLIPS]):
        snr, label = None, CLEAN_LABEL
        if index >= TOTAL_CLIPS - LOUD_NOISE_CLIPS:
            snr, label = LOUD_SNR_DB, LOUD_LABEL
        elif index >= TOTAL_CLIPS - LOUD_NOISE_CLIPS - MODERATE_NOISE_CLIPS:
            snr, label = MODERATE_SNR_DB, MODERATE_LABEL
        counters[label] = counters.get(label, 0) + 1
        clips.append(
            Clip(name=f"{label}-{counters[label]:02d}", utterances=group, snr_db=snr)
        )
    return clips


def build_clip(clip: Clip, out_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as workdir:
        clean = Path(workdir) / "clean.wav"
        concat_utterances(clip.utterances, clean)
        source = clean
        if clip.snr_db is not None:
            source = Path(workdir) / "mixed.wav"
            mix_noise(clean, source, clip.snr_db, Path(workdir))
        encode_opus(source, out_dir / f"{clip.name}.ogg")
    (out_dir / f"{clip.name}.txt").write_text(clip.reference + "\n", encoding="utf-8")


def encode_opus(source: Path, target: Path) -> None:
    """ogg/opus at voice bitrate — what the pipeline's own splitter emits.

    Keeps a 29-clip corpus at ~4 MB instead of ~34 MB of wav, small enough
    to commit next to the harness.
    """
    _ffmpeg("-i", str(source), "-c:a", "libopus", "-b:a", "24k", str(target))


def concat_utterances(utterances: list[Utterance], target: Path) -> None:
    list_file = target.with_suffix(".list")
    list_file.write_text(
        "".join(f"file '{u.audio_path}'\n" for u in utterances), encoding="utf-8"
    )
    _ffmpeg(
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    )


def mix_noise(clean: Path, target: Path, snr_db: float, workdir: Path) -> None:
    """Overlay pink noise at a measured signal-to-noise ratio.

    Both RMS levels are measured with ``astats`` rather than assumed from
    generator amplitudes, so "5 dB SNR" in a filename is actually true of
    the audio.
    """
    noise = workdir / "noise.wav"
    duration = probe_duration(clean)
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"anoisesrc=color=pink:sample_rate=16000:amplitude=0.3:duration={duration}",
        "-ac",
        "1",
        str(noise),
    )
    gain_db = (measure_rms_db(clean) - snr_db) - measure_rms_db(noise)
    _ffmpeg(
        "-i",
        str(clean),
        "-i",
        str(noise),
        "-filter_complex",
        f"[1:a]volume={gain_db:.2f}dB[n];"
        "[0:a][n]amix=inputs=2:duration=first:normalize=0",
        str(target),
    )


def build_garbage_clip(target: Path, source: str, seconds: int) -> None:
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        source,
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "24k",
        str(target),
    )


def probe_duration(path: Path) -> float:
    stderr = _ffmpeg_stderr("-i", str(path), "-f", "null", "-")
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", stderr)
    if match is None:
        raise SystemExit(f"could not probe duration of {path.name}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def measure_rms_db(path: Path) -> float:
    stderr = _ffmpeg_stderr(
        "-i", str(path), "-af", "astats=measure_perchannel=none", "-f", "null", "-"
    )
    match = re.search(r"RMS level dB: (-?\d+\.?\d*)", stderr)
    if match is None:
        raise SystemExit(f"could not measure RMS of {path.name}")
    return float(match.group(1))


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
    )


def _ffmpeg_stderr(*args: str) -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", *args], capture_output=True, text=True
    )
    return result.stderr


if __name__ == "__main__":
    main()
