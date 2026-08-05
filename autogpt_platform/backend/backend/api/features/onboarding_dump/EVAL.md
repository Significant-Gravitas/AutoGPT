# Brain-dump transcription eval

WER (word error rate) harness for the onboarding brain-dump transcription
pipeline (`transcription.py`). It runs the **real** pipeline — same models,
same retry/fallback, same ffmpeg splitting and stitching — against a corpus of
recordings with human-written reference transcripts, and reports how far the
machine transcript drifts from the truth.

**Release gate: aggregate (pooled) WER must stay under 5%.** The harness exits
`1` when it is at or above 5%, `0` otherwise, so it can be wired into a release
check as-is.

## Running it

```bash
# from autogpt_platform/backend
poetry run brain-dump-eval --dir /path/to/dumps

# override the primary model (fallback stays whisper-1)
poetry run brain-dump-eval --dir /path/to/dumps --model gpt-4o-mini-transcribe

# duration fallback for chunked files whose container header reports none
poetry run brain-dump-eval --dir /path/to/dumps --duration-secs 1800

# also dump machine-readable results
poetry run brain-dump-eval --dir /path/to/dumps --json /tmp/wer.json
```

Run `poetry install` once after pulling this in — `brain-dump-eval` is a new
console script and won't exist in an older virtualenv. Equivalent without it:

```bash
poetry run python -m backend.api.features.onboarding_dump.brain_dump_eval --dir /path/to/dumps
```

Requires a direct OpenAI key (`OPENAI_INTERNAL_API_KEY` or `OPENAI_API_KEY`) —
OpenRouter does not implement `/audio/transcriptions`. Recordings above the
single-request byte cap also need `ffmpeg` on `PATH`.

**This costs real money and real minutes.** A 25-file, 2–15 minute corpus is
roughly 3 hours of audio per run. Run it before a transcription change ships,
not on every commit.

## The corpus

Put the recordings and their reference transcripts in one flat directory,
matched **by basename**:

```
dumps/
  dump01.webm      dump01.txt
  dump02.m4a       dump02.txt
  dump03-es.mp3    dump03-es.txt
```

- Audio extensions: `.webm` `.mp4` `.m4a` `.mp3` `.wav` `.ogg`
- Reference: same basename, `.txt`, UTF-8, plain text (no timestamps, no
  speaker labels)
- Unmatched files are skipped and listed at the bottom of the report, so a
  typo'd basename never silently shrinks the corpus.

### What to record — please help fill this out

We need **25+ real rambling dumps**. Not scripts read aloud: the pipeline's
failure mode is messy natural speech, so read-aloud audio will pass at ~1% WER
and tell us nothing. Target mix:

- **25+ recordings**, each **2–15 minutes** long
- **Mixed accents** — at minimum US, UK, Indian, and non-native-English
  speakers
- **Background noise** — a few from a café, a street, a room with a fan, a
  phone on speaker
- **At least 3–4 non-English dumps** (e.g. Spanish, French, Hindi). The
  pipeline must transcribe in the language spoken, never translate — a
  non-English dump that comes back in English is a bug even at low WER
- **At least 2 long ones (>20 MB)** so the chunked/stitched path is exercised
- Real rambling: false starts, "um", topic switches, self-corrections, long
  pauses

### Writing the reference transcript

Type what was actually said, verbatim: keep filler words and false starts, drop
nothing. Don't clean up grammar. Punctuation, casing and hyphenation don't
matter — the harness lowercases, strips punctuation and collapses whitespace
before comparing — but *words* do.

## Reading the report

```
file                      WER   sub   ins   del     ref    secs  segments  model
dump01                  3.20%     8     2     1     344    12.4  1         gpt-4o-transcribe
dump02                  7.10%    15     4     3     312    41.9  3         gpt-4o-transcribe,whisper-1

Stitch boundaries (chunked files):
  dump02 — 3 segments
    seam 0/1: dedup dropped 7 word(s)
      ...so then I went back to the drawing
      board and I thought maybe the whole thing
    seam 1/2: dedup dropped 0 word(s)
      ...

pooled WER      4.30%  (656 reference words)
mean WER        5.15%
median WER      5.15%
wall clock       54.3s

Release gate: aggregate WER must stay under 5% — PASS (exit 0)
```

- **pooled WER** — errors over all reference words in the corpus. This is the
  gated number; long files count more, which is what we want.
- **mean / median WER** — per-file averages. A mean far above the pooled number
  means a small file is failing badly; look at the per-file rows.
- **sub / ins / del** — substitutions dominate for accents and jargon,
  insertions for hallucination (the failure mode that matters most: a
  hallucinated sentence becomes a "fact" about the user), deletions for
  dropped audio and bad seams.
- **segments** — `1` means single-request; more means the chunked path ran.
- **model** — which STT model actually produced the transcript. The pipeline
  falls back silently, so a run meant to gate `gpt-4o-transcribe` that shows
  `whisper-1` here measured the fallback, not the model you asked for. A
  chunked file lists every model that contributed.
- **Stitch boundaries** — for chunked files only, the last ~10 words of each
  segment and the first ~10 of the next, plus how many words the overlap-dedup
  removed. `dropped 0` at a seam usually means the dedup failed to find the
  overlap and the transcript now repeats a phrase; a large drop means it ate
  real speech. Both show up as insertions/deletions in the WER but are only
  diagnosable here.

A file whose transcription raised is shown as `ERROR` and forces a failing exit
regardless of WER.

## Notes / limits

- The harness has no duration metadata for the files, so it takes the
  single-request vs chunked decision on the byte cap
  (`SINGLE_REQUEST_MAX_BYTES`) alone — exactly what
  `transcribe(audio, filename, duration_secs=None)` does. In production the
  frontend supplies a duration, so a long-but-small recording can chunk there
  and not here. When a chunked file's container header carries no duration
  either (the norm for browser `MediaRecorder` webm), pass `--duration-secs`;
  an upper bound is safe, since the split stops at the first empty segment.
- WER is computed with an inline Levenshtein over normalised words
  (`brain_dump_wer.py`); no `jiwer` dependency.
- The corpus is deliberately **not** committed — the recordings are personal
  speech. Keep them in shared storage and point `--dir` at a local copy.
