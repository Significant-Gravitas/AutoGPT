"""Quality gate between transcription and personalized generation.

A transcription request succeeding is not the same as the user saying
anything. Silence, background noise, an STT hallucination or four seconds
of filler all come back as a "successful" transcript — and everything
downstream (greeting, suggested prompts, provider picks) would confidently
personalize from it.

Layered on purpose so the common cases stay free:

1. Deterministic reject — unmistakably unusable text (empty, no letters,
   a looping/repeated transcript) never reaches a model.
2. Deterministic pass — clearly substantial text (a real rambling dump)
   never pays for a model call either.
3. Everything in between gets one cheap LLM judgement: "is there enough
   concrete information here to personalize from?". Length alone cannot
   make this call — "Automate my Shopify refund emails" is six words and
   must pass, "uh hello hello testing" must not.

The gate never judges grammar, accent, fluency or language: a coherent
dump in any language passes. And it never deletes anything — the caller
keeps the audio and transcript on the row, exactly like every other
failure in this pipeline.
"""

import asyncio
import logging
import os
import zlib
from typing import Literal

from backend.api.features.onboarding_dump.parsing import parse_response_json
from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)

# Stored in ``OnboardingBrainDump.errorCode`` (a plain string column, so no
# enum migration) and read by the frontend to show the recovery screen.
NO_USABLE_SPEECH = "no_usable_speech"
INSUFFICIENT_CONTENT = "insufficient_content"
QUALITY_ERROR_CODES = frozenset({NO_USABLE_SPEECH, INSUFFICIENT_CONTENT})

_MODEL = os.environ.get("BRAIN_DUMP_QUALITY_MODEL", "anthropic/claude-haiku-4-5")
# The gate runs inside the finalize request, after transcription already
# spent part of the frontend proxy's 30s budget. A 100-token Haiku verdict
# typically lands in a couple of seconds; a stalled provider must degrade
# to the recoverable reject quickly instead of eating the rest of the
# request budget.
_TIMEOUT_SECONDS = 8

# Above this many words a transcript is clearly substantial and skips the
# LLM entirely — the vast majority of real dumps land here, which is what
# keeps the gate's cost near zero. ``split()`` can't see words in
# space-less scripts (Chinese, Japanese, ...), so those get a
# character-based route instead — but only when the chars-per-"word"
# ratio proves the script really is space-less, or 25 words of English
# could slip past the word threshold on characters alone. Both routes
# apply *after* the repetition checks, so a looping transcript cannot
# buy its way through on sheer length.
CLEAR_PASS_WORDS = 40
CLEAR_PASS_CHARS = 100
SPACELESS_CHARS_PER_WORD = 10

# The classic STT failure on silence/noise is a decoder loop: the same
# word or phrase over and over. Both checks are standard Whisper-style
# signals — a low unique-word ratio and a text that zlib collapses.
REPETITION_MIN_WORDS = 20
MIN_UNIQUE_WORD_RATIO = 0.3
COMPRESSION_MIN_CHARS = 80
MAX_COMPRESSION_RATIO = 2.4

_PROMPT = """A new user of an automation platform was asked to describe \
their work and what they'd like automated. Below is the raw transcript \
(spoken or typed). Decide whether it contains enough concrete information \
to personalize a greeting or propose work for them — anything about who \
they are, what they do, tools they use, or tasks they want handled counts.

The transcript may be in ANY language; a coherent non-English answer \
counts fully. Ignore grammar, fluency, accent artifacts and how \
interesting the work sounds. Judge ONLY whether there is usable substance.

Answer "usable": false when the text is only filler, greetings, mic \
testing, repeated fragments, or generic phrases a speech model produces \
from silence or background noise (e.g. "thank you for watching").

Return ONLY valid JSON: {{"usable": true}} or {{"usable": false}}.

Transcript:
{transcript}
"""


async def check_transcript_quality(transcript: str) -> str | None:
    """Return an error code when ``transcript`` can't support personalization.

    ``None`` means the dump is usable and the personalized pipeline may
    run. Deterministic answers are preferred in both directions; only the
    ambiguous middle costs an LLM call.
    """
    text = transcript.strip()
    verdict = _deterministic_verdict(text)
    if verdict == "reject":
        return NO_USABLE_SPEECH
    if verdict == "pass":
        return None
    return await _semantic_verdict(text)


def _deterministic_verdict(text: str) -> Literal["pass", "reject", "ambiguous"]:
    if not text or not any(ch.isalnum() for ch in text):
        return "reject"

    words = [word for word in text.split() if any(ch.isalnum() for ch in word)]
    if not words:
        return "reject"
    if len(words) >= REPETITION_MIN_WORDS:
        unique_ratio = len({w.lower() for w in words}) / len(words)
        if unique_ratio < MIN_UNIQUE_WORD_RATIO:
            return "reject"
    if (
        len(text) >= COMPRESSION_MIN_CHARS
        and _compression_ratio(text) > MAX_COMPRESSION_RATIO
    ):
        return "reject"

    alnum_chars = sum(1 for ch in text if ch.isalnum())
    is_spaceless_script = alnum_chars / len(words) >= SPACELESS_CHARS_PER_WORD
    if len(words) >= CLEAR_PASS_WORDS or (
        is_spaceless_script and alnum_chars >= CLEAR_PASS_CHARS
    ):
        return "pass"
    return "ambiguous"


def _compression_ratio(text: str) -> float:
    """How far zlib collapses ``text`` — high means the decoder was looping."""
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw))


async def _semantic_verdict(text: str) -> str | None:
    """One cheap LLM call for the short-but-nonempty middle band.

    Every failure path here — no client, timeout, malformed output —
    resolves to ``INSUFFICIENT_CONTENT`` rather than a silent pass: an
    ambiguous dump must never produce confident personalized output just
    because the judge was unavailable. The outcome is recoverable by
    design (the user is offered retry / type / skip), which is the safe
    documented behavior for a quality-model outage.
    """
    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        logger.warning("Brain dump quality check: no LLM client configured")
        return INSUFFICIENT_CONTENT

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": _PROMPT.format(transcript=text)}],
                temperature=0,
                max_tokens=100,
            ),
            timeout=_TIMEOUT_SECONDS,
        )
        content = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Brain dump quality check failed: %s", e)
        return INSUFFICIENT_CONTENT

    data = parse_response_json(content)
    if not isinstance(data, dict) or not isinstance(data.get("usable"), bool):
        logger.warning("Brain dump quality check: malformed verdict")
        return INSUFFICIENT_CONTENT
    return None if data["usable"] else INSUFFICIENT_CONTENT
