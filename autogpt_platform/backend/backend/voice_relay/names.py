"""Names in the transcript of record, put right from the call's captions.

The transcript of record is made from the person's recording one turn at a
time, with only their first name for context, so it can mishear a name the
live model wrote correctly: their business, a product, a tool. After the
transcript is made, a cheap model proposes replacements for misheard names,
each one a span of the transcript and the name as the captions wrote it.
Code then takes a replacement only when all of these hold:

- the old span stands as a whole word in the transcript and nowhere in the
  captions: where the captions wrote it too, it was not misheard;
- the new one stands as a whole word, case and all, in both the person's
  captions and Otto's;
- each is a capitalised name of at most four words, with no markup in it,
  and a name replaces a name: no word is added or dropped around it, and
  they share no word;
- the two are close in spelling, or in sound once vowels and doubled letters
  are set aside;
- a call takes at most ``MAX_REPLACEMENTS``.

Everything else in the transcript is left as it is: the person's words stay
the record, and no sentence is ever added. The new name is written as it
stands, never read as a pattern, and the server's tags are taken out of the
transcript again afterwards (``postcall``).
"""

import difflib
import json
import logging
import re
import time
from typing import Protocol

from google.genai import Client, types
from pydantic import BaseModel, ConfigDict

from backend.voice_relay.config import DETAILS_MODEL

logger = logging.getLogger(__name__)

MAX_NAME_WORDS = 4
# The most replacements one call's transcript takes.
MAX_REPLACEMENTS = 3
# Never part of a name: a replacement holding one is refused.
_MARKUP = ("<", ">", "\\")
MAX_CAPTION_CHARS = 6_000

PROMPT = """\
Below is a USER's speech from a call, transcribed from their recording (RECORD), and the live captions of the same call (CAPTIONS: the USER's lines and Otto's lines). The RECORD may have misheard proper nouns: names of businesses, products, apps, places or people.

Propose replacements only for proper nouns in RECORD that are misheard versions of a proper noun written in CAPTIONS. Copy "from" exactly as it appears in RECORD and "to" exactly as it appears in CAPTIONS. Propose nothing else. Return JSON {{"replacements": [{{"from": "...", "to": "..."}}]}}, an empty list when nothing needs changing.

RECORD:
{record}

CAPTIONS:
USER: {user}
OTTO: {otto}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "replacements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                "required": ["from", "to"],
            },
        }
    },
    "required": ["replacements"],
}


class NamesRead(BaseModel):
    """The transcript with its names put right, and what the read cost."""

    model_config = ConfigDict(frozen=True)

    transcript: str
    replaced: list[tuple[str, str]]
    tokens_in: int
    tokens_out: int
    latency_s: float


def _words(text: str) -> set[str]:
    return set(re.sub(r"[^\w]+", " ", text.lower()).split())


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _sound_key(letters: str) -> str:
    key = (
        letters.replace("ph", "f")
        .replace("ck", "k")
        .replace("c", "k")
        .replace("z", "s")
    )
    key = key[0] + re.sub(r"[aeiouyhw]", "", key[1:])
    return re.sub(r"(.)\1+", r"\1", key)


def sounds_close(old: str, new: str) -> bool:
    a, b = _letters(old), _letters(new)
    if not a or not b:
        return False
    if difflib.SequenceMatcher(None, a, b).ratio() >= 0.6:
        return True
    return difflib.SequenceMatcher(None, _sound_key(a), _sound_key(b)).ratio() >= 0.75


def _whole_word(span: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(span)}(?!\w)", flags)


def accepted(old: str, new: str, *, transcript: str, user: str, otto: str) -> bool:
    """Whether code takes the replacement of *old* by *new* (module rules)."""
    if not old or not new or old == new:
        return False
    if any(mark in old + new for mark in _MARKUP):
        return False
    if not old[0].isupper() or not new[0].isupper():
        return False
    if not _whole_word(old).search(transcript):
        return False
    # The new name is written, case and all, as a whole word on both sides of
    # the call.
    if not (_whole_word(new).search(user) and _whole_word(new).search(otto)):
        return False
    # The captions wrote the old span too: the record and the captions agree
    # on it, so it was not misheard.
    if _whole_word(old, re.IGNORECASE).search(f"{user}\n{otto}"):
        return False
    old_words, new_words = _words(old), _words(new)
    if not new_words or len(old_words) != len(new_words) or old_words & new_words:
        return False
    if len(old.split()) > MAX_NAME_WORDS or len(new.split()) > MAX_NAME_WORDS:
        return False
    if not new[0].isupper():
        return False
    return sounds_close(old, new)


def apply_replacements(
    transcript: str, proposals: object, *, user: str, otto: str
) -> tuple[str, list[tuple[str, str]]]:
    """*transcript* with each accepted replacement made wherever its old span
    stands as a whole word; every other character untouched."""
    out = transcript
    made: list[tuple[str, str]] = []
    if not isinstance(proposals, list):
        return out, made
    for proposal in proposals:
        if len(made) >= MAX_REPLACEMENTS:
            break
        if not isinstance(proposal, dict):
            continue
        old = str(proposal.get("from", "")).strip()
        new = str(proposal.get("to", "")).strip()
        if not accepted(old, new, transcript=out, user=user, otto=otto):
            continue
        replaced = _whole_word(old).sub(lambda _: new, out)
        if replaced != out:
            out = replaced
            made.append((old, new))
    return out, made


async def reconcile_names(
    client: Client,
    *,
    model: str,
    transcript: str,
    user_captions: str,
    otto_captions: str,
) -> NamesRead:
    """Put the transcript's misheard names right from the captions. Raises
    what the client raises; a caller keeps the transcript as it was then."""
    started = time.monotonic()
    response = await client.aio.models.generate_content(
        model=model,
        contents=PROMPT.format(
            record=transcript,
            user=user_captions[-MAX_CAPTION_CHARS:],
            otto=otto_captions[-MAX_CAPTION_CHARS:],
        ),
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_json_schema=SCHEMA,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    try:
        proposals = json.loads(response.text or "{}").get("replacements")
    except (ValueError, AttributeError):
        proposals = None
    fixed, made = apply_replacements(
        transcript, proposals, user=user_captions, otto=otto_captions
    )
    usage = response.usage_metadata
    return NamesRead(
        transcript=fixed,
        replaced=made,
        tokens_in=(usage.prompt_token_count or 0) if usage else 0,
        tokens_out=(
            (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
            if usage
            else 0
        ),
        latency_s=round(time.monotonic() - started, 3),
    )


class NamesReader(Protocol):
    async def __call__(
        self, transcript: str, *, user_captions: str, otto_captions: str
    ) -> NamesRead: ...


class GeminiNamesReader:
    def __init__(self, client: Client, *, model: str = DETAILS_MODEL) -> None:
        self._client = client
        self._model = model

    async def __call__(
        self, transcript: str, *, user_captions: str, otto_captions: str
    ) -> NamesRead:
        return await reconcile_names(
            self._client,
            model=self._model,
            transcript=transcript,
            user_captions=user_captions,
            otto_captions=otto_captions,
        )
