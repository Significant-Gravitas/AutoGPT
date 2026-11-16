"""Names in the transcript of record, put right from the call's captions.

The transcript of record is made from the person's recording one turn at a
time, with only their first name for context, so it can mishear a name the
live model wrote correctly: their business, a product, a tool. After the
transcript is made, a cheap model proposes replacements for misheard names,
each one a span of the transcript and the name as the captions wrote it.
Code then takes a replacement only when all of these hold:

- the old span is in the transcript and the new one is in the captions, word
  for word, both on the person's side and on Otto's;
- each is a name of at most four words, the new one capitalised, and a name
  replaces a name: no word is added or dropped around it, and they share no
  word;
- the two are close in spelling, or in sound once vowels and doubled letters
  are set aside.

Everything else in the transcript is left as it is: the person's words stay
the record, and no sentence is ever added.
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


def accepted(old: str, new: str, *, transcript: str, user: str, otto: str) -> bool:
    """Whether code takes the replacement of *old* by *new* (module rules)."""
    if not old or not new or old == new or old not in transcript:
        return False
    if new.lower() not in user.lower() or new.lower() not in otto.lower():
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
        if not isinstance(proposal, dict):
            continue
        old = str(proposal.get("from", "")).strip()
        new = str(proposal.get("to", "")).strip()
        if not accepted(old, new, transcript=out, user=user, otto=otto):
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(old)}(?!\w)")
        replaced = pattern.sub(new, out)
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
