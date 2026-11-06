"""Stock phrases speech-to-text models write when they hear no speech.

Fed silence or room noise, a speech-to-text model writes the outros and
captions of the videos it was trained on. The quality gate rejects a voice
take that is exactly one of these as ``NO_USABLE_SPEECH``, never as
"insufficient", so the not-enough screen never shows the person words they
did not say. Only well-known artifacts belong here, stored as
``normalise_for_artifact_match`` writes them. Typed text never matches: a
keyboard cannot mishear.
"""

import re

STT_ARTIFACT_PHRASES = frozenset(
    {
        "thank you",
        "thank you so much",
        "thanks for watching",
        "thank you for watching",
        "thank you so much for watching",
        "please subscribe",
        "like and subscribe",
        "bye",
        "bye bye",
        "you",
        "music",
        "applause",
        "subtitles by the amara org community",
    }
)
_NON_WORD = re.compile(r"[\W_]+")


def is_stt_artifact(text: str) -> bool:
    """Whether ``text`` is, word for word, a known speech-to-text artifact."""
    return normalise_for_artifact_match(text) in STT_ARTIFACT_PHRASES


def normalise_for_artifact_match(text: str) -> str:
    """``text`` lowercased, with punctuation and symbols turned into spaces
    and runs of whitespace collapsed.

    "[Music]", "♪ Music ♪" and "Bye-bye!" become "music", "music" and
    "bye bye".
    """
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())
