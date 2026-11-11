"""Stock phrases speech-to-text models write when they hear no speech.

Fed silence or room noise, a speech-to-text model writes the outros and
captions of the videos it was trained on, often several in a row. The
quality gate rejects a voice take made of nothing but these as
``NO_USABLE_SPEECH``, never as "insufficient", so the not-enough screen
never shows the person words they did not say. Only well-known artifacts
belong here, stored as ``normalise_for_artifact_match`` writes them. Typed
text never matches: a keyboard cannot mishear.
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
_LONGEST_PHRASE_WORDS = max(len(phrase.split()) for phrase in STT_ARTIFACT_PHRASES)


def is_stt_artifact(text: str) -> bool:
    """Whether ``text`` is nothing but known artifact phrases, one or more.

    "Thank you.", "Thank you. Bye." and "[Music] [Applause]" are; "Thank you
    for taking this off my plate." is not, so it still reaches the judge.
    """
    words = normalise_for_artifact_match(text).split()
    if not words:
        return False
    # covered[i]: the first i words split exactly into artifact phrases.
    covered = [True] + [False] * len(words)
    for end in range(1, len(words) + 1):
        covered[end] = any(
            covered[start] and " ".join(words[start:end]) in STT_ARTIFACT_PHRASES
            for start in range(max(0, end - _LONGEST_PHRASE_WORDS), end)
        )
    return covered[-1]


def normalise_for_artifact_match(text: str) -> str:
    """``text`` lowercased, with punctuation and symbols turned into spaces
    and runs of whitespace collapsed.

    "[Music]", "♪ Music ♪" and "Bye-bye!" become "music", "music" and
    "bye bye".
    """
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())
