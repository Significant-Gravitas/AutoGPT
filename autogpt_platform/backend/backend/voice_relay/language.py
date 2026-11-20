"""Which language a turn's caption is in, when that is plain from its words.

The live model names no language on its captions, so a call counts Otto's
turns that came out in a language other than the call's from the words
alone: the script a turn is written in, or for a Latin-script turn, the
common short words of each language it has. It answers only when one
language clearly leads, and None otherwise, so a short or mixed turn is
never counted.
"""

import re
import unicodedata

# A caption's script, for the call languages written in one of their own.
_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("HIRAGANA", "ja"),
    ("KATAKANA", "ja"),
    ("HANGUL", "ko"),
    ("CJK", "zh"),
    ("ARABIC", "ar"),
    ("DEVANAGARI", "hi"),
    ("THAI", "th"),
    ("CYRILLIC", "ru"),
)
# Letters only Ukrainian of the Cyrillic call languages writes.
_UKRAINIAN = frozenset("іїєґІЇЄҐ")

# The commonest short words of each Latin-script call language, without the
# ones two of them share closely enough to mislead.
_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        "the and you your what that this with for are have how about do "
        "does it is to of in on we our they i my me so who which".split()
    ),
    "es": frozenset(
        "el la los las y que de en es un una por para con tu su qué cómo "
        "lo te se del al pero muy más yo mi usted ustedes también".split()
    ),
    "pt": frozenset(
        "o a os as e que de em é um uma por para com você seu sua não "
        "do da dos das no na mais muito também eu meu isso".split()
    ),
    "fr": frozenset(
        "le la les et que de en est un une pour avec vous votre ne pas "
        "du des au je mon ce qui sur très aussi".split()
    ),
    "de": frozenset(
        "der die das und ist ein eine für mit sie ihr nicht ich mein zu "
        "auf den dem was wie auch sehr".split()
    ),
    "it": frozenset(
        "il lo la gli le e che di è un una per con lei suo non io mio "
        "della del sono anche molto cosa come".split()
    ),
    "nl": frozenset(
        "de het een en is van dat voor met je jij uw niet ik mijn wat "
        "hoe ook zijn op".split()
    ),
    "pl": frozenset(
        "i w na nie że to jest się z do co jak pan pani mój moja także "
        "bardzo czy".split()
    ),
    "tr": frozenset(
        "ve bir bu da de ne için ile çok mi mı siz sizin ben benim nasıl "
        "ama gibi".split()
    ),
    "id": frozenset(
        "dan yang di ke dari ini itu untuk dengan anda saya tidak apa "
        "bagaimana juga sangat".split()
    ),
    "vi": frozenset(
        "và là của có không cho với bạn tôi này được những một các như "
        "gì rất".split()
    ),
    "sv": frozenset(
        "och är en ett att det som för med du din inte jag min vad hur "
        "också mycket på".split()
    ),
}
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
# The fewest of a language's words a turn needs, and how far it must lead
# the next language, before the turn is taken to be in it.
_MIN_HITS = 3
_LEAD = 2


def spoken_language(text: str) -> str | None:
    """The call language *text* is plainly in, or None when it is not plain."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    script = _by_script(letters)
    if script is not None:
        return script
    words = [word.lower() for word in _WORD.findall(text)]
    hits = sorted(
        (
            (sum(word in vocab for word in words), code)
            for code, vocab in _WORDS.items()
        ),
        reverse=True,
    )
    (best, code), (runner_up, _) = hits[0], hits[1]
    if best >= _MIN_HITS and best >= _LEAD * max(runner_up, 1):
        return code
    return None


def _by_script(letters: list[str]) -> str | None:
    counts: dict[str, int] = {}
    for ch in letters:
        name = unicodedata.name(ch, "")
        for prefix, code in _SCRIPTS:
            if name.startswith(prefix):
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return None
    code, count = max(counts.items(), key=lambda item: item[1])
    if count * 2 < len(letters):
        return None
    # Japanese writes kanji beside its kana; a turn with any kana is Japanese.
    if code == "zh" and counts.get("ja"):
        return "ja"
    if code == "ru" and any(ch in _UKRAINIAN for ch in letters):
        return "uk"
    return code
