"""Characters that hide in a name a person reads.

A folder's name goes into a dialog a person reads, and from there into the
path a chat is bound to; an app's name is what a launch names. Such a name
must not hold a character that shows as nothing or changes the order of what
is shown. The bidi controls (U+202A to U+202E, U+2066 to U+2069, U+200E,
U+200F, U+061C) reorder a name, and a zero-width character shows as nothing:
`Documents` followed by one looks like a second `Documents`.

So a name with a character of a Unicode `C` category in it counts as hiding
something: a control, a format character (the bidi controls and the
zero-width ones are format characters), a private-use or unassigned code
point, a lone surrogate.

Two format characters are part of how some scripts are written, and a name
that uses one of them that way hides nothing:

- U+200C ZERO WIDTH NON-JOINER keeps two letters of a cursive script from
  joining, inside many Persian words, and keeps two Indic consonants from
  forming a conjunct.
- U+200D ZERO WIDTH JOINER asks for an Indic conjunct or half form, and
  builds one picture from several in an emoji sequence (a family, a rainbow
  flag).

Where each may stand is Unicode's own rule for names, CONTEXTJ in IDNA 2008
(RFC 5892, appendix A.1 and A.2). The `idna` package implements it
(`idna.valid_contextj`) and carries the joining types it needs, which
Python's `unicodedata` doesn't; httpx brings that package, and the shim
names it among its own dependencies. An emoji sequence is no part of that
rule (IDNA has no emoji), and is taken as below.

What is let through:

- Either one right after a virama (a character of combining class 9),
  wherever in the name that is. Malayalam and Bengali wrote a word's final
  consonant as virama and joiner before they had a letter for it, so the end
  of a name is no exception. (A.1 and A.2.)
- U+200C between a letter that joins the one after it and a letter that
  joins the one before it, marks between them set aside: Arabic, Syriac,
  N'Ko, Mongolian and the other cursive scripts. (A.1.)
- U+200D between two pictographs, the first with an emoji selector (U+FE0F)
  or a skin tone after it. A pictograph here is a symbol (general category
  So) in the symbol blocks (U+2190 to U+2BFF) or the pictograph blocks
  (U+1F000 to U+1FAFF), or the arrow U+2194, which Unicode files under math
  and which the head-shaking faces are built with.

What still counts as hidden:

- Either one at the start of a name, beside another one, or between letters
  that don't join: Latin, Cyrillic, Han. U+200C at the end of a name, except
  after a virama.
- U+200C after an Arabic letter that doesn't join the next one (alef, dal,
  reh, waw). It changes nothing there, and CONTEXTJ refuses it.
- U+200D in Arabic-script text, where it shows a letter's joining form, and
  Kannada's ra, joiner, virama. CONTEXTJ has neither.
- The tag characters (U+E0020 to U+E007F) that England's, Scotland's and
  Wales' flags are written with, like every other format character.

What this doesn't decide:

- Whether a font draws two pictographs with a joiner between them as one
  picture. Between two that are no sequence the joiner shows as nothing, and
  the name reads like the one without it.
- Anything outside the `C` categories. A letter that looks like another one,
  another script's space, and the few characters elsewhere that show as
  nothing (a variation selector after a letter, a Hangul filler) are not
  refused, here or before.
- A character newer than this Python's Unicode tables. To them it is
  unassigned, and it is refused.
"""

from __future__ import annotations

import unicodedata

import idna

ZERO_WIDTH_NON_JOINER = "\u200c"
ZERO_WIDTH_JOINER = "\u200d"

_EMOJI_SELECTOR = "\ufe0f"
_SKIN_TONES = range(0x1F3FB, 0x1F400)
# U+2194 LEFT RIGHT ARROW: a math symbol (Sm) to Unicode, and a pictograph in
# the sequences that use it.
_PICTOGRAPHS_FILED_ELSEWHERE = frozenset("↔")


def has_hidden_character(name: str) -> bool:
    """Whether `name` holds a character that shows as nothing or reorders
    what is shown: any character of a `C` category, except a joiner or
    non-joiner where its script uses one (see the module's docstring)."""
    for index, character in enumerate(name):
        if not unicodedata.category(character).startswith("C"):
            continue
        if character in (ZERO_WIDTH_NON_JOINER, ZERO_WIDTH_JOINER) and _joins(name, index):
            continue
        return True
    return False


def _joins(name: str, index: int) -> bool:
    """Whether the joiner or non-joiner at `index` stands where its script
    uses one."""
    try:
        if idna.valid_contextj(name, index):
            return True
    except ValueError:
        # idna looks up the name of the character before it, and one without
        # a name (a control, an unassigned code point) is no letter.
        return False
    return name[index] == ZERO_WIDTH_JOINER and _between_pictographs(name, index)


def _between_pictographs(name: str, index: int) -> bool:
    before = index - 1
    if before >= 0 and (name[before] == _EMOJI_SELECTOR or ord(name[before]) in _SKIN_TONES):
        before -= 1
    return (
        before >= 0
        and _is_pictograph(name[before])
        and index + 1 < len(name)
        and _is_pictograph(name[index + 1])
    )


def _is_pictograph(character: str) -> bool:
    code = ord(character)
    if not (0x2190 <= code <= 0x2BFF or 0x1F000 <= code <= 0x1FAFF):
        return False
    return unicodedata.category(character) == "So" or character in _PICTOGRAPHS_FILED_ELSEWHERE


__all__ = ["ZERO_WIDTH_JOINER", "ZERO_WIDTH_NON_JOINER", "has_hidden_character"]
