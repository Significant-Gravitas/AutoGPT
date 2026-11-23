"""Whether a drafted task restates one that is already taken, in other words.

No model call: both sides are cut down to content-word stems (stop words and
verbs that name no particular action dropped, plurals and verb endings
folded), then compared as sets.

Where the line sits: a draft repeats a taken item when they share at least
``_MIN_SHARED`` stems and those make up ``_TITLE_OVERLAP`` of all stems in the
two titles together, or, when both carry a goal, ``_WITH_GOAL_OVERLAP`` of
title and goal together. A real follow-up adds its own action or object
("Turn the supplier comparison into a deck" after "Compare supplier offers"
shares 2 of 7 stems) and stays well under both; a reword keeps the subject
and swaps a verb or a synonym ("Work out the paneer price impact across
dairy supplier offers" against "Compare dairy supplier offers and cost
impact of paneer prices" shares 6 of 8). The cost is a follow-up that only
changes the verb over the same four or more nouns ("Order paneer from Kisan
Fresh at the revised rate" after "Get a revised paneer rate from Kisan
Fresh") reads as a repeat; the prompt carries that distinction instead.
"""

import re

from pydantic import BaseModel

_MIN_SHARED = 3
_TITLE_OVERLAP = 0.7
_WITH_GOAL_OVERLAP = 0.6
_WORD_RE = re.compile(r"[a-z0-9]+")
# Function words, and verbs that say nothing about which task it is.
_STOP = frozenset(
    """a an the and or but of for to in on at by with from into onto over per
    via about after before across up out off this that these those it its
    their our your you we they them us me my is are be been being was were
    will can could should would may might must as so than then also all any
    each every some new next task tasks get got make made do does done work
    working take put set sort handle look looking follow following figure
    keep let need needed help start see""".split()
)


class TakenItem(BaseModel):
    """Something a draft must not restate: a board task, a live, dismissed
    or recently retired suggestion, or another draft of the same batch."""

    title: str
    goal: str | None = None


def is_repeat(title: str, goal: str | None, taken: TakenItem) -> bool:
    title_stems, taken_stems = stems(title), stems(taken.title)
    if not title_stems or not taken_stems:
        return _folded(title) == _folded(taken.title)
    if title_stems == taken_stems or _overlaps(
        title_stems, taken_stems, _TITLE_OVERLAP
    ):
        return True
    if not goal or not taken.goal:
        return False
    return _overlaps(
        title_stems | stems(goal), taken_stems | stems(taken.goal), _WITH_GOAL_OVERLAP
    )


def stems(text: str) -> set[str]:
    """The content words of ``text``, each cut to a rough stem."""
    words = _WORD_RE.findall(text.lower())
    return {_stem(word) for word in words if word not in _STOP and len(word) > 2}


def _overlaps(a: set[str], b: set[str], ratio: float) -> bool:
    shared = len(a & b)
    return shared >= _MIN_SHARED and shared / len(a | b) >= ratio


def _stem(word: str) -> str:
    """Fold the endings that rewording changes: prices/price, revised/revise,
    offers/offer, booking/book. Rough on purpose; both sides get the same."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("ing", "ed")) and len(word) > 5:
        word = word[:-3] if word.endswith("ing") else word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


def _folded(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))
