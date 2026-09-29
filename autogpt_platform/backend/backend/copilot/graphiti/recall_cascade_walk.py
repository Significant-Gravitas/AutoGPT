"""What a forget's cascade carries from round to round (``recall_cascade.py``).

A ``Walk`` maps every fact and episode the cascade has reached to the root
it descends from, the uuid its retractions name in their reason
(``derived_from_forgotten:<root>``): the fact the user forgot, or, for a
cascade the derivation backfill starts from a hidden episode whose forgotten
fact is gone, the root that episode was hidden for (``redacted_for``) or the
episode itself.
"""

from typing import Any

from pydantic import BaseModel, Field

# ``expiration_reason`` of a fact the cascade retracted, before the colon.
DERIVED_FROM_FORGOTTEN = "derived_from_forgotten"


def derived_reason(root: str) -> str:
    """The reason recorded on a fact derived from forgotten fact ``root``."""
    return f"{DERIVED_FROM_FORGOTTEN}:{root}"


class Walk(BaseModel):
    """The fact ``roots`` a cascade starts from and the hidden ``seeds``
    (episode to the root it names), the ``names`` of all their roots, every
    fact and episode reached so far mapped to its root, how many more
    derived items it may retire, and whether it erases their text."""

    roots: list[str]
    seeds: dict[str, str] = Field(default_factory=dict)
    names: list[str]
    root_of: dict[str, str]
    budget: int
    erase: bool = False

    @classmethod
    def start(
        cls,
        roots: list[str],
        seeds: dict[str, str],
        *,
        budget: int,
        erase: bool,
    ) -> "Walk":
        names = list(dict.fromkeys([*roots, *seeds.values()]))
        root_of = {root: root for root in roots} | seeds
        return cls(
            roots=roots,
            seeds=seeds,
            names=names,
            root_of=root_of,
            budget=budget,
            erase=erase,
        )

    def root(self, via: list[str]) -> str:
        """The root of the first item in ``via`` the walk has reached."""
        return next((self.root_of[x] for x in via if x in self.root_of), self.names[0])

    def reach(self, rows: list[dict[str, Any]]) -> list[str]:
        """Record each row's ``uuid`` as reached ``via`` its items; the uuids
        not reached before."""
        new = [row for row in rows if row["uuid"] not in self.root_of]
        for row in new:
            self.root_of[row["uuid"]] = self.root(row["via"])
        return [row["uuid"] for row in new]


class Frontier(BaseModel):
    """What the last round retracted and hid: the next round's search."""

    facts: list[str] = Field(default_factory=list)
    episodes: list[str] = Field(default_factory=list)


class Found(BaseModel):
    """One round's finds, rows of ``uuid`` and ``via`` (and, for a fact,
    ``live``); ``truncated`` when the walk's budget cut them short."""

    facts: list[dict[str, Any]] = Field(default_factory=list)
    episodes: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
