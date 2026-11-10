"""Content identity and three-way merging for skill packages.

A skill is a folder, not a file: ``SKILL.md`` plus whatever sits beside it
(``references/``, ``scripts/``, a LICENSE). Two places need to agree on what
"the same package" means without comparing bytes: the skills catalog, which
records a ``tree_sha256`` per package in ``release.json``, and the workspace,
which already stores a SHA-256 per file it writes. This module is the shared
formula, so a hash computed from a catalog checkout, from a marketplace
version's rows and from a user's folder listing is the same number.

The merge below is what lets a catalog update reach a copy the user has
edited. It is ``diff3`` per file with the user's side winning any hunk both
sides changed: an update never removes something the user wrote.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from itertools import accumulate

from pydantic import BaseModel, ConfigDict

SKILL_MD = "SKILL.md"

_NEWLINE = "\n"
# Above this many lines on any side the line alignment is not worth its CPU
# (it runs where every chat session on the process waits): the file counts as
# a conflict and the user's copy stays as it is.
MAX_MERGE_LINES = 5_000
# What one merge's alignments may spend, about a second of CPU; a file that
# needs more counts as a conflict too.
MAX_MERGE_COMPARISONS = 10_000_000


def file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def package_tree_sha256(files: Iterable[tuple[str, str, bool]]) -> str:
    """The catalog's ``tree_sha256``: SHA-256 of the canonical JSON of the
    ``(path, sha256, executable)`` list sorted by path. ``SKILL.md`` is one of
    the files. No timestamp, no environment, no commit."""
    listed = sorted(
        (
            {"path": path, "sha256": sha256, "executable": bool(executable)}
            for path, sha256, executable in files
        ),
        key=lambda item: item["path"],
    )
    canonical = json.dumps(
        listed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PackageFile(BaseModel):
    """One file of a package by its path relative to the skill folder."""

    model_config = ConfigDict(frozen=True)

    content: bytes
    executable: bool = False

    @property
    def sha256(self) -> str:
        return file_sha256(self.content)


Package = Mapping[str, PackageFile]


def package_hash(package: Package) -> str:
    return package_tree_sha256(
        (path, entry.sha256, entry.executable) for path, entry in package.items()
    )


class MergedText(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    # True when a hunk changed on both sides and the user's version was kept.
    conflicted: bool


class MergeBudget(BaseModel):
    """Comparisons the line alignments of one merge may still spend. The
    files of a package share one, so its cost does not grow with its files."""

    comparisons: int = MAX_MERGE_COMPARISONS


class _OverBudget(Exception):
    pass


def merge_text(
    base: str, ours: str, theirs: str, budget: MergeBudget | None = None
) -> MergedText:
    """Three-way merge of *ours* (the user's copy) and *theirs* (the update)
    against their common *base*, line by line.

    Hunks only one side touched take that side. A hunk both sides changed the
    same way is not a conflict. A hunk both sides changed differently keeps
    the user's lines and reports ``conflicted``: nothing the user wrote is
    ever dropped, and the update's version of that hunk is what they lose,
    which they can always fetch again from the marketplace.
    """
    if ours == theirs:
        return MergedText(text=ours, conflicted=False)
    if base == ours:
        return MergedText(text=theirs, conflicted=False)
    if base == theirs:
        return MergedText(text=ours, conflicted=False)
    # Split on the newline itself rather than with splitlines(): a trailing
    # newline then shows up as a final empty element, so whether a file ends
    # in one merges like any other line instead of making "b" and "b" plus a
    # newline two different lines.
    base_lines = base.split(_NEWLINE)
    our_lines = ours.split(_NEWLINE)
    their_lines = theirs.split(_NEWLINE)
    if max(len(base_lines), len(our_lines), len(their_lines)) > MAX_MERGE_LINES:
        return MergedText(text=ours, conflicted=True)
    try:
        chunks = _diff3_chunks(
            base_lines, our_lines, their_lines, budget or MergeBudget()
        )
    except _OverBudget:
        return MergedText(text=ours, conflicted=True)
    out: list[str] = []
    conflicted = False
    for chunk in chunks:
        if chunk.stable:
            out.extend(chunk.ours)
            continue
        if chunk.ours == chunk.base:
            out.extend(chunk.theirs)
        elif chunk.theirs == chunk.base or chunk.ours == chunk.theirs:
            out.extend(chunk.ours)
        else:
            conflicted = True
            out.extend(chunk.ours)
    return MergedText(text=_NEWLINE.join(out), conflicted=conflicted)


class _Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    stable: bool
    base: list[str]
    ours: list[str]
    theirs: list[str]


def _diff3_chunks(
    base: list[str], ours: list[str], theirs: list[str], budget: MergeBudget
) -> list[_Chunk]:
    """Split the three sequences into alternating stable and unstable chunks.

    A base line is *stable* when it survives unchanged into both sides and its
    neighbours do too, so the three sequences advance in step across it.
    Everything between two stable runs is one unstable chunk holding each
    side's lines for that gap, which is what the merge policy decides on.
    """
    ours_at = _match_index(base, ours, budget)
    theirs_at = _match_index(base, theirs, budget)
    chunks: list[_Chunk] = []
    b = o = t = 0
    i = 0
    while i < len(base):
        if ours_at[i] < 0 or theirs_at[i] < 0:
            i += 1
            continue
        # Start of a stable run at base[i]; extend while all three step together.
        j = i
        while (
            j + 1 < len(base)
            and ours_at[j + 1] == ours_at[j] + 1
            and theirs_at[j + 1] == theirs_at[j] + 1
        ):
            j += 1
        o_start, t_start = ours_at[i], theirs_at[i]
        if (i, o_start, t_start) != (b, o, t):
            chunks.append(
                _Chunk(
                    stable=False,
                    base=base[b:i],
                    ours=ours[o:o_start],
                    theirs=theirs[t:t_start],
                )
            )
        run = base[i : j + 1]
        chunks.append(_Chunk(stable=True, base=run, ours=run, theirs=run))
        b, o, t = j + 1, ours_at[j] + 1, theirs_at[j] + 1
        i = j + 1
    if (b, o, t) != (len(base), len(ours), len(theirs)):
        chunks.append(
            _Chunk(stable=False, base=base[b:], ours=ours[o:], theirs=theirs[t:])
        )
    return chunks


def _match_index(base: list[str], other: list[str], budget: MergeBudget) -> list[int]:
    """For each base line, its matched index in *other*, or -1.

    ``get_matching_blocks()`` one search at a time, each paid for before it
    runs: a search walks every copy in *other* of each base line in its range.
    """
    matched = [-1] * len(base)
    # The shared ends need no search, and after an ordinary edit they are most
    # of the file.
    limit = min(len(base), len(other))
    head = 0
    while head < limit and base[head] == other[head]:
        matched[head] = head
        head += 1
    tail = 0
    while tail < limit - head and base[-1 - tail] == other[-1 - tail]:
        tail += 1
        matched[-tail] = len(other) - tail
    matcher = SequenceMatcher(None, base, other, autojunk=False)
    copies = Counter(other)
    cost_before = list(accumulate((1 + copies[line] for line in base), initial=0))
    ranges = [(head, len(base) - tail, head, len(other) - tail)]
    while ranges:
        alo, ahi, blo, bhi = ranges.pop()
        cost = cost_before[ahi] - cost_before[alo]
        if cost > budget.comparisons:
            raise _OverBudget
        budget.comparisons -= cost
        a, b, size = matcher.find_longest_match(alo, ahi, blo, bhi)
        for k in range(size):
            matched[a + k] = b + k
        if size and alo < a and blo < b:
            ranges.append((alo, a, blo, b))
        if size and a + size < ahi and b + size < bhi:
            ranges.append((a + size, ahi, b + size, bhi))
    return matched


class MergedPackage(BaseModel):
    files: dict[str, PackageFile] = {}
    # Paths where both sides changed and the user's side was kept.
    conflicts: list[str] = []

    @property
    def conflicted(self) -> bool:
        return bool(self.conflicts)


def merge_packages(base: Package, ours: Package, theirs: Package) -> MergedPackage:
    """Merge an update (*theirs*) into the user's copy (*ours*) file by file,
    using the version they were installed from (*base*) to tell an edit from
    an unchanged file.

    Per path: a side that matches the base did nothing, so the other side's
    change (an edit, an addition, a deletion) goes through. Both sides
    changing the same way is agreement. Both sides changing differently is a
    conflict resolved in the user's favour: text is merged hunk by hunk with
    the user's hunks winning, anything else keeps the user's file, and a file
    the user deleted stays deleted. The executable bit merges the same way.
    """
    merged = MergedPackage()
    budget = MergeBudget()
    for path in sorted(set(base) | set(ours) | set(theirs)):
        b, o, t = base.get(path), ours.get(path), theirs.get(path)
        if o == t:
            if o is not None:
                merged.files[path] = o
            continue
        if b is None:
            # Not in the installed version: one side added it, or both did.
            if o is None:
                assert t is not None
                merged.files[path] = t
            elif t is None:
                merged.files[path] = o
            else:
                merged.files[path] = _merge_file(
                    path, None, o, t, merged.conflicts, budget
                )
            continue
        if o == b:
            # Untouched by the user: take the update, deletion included.
            if t is not None:
                merged.files[path] = t
            continue
        if t == b:
            # Untouched upstream: keep the user's version, deletion included.
            if o is not None:
                merged.files[path] = o
            continue
        # Both sides changed it differently.
        if o is None:
            # User deleted it; the update changed it. The user wins.
            merged.conflicts.append(path)
            continue
        if t is None:
            # Update deleted it; the user changed it. The user's file stays.
            merged.conflicts.append(path)
            merged.files[path] = o
            continue
        merged.files[path] = _merge_file(path, b, o, t, merged.conflicts, budget)
    return merged


def _merge_file(
    path: str,
    base: PackageFile | None,
    ours: PackageFile,
    theirs: PackageFile,
    conflicts: list[str],
    budget: MergeBudget,
) -> PackageFile:
    executable = _merge_flag(
        base.executable if base is not None else None,
        ours.executable,
        theirs.executable,
    )
    try:
        base_text = base.content.decode("utf-8") if base is not None else ""
        our_text = ours.content.decode("utf-8")
        their_text = theirs.content.decode("utf-8")
    except UnicodeDecodeError:
        conflicts.append(path)
        return PackageFile(content=ours.content, executable=executable)
    if base is None:
        # Added on both sides with different content: nothing to merge against.
        conflicts.append(path)
        return PackageFile(content=ours.content, executable=executable)
    result = merge_text(base_text, our_text, their_text, budget)
    if result.conflicted:
        conflicts.append(path)
    return PackageFile(content=result.text.encode("utf-8"), executable=executable)


def _merge_flag(base: bool | None, ours: bool, theirs: bool) -> bool:
    if ours == theirs or base is None:
        return ours
    return theirs if ours == base else ours
