"""Three-way merge of an installed marketplace skill onto a newer version.

``base`` is the version the copy was installed from, ``theirs`` is the copy as
it stands in the owner's folder, and ``ours`` is the listing's new version.
Whatever the owner never touched takes the new value; whatever they changed
keeps their value. Instruction text merges line by line, so an owner's edit to
one paragraph and a catalog fix to another both survive. Where both sides
changed the same lines the owner's version wins, and the clash is returned as
a conflict so it can be surfaced later rather than lost.
"""

from difflib import SequenceMatcher
from typing import NamedTuple

from pydantic import BaseModel


class SkillContent(BaseModel):
    description: str
    body: str
    triggers: list[str]
    # relative path -> bytes; the SKILL.md itself is not in here.
    files: dict[str, bytes]


class MergeConflict(BaseModel):
    field: str
    base: str
    ours: str
    theirs: str


class MergedSkill(BaseModel):
    content: SkillContent
    conflicts: list[MergeConflict]


def merge_skill(
    base: SkillContent, theirs: SkillContent, ours: SkillContent
) -> MergedSkill:
    conflicts: list[MergeConflict] = []
    description = _merge_value(
        "description", base.description, theirs.description, ours.description, conflicts
    )
    body = merge_text(base.body, theirs.body, ours.body, conflicts)
    triggers = _merge_list(base.triggers, theirs.triggers, ours.triggers)
    files = _merge_files(base.files, theirs.files, ours.files, conflicts)
    return MergedSkill(
        content=SkillContent(
            description=description, body=body, triggers=triggers, files=files
        ),
        conflicts=conflicts,
    )


def merge_text(
    base: str, theirs: str, ours: str, conflicts: list[MergeConflict]
) -> str:
    """Line-level diff3. Overlapping or touching edits keep the owner's lines."""
    if theirs == base or theirs == ours:
        return ours
    if ours == base:
        return theirs
    base_lines = base.splitlines(keepends=True)
    theirs_lines = theirs.splitlines(keepends=True)
    ours_lines = ours.splitlines(keepends=True)
    hunks = sorted(
        _hunks(base_lines, theirs_lines, "theirs")
        + _hunks(base_lines, ours_lines, "ours"),
        key=lambda h: (h.start, h.end),
    )
    merged: list[str] = []
    position = 0
    for cluster in _clusters(hunks):
        start = min(h.start for h in cluster)
        end = max(h.end for h in cluster)
        merged += base_lines[position:start]
        theirs_side = _apply(base_lines, start, end, cluster, "theirs")
        ours_side = _apply(base_lines, start, end, cluster, "ours")
        sides = {h.side for h in cluster}
        if sides == {"ours"}:
            merged += ours_side
        elif sides == {"theirs"} or theirs_side == ours_side:
            merged += theirs_side
        else:
            merged += theirs_side
            conflicts.append(
                MergeConflict(
                    field="body",
                    base="".join(base_lines[start:end]),
                    ours="".join(ours_side),
                    theirs="".join(theirs_side),
                )
            )
        position = end
    merged += base_lines[position:]
    return "".join(merged)


class _Hunk(NamedTuple):
    start: int
    end: int
    lines: list[str]
    side: str


def _hunks(base: list[str], other: list[str], side: str) -> list[_Hunk]:
    return [
        _Hunk(i1, i2, other[j1:j2], side)
        for tag, i1, i2, j1, j2 in SequenceMatcher(
            None, base, other, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    ]


def _clusters(hunks: list[_Hunk]) -> list[list[_Hunk]]:
    """Group hunks whose base ranges overlap or touch."""
    clusters: list[list[_Hunk]] = []
    end = -1
    for hunk in hunks:
        if clusters and hunk.start <= end:
            clusters[-1].append(hunk)
            end = max(end, hunk.end)
        else:
            clusters.append([hunk])
            end = hunk.end
    return clusters


def _apply(
    base: list[str], start: int, end: int, cluster: list[_Hunk], side: str
) -> list[str]:
    """Base lines ``start:end`` with one side's hunks from the cluster applied."""
    result: list[str] = []
    position = start
    for hunk in (h for h in cluster if h.side == side):
        result += base[position : hunk.start]
        result += hunk.lines
        position = hunk.end
    result += base[position:end]
    return result


def _merge_value(
    field: str, base: str, theirs: str, ours: str, conflicts: list[MergeConflict]
) -> str:
    if theirs == base or theirs == ours:
        return ours
    if ours != base:
        conflicts.append(
            MergeConflict(field=field, base=base, ours=ours, theirs=theirs)
        )
    return theirs


def _merge_list(base: list[str], theirs: list[str], ours: list[str]) -> list[str]:
    """Keep the owner's additions and removals; apply the catalog's."""
    removed = set(base) - set(ours)
    kept = [item for item in theirs if item not in removed]
    return kept + [item for item in ours if item not in base and item not in kept]


def _merge_files(
    base: dict[str, bytes],
    theirs: dict[str, bytes],
    ours: dict[str, bytes],
    conflicts: list[MergeConflict],
) -> dict[str, bytes]:
    merged: dict[str, bytes] = {}
    for path in sorted(set(base) | set(theirs) | set(ours)):
        b, t, o = base.get(path), theirs.get(path), ours.get(path)
        if t == b or t == o:
            chosen = o
        else:
            chosen = t
            if o != b:
                conflicts.append(
                    MergeConflict(
                        field=f"files/{path}",
                        base=_describe(b),
                        ours=_describe(o),
                        theirs=_describe(t),
                    )
                )
        if chosen is not None:
            merged[path] = chosen
    return merged


def _describe(content: bytes | None) -> str:
    if content is None:
        return "(absent)"
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return f"({len(content)} bytes)"
