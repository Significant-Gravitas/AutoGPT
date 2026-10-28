"""Merging a ``SKILL.md``: frontmatter by field, body by line.

A copy the owner edited through ``store_skill`` has its frontmatter
re-rendered by the runtime, so line by line it differs from the catalog's
text even when no field changed. Merged as text, every later catalog change
to a field would collide with a "change" the owner never made, and the
owner's stale field would win. The frontmatter is a mapping, so it merges as
one: field by field, the owner's value winning a field both sides changed.
The body is prose and merges by line like any other file.
"""

import re
from typing import Any

import yaml
from pydantic import BaseModel

from backend.data.skill_package import MergedText, merge_text

# The closing fence's newline is required so the text goes back together
# byte for byte; a file without one merges as plain text.
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_MISSING = object()


def merge_skill_markdown(base: str, ours: str, theirs: str) -> MergedText:
    """Three-way merge of a ``SKILL.md``: *ours* is the owner's copy, *theirs*
    the update, *base* the version the copy was installed from. The owner's
    side wins any field or hunk both sides changed differently."""
    if ours == theirs:
        return MergedText(text=ours, conflicted=False)
    if base == ours:
        return MergedText(text=theirs, conflicted=False)
    if base == theirs:
        return MergedText(text=ours, conflicted=False)
    b, o, t = _split(base), _split(ours), _split(theirs)
    if b is None or o is None or t is None:
        return merge_text(base, ours, theirs)
    frontmatter = _merge_frontmatter(b.frontmatter, o.frontmatter, t.frontmatter)
    body = merge_text(b.body, o.body, t.body)
    return MergedText(
        text=f"---\n{frontmatter.text}\n---\n{body.text}",
        conflicted=frontmatter.conflicted or body.conflicted,
    )


class _Split(BaseModel):
    frontmatter: str
    body: str


def _split(text: str) -> _Split | None:
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return None
    return _Split(frontmatter=match.group(1), body=match.group(2))


def _merge_frontmatter(base: str, ours: str, theirs: str) -> MergedText:
    """Field by field. The text comes back untouched from whichever side the
    result equals, so a frontmatter nobody changed keeps the catalog's bytes;
    only a real three-way combination is re-rendered."""
    if ours == theirs or base == theirs:
        return MergedText(text=ours, conflicted=False)
    if base == ours:
        return MergedText(text=theirs, conflicted=False)
    b, o, t = _mapping(base), _mapping(ours), _mapping(theirs)
    if b is None or o is None or t is None:
        return merge_text(base, ours, theirs)
    merged: dict[str, Any] = {}
    conflicted = False
    for key in [
        *t,
        *(k for k in o if k not in t),
        *(k for k in b if k not in t and k not in o),
    ]:
        bv, ov, tv = b.get(key, _MISSING), o.get(key, _MISSING), t.get(key, _MISSING)
        if _same(ov, tv):
            value = ov
        elif _same(ov, bv):
            value = tv
        elif _same(tv, bv):
            value = ov
        else:
            conflicted = True
            value = ov
        if value is not _MISSING:
            merged[key] = value
    if _same_mapping(merged, t):
        return MergedText(text=theirs, conflicted=conflicted)
    if _same_mapping(merged, o):
        return MergedText(text=ours, conflicted=conflicted)
    rendered = yaml.safe_dump(merged, sort_keys=False).strip()
    return MergedText(text=rendered, conflicted=conflicted)


def _mapping(text: str) -> dict[str, Any] | None:
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, dict):
        return None
    return {str(key): value for key, value in loaded.items()}


def _same(a: object, b: object) -> bool:
    return _scalar(a) == _scalar(b)


def _same_mapping(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)


def _scalar(value: object) -> object:
    # ``version: 1`` in the catalog and ``version: '1'`` after the runtime
    # re-rendered it are the same field value, not an edit.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return value
