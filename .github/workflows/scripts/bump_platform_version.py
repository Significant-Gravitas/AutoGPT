#!/usr/bin/env python3
"""Set the platform's package versions to the newest release's version.

    bump_platform_version.py <release tag> [--root <repository root>]
        [--tags <file with the published releases' tag names, one per line>]

The tag is a platform release tag, autogpt-platform-beta-v0.12.1, and the
version is the part after the "v". It goes into the backend's pyproject.toml
([tool.poetry] version) and the frontend's package.json (top-level
"version"). Only those lines change, so the rest of each file stays as it is.

The version set is the newest platform release's among the tag and those
in --tags, not always the tag's: a hotfix on an older line must not replace
a pending bump to a newer version, and whichever run is left after GitHub
drops a queued one still sets the newest. A file whose version is already
the same or newer is left alone.

When GITHUB_OUTPUT is set, it gets version=<version> and changed=true|false.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

TAG_RE = re.compile(r"autogpt-platform-(?:beta-)?v(\d+\.\d+\.\d+)")
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")

PYPROJECT = pathlib.Path("autogpt_platform/backend/pyproject.toml")
PACKAGE_JSON = pathlib.Path("autogpt_platform/frontend/package.json")

# The version line of the [tool.poetry] table, between its header and the
# next table header.
_POETRY_VERSION_RE = re.compile(
    r'(?m)^\[tool\.poetry\][ \t]*\n(?:(?!\[).*\n)*?version[ \t]*=[ \t]*"(?P<version>[^"]*)"'
)
# package.json is formatted with two-space indents, so the top-level
# "version" is the one key at exactly that depth.
_PACKAGE_VERSION_RE = re.compile(r'(?m)^  "version":[ \t]*"(?P<version>[^"]*)"')


class BumpError(Exception):
    pass


def version_from_tag(tag: str) -> str:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise BumpError(
            f"Release tag {tag!r} is not a platform release tag "
            "(autogpt-platform-beta-v<major>.<minor>.<patch>)."
        )
    return match.group(1)


def _as_tuple(version: str) -> tuple[int, ...]:
    if not VERSION_RE.fullmatch(version):
        raise BumpError(f"Version {version!r} is not <major>.<minor>.<patch>.")
    return tuple(int(part) for part in version.split("."))


def _replace_version(
    text: str, pattern: re.Pattern[str], new: str, name: str
) -> tuple[str, bool]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise BumpError(f"Expected one version line in {name}, found {len(matches)}.")
    [match] = matches
    current = match.group("version")
    if _as_tuple(current) >= _as_tuple(new):
        print(f"{name}: {current} is already at or past {new}; left as is.")
        return text, False
    start, end = match.span("version")
    print(f"{name}: {current} -> {new}")
    return text[:start] + new + text[end:], True


def bump_pyproject(text: str, version: str) -> tuple[str, bool]:
    return _replace_version(text, _POETRY_VERSION_RE, version, str(PYPROJECT))


def bump_package_json(text: str, version: str) -> tuple[str, bool]:
    new_text, changed = _replace_version(
        text, _PACKAGE_VERSION_RE, version, str(PACKAGE_JSON)
    )
    if changed and json.loads(new_text).get("version") != version:
        raise BumpError(
            f"The version line changed in {PACKAGE_JSON} is not its top-level version."
        )
    return new_text, changed


def newest_version(version: str, tags: list[str]) -> str:
    """The newest of *version* and the platform release versions in *tags*.

    Other products' tags and unprefixed ones don't count.
    """
    versions = [version]
    versions += [match.group(1) for tag in tags if (match := TAG_RE.fullmatch(tag))]
    return max(versions, key=_as_tuple)


def bump(root: pathlib.Path, version: str) -> bool:
    changed_any = False
    for path, bump_text in (
        (PYPROJECT, bump_pyproject),
        (PACKAGE_JSON, bump_package_json),
    ):
        file = root / path
        try:
            text = file.read_text(encoding="utf-8")
            new_text, changed = bump_text(text, version)
        except (OSError, ValueError) as error:  # ValueError: invalid JSON
            raise BumpError(f"{path}: {error}") from error
        if changed:
            file.write_text(new_text, encoding="utf-8")
            changed_any = True
    return changed_any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--tags", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        version = version_from_tag(args.tag)
        try:
            tags = args.tags.read_text(encoding="utf-8").split() if args.tags else []
        except OSError as error:
            raise BumpError(f"Can't read the tag list: {error}") from error
        newest = newest_version(version, tags)
        if newest != version:
            print(f"A newer platform release than {args.tag} exists: using {newest}.")
        version = newest
        changed = bump(args.root, version)
    except BumpError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as file:
            file.write(f"version={version}\nchanged={str(changed).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
