# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

import urllib.parse
from typing import Any, List, Optional, Set

from ._parser import parse_requirement
from ._tokenizer import ParserSyntaxError
from .markers import Marker, _normalize_extra_values
from .specifiers import SpecifierSet


class InvalidRequirement(ValueError):
    """
    An invalid requirement was found, users should refer to PEP 508.
    """


class Requirement:
    """Parse a requirement.

    Parse a given requirement string into its parts, such as name, specifier,
    URL, and extras. Raises InvalidRequirement on a badly-formed requirement
    string.
    """

    # TODO: Can we test whether something is contained within a requirement?
    #       If so how do we do that? Do we need to test against the _name_ of
    #       the thing as well as the version? What about the markers?
    # TODO: Can we normalize the name and extra name?

    def __init__(self, requirement_string: str) -> None:
        try:
            parsed = parse_requirement(requirement_string)
        except ParserSyntaxError as e:
            raise InvalidRequirement(str(e)) from e

        self.name: str = parsed.name
        if parsed.url:
            parsed_url = urllib.parse.urlparse(parsed.url)
            if parsed_url.scheme == "file":
                if urllib.parse.urlunparse(parsed_url) != parsed.url:
                    raise InvalidRequirement("Invalid URL given")
            elif not (parsed_url.scheme and parsed_url.netloc) or (
                not parsed_url.scheme and not parsed_url.netloc
            ):
                raise InvalidRequirement(f"Invalid URL: {parsed.url}")
            self.url: Optional[str] = parsed.url
        else:
            self.url = None
        self.extras: Set[str] = set(parsed.extras if parsed.extras else [])
        self.specifier: SpecifierSet = SpecifierSet(parsed.specifier)
        self.marker: Optional[Marker] = None
        if parsed.marker is not None:
            self.marker = Marker.__new__(Marker)
            self.marker._markers = _normalize_extra_values(parsed.marker)

    def __str__(self) -> str:
        parts: List[str] = [self.name]

        if self.extras:
            formatted_extras = ",".join(sorted(self.extras))
            parts.append(f"[{formatted_extras}]")

        if self.specifier:
            parts.append(str(self.specifier))

        if self.url:
            parts.append(f"@ {self.url}")
            if self.marker:
                parts.append(" ")

        if self.marker:
            parts.append(f"; {self.marker}")

        return "".join(parts)

    def __repr__(self) -> str:
        return f"<Requirement('{self}')>"

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, str(self)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Requirement):
            return NotImplemented

        return (
            self.name == other.name
            and self.extras == other.extras
            and self.specifier == other.specifier
            and self.url == other.url
            and self.marker == other.marker
        )
