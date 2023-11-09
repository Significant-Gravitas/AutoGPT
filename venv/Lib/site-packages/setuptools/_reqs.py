from typing import Callable, Iterable, Iterator, TypeVar, Union, overload

import setuptools.extern.jaraco.text as text
from setuptools.extern.packaging.requirements import Requirement

_T = TypeVar("_T")
_StrOrIter = Union[str, Iterable[str]]


def parse_strings(strs: _StrOrIter) -> Iterator[str]:
    """
    Yield requirement strings for each specification in `strs`.

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    return text.join_continuation(map(text.drop_comment, text.yield_lines(strs)))


@overload
def parse(strs: _StrOrIter) -> Iterator[Requirement]:
    ...


@overload
def parse(strs: _StrOrIter, parser: Callable[[str], _T]) -> Iterator[_T]:
    ...


def parse(strs, parser=Requirement):
    """
    Replacement for ``pkg_resources.parse_requirements`` that uses ``packaging``.
    """
    return map(parser, parse_strings(strs))
