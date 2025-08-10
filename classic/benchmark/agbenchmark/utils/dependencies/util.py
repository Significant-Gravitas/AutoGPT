""" Utility functions to process the identifiers of tests. """
import re
from typing import Iterator

from _pytest.mark.structures import Mark
from _pytest.nodes import Item

from .constants import MARKER_KWARG_ID, MARKER_NAME

REGEX_PARAMETERS = re.compile(r"\[.+\]$")


def clean_nodeid(nodeid: str) -> str:
    """
    Remove any superfluous ::() from a node id.

    >>> clean_nodeid('test_file.py::TestClass::()::test')
    'test_file.py::TestClass::test'
    >>> clean_nodeid('test_file.py::TestClass::test')
    'test_file.py::TestClass::test'
    >>> clean_nodeid('test_file.py::test')
    'test_file.py::test'
    """
    return nodeid.replace("::()::", "::")


def strip_nodeid_parameters(nodeid: str) -> str:
    """
    Strip parameters from a node id.

    >>> strip_nodeid_parameters('test_file.py::TestClass::test[foo]')
    'test_file.py::TestClass::test'
    >>> strip_nodeid_parameters('test_file.py::TestClass::test')
    'test_file.py::TestClass::test'
    """
    return REGEX_PARAMETERS.sub("", nodeid)


def get_absolute_nodeid(nodeid: str, scope: str) -> str:
    """
    Transform a possibly relative node id to an absolute one
    using the scope in which it is used.

    >>> scope = 'test_file.py::TestClass::test'
    >>> get_absolute_nodeid('test2', scope)
    'test_file.py::TestClass::test2'
    >>> get_absolute_nodeid('TestClass2::test2', scope)
    'test_file.py::TestClass2::test2'
    >>> get_absolute_nodeid('test_file2.py::TestClass2::test2', scope)
    'test_file2.py::TestClass2::test2'
    """
    parts = nodeid.split("::")
    # Completely relative (test_name): add the full current scope (file::class or file)
    if len(parts) == 1:
        base_nodeid = scope.rsplit("::", 1)[0]
        nodeid = f"{base_nodeid}::{nodeid}"
    # Contains some scope already (Class::test_name), so only add the current file scope
    elif "." not in parts[0]:
        base_nodeid = scope.split("::", 1)[0]
        nodeid = f"{base_nodeid}::{nodeid}"
    return clean_nodeid(nodeid)


def get_name(item: Item) -> str:
    """
    Get all names for a test.

    This will use the following methods to determine the name of the test:
        - If given, the custom name(s) passed to the keyword argument name on the marker
    """
    name = ""

    # Custom name
    markers = get_markers(item, MARKER_NAME)
    for marker in markers:
        if MARKER_KWARG_ID in marker.kwargs:
            name = marker.kwargs[MARKER_KWARG_ID]

    return name


def get_markers(item: Item, name: str) -> Iterator[Mark]:
    """Get all markers with the given name for a given item."""
    for marker in item.iter_markers():
        if marker.name == name:
            yield marker
