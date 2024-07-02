"""
A module to manage dependencies between pytest tests.

This module provides the methods implementing the main logic.
These are used in the pytest hooks that are in __init__.py.
"""

import collections
import os
from typing import Any, Generator

import colorama
import networkx
from pytest import Function, Item

from agbenchmark.challenges.base import BaseChallenge

from .constants import MARKER_KWARG_DEPENDENCIES, MARKER_NAME
from .graphs import graph_interactive_network
from .util import clean_nodeid, get_absolute_nodeid, get_markers, get_name


class TestResult(object):
    """Keeps track of the results of a single test."""

    STEPS = ["setup", "call", "teardown"]
    GOOD_OUTCOMES = ["passed"]

    def __init__(self, nodeid: str) -> None:
        """Create a new instance for a test with a given node id."""
        self.nodeid = nodeid
        self.results: dict[str, Any] = {}

    def register_result(self, result: Any) -> None:
        """Register a result of this test."""
        if result.when not in self.STEPS:
            raise ValueError(
                f"Received result for unknown step {result.when} of test {self.nodeid}"
            )
        if result.when in self.results:
            raise AttributeError(
                f"Received multiple results for step {result.when} "
                f"of test {self.nodeid}"
            )
        self.results[result.when] = result.outcome

    @property
    def success(self) -> bool:
        """Whether the entire test was successful."""
        return all(
            self.results.get(step, None) in self.GOOD_OUTCOMES for step in self.STEPS
        )


class TestDependencies(object):
    """Information about the resolved dependencies of a single test."""

    def __init__(self, item: Item, manager: "DependencyManager") -> None:
        """Create a new instance for a given test."""
        self.nodeid = clean_nodeid(item.nodeid)
        self.dependencies = set()
        self.unresolved = set()

        markers = get_markers(item, MARKER_NAME)
        dependencies = [
            dep
            for marker in markers
            for dep in marker.kwargs.get(MARKER_KWARG_DEPENDENCIES, [])
        ]
        for dependency in dependencies:
            # If the name is not known, try to make it absolute (file::[class::]method)
            if dependency not in manager.name_to_nodeids:
                absolute_dependency = get_absolute_nodeid(dependency, self.nodeid)
                if absolute_dependency in manager.name_to_nodeids:
                    dependency = absolute_dependency

            # Add all items matching the name
            if dependency in manager.name_to_nodeids:
                for nodeid in manager.name_to_nodeids[dependency]:
                    self.dependencies.add(nodeid)
            else:
                self.unresolved.add(dependency)


class DependencyManager(object):
    """Keep track of tests, their names and their dependencies."""

    def __init__(self) -> None:
        """Create a new DependencyManager."""
        self.options: dict[str, Any] = {}
        self._items: list[Function] | None = None
        self._name_to_nodeids: Any = None
        self._nodeid_to_item: Any = None
        self._results: Any = None

    @property
    def items(self) -> list[Function]:
        """The collected tests that are managed by this instance."""
        if self._items is None:
            raise AttributeError("The items attribute has not been set yet")
        return self._items

    @items.setter
    def items(self, items: list[Function]) -> None:
        if self._items is not None:
            raise AttributeError("The items attribute has already been set")
        self._items = items

        self._name_to_nodeids = collections.defaultdict(list)
        self._nodeid_to_item = {}
        self._results = {}
        self._dependencies = {}

        for item in items:
            nodeid = clean_nodeid(item.nodeid)
            # Add the mapping from nodeid to the test item
            self._nodeid_to_item[nodeid] = item
            # Add the mappings from all names to the node id
            name = get_name(item)
            self._name_to_nodeids[name].append(nodeid)
            # Create the object that will contain the results of this test
            self._results[nodeid] = TestResult(clean_nodeid(item.nodeid))

        # Don't allow using unknown keys on the name_to_nodeids mapping
        self._name_to_nodeids.default_factory = None

        for item in items:
            nodeid = clean_nodeid(item.nodeid)
            # Process the dependencies of this test
            # This uses the mappings created in the previous loop,
            # and can thus not be merged into that loop
            self._dependencies[nodeid] = TestDependencies(item, self)

    @property
    def name_to_nodeids(self) -> dict[str, list[str]]:
        """A mapping from names to matching node id(s)."""
        assert self.items is not None
        return self._name_to_nodeids

    @property
    def nodeid_to_item(self) -> dict[str, Function]:
        """A mapping from node ids to test items."""
        assert self.items is not None
        return self._nodeid_to_item

    @property
    def results(self) -> dict[str, TestResult]:
        """The results of the tests."""
        assert self.items is not None
        return self._results

    @property
    def dependencies(self) -> dict[str, TestDependencies]:
        """The dependencies of the tests."""
        assert self.items is not None
        return self._dependencies

    def print_name_map(self, verbose: bool = False) -> None:
        """Print a human-readable version of the name -> test mapping."""
        print("Available dependency names:")
        for name, nodeids in sorted(self.name_to_nodeids.items(), key=lambda x: x[0]):
            if len(nodeids) == 1:
                if name == nodeids[0]:
                    # This is just the base name, only print this when verbose
                    if verbose:
                        print(f"  {name}")
                else:
                    # Name refers to a single node id, so use the short format
                    print(f"  {name} -> {nodeids[0]}")
            else:
                # Name refers to multiple node ids, so use the long format
                print(f"  {name} ->")
                for nodeid in sorted(nodeids):
                    print(f"    {nodeid}")

    def print_processed_dependencies(self, colors: bool = False) -> None:
        """Print a human-readable list of the processed dependencies."""
        missing = "MISSING"
        if colors:
            missing = f"{colorama.Fore.RED}{missing}{colorama.Fore.RESET}"
            colorama.init()
        try:
            print("Dependencies:")
            for nodeid, info in sorted(self.dependencies.items(), key=lambda x: x[0]):
                descriptions = []
                for dependency in info.dependencies:
                    descriptions.append(dependency)
                for dependency in info.unresolved:
                    descriptions.append(f"{dependency} ({missing})")
                if descriptions:
                    print(f"  {nodeid} depends on")
                    for description in sorted(descriptions):
                        print(f"    {description}")
        finally:
            if colors:
                colorama.deinit()

    @property
    def sorted_items(self) -> Generator:
        """
        Get a sorted list of tests where all tests are sorted after their dependencies.
        """
        # Build a directed graph for sorting
        build_skill_tree = os.getenv("BUILD_SKILL_TREE")
        BUILD_SKILL_TREE = (
            build_skill_tree.lower() == "true" if build_skill_tree else False
        )
        dag = networkx.DiGraph()

        # Insert all items as nodes, to prevent items that have no dependencies
        # and are not dependencies themselves from being lost
        dag.add_nodes_from(self.items)

        # Insert edges for all the dependencies
        for item in self.items:
            nodeid = clean_nodeid(item.nodeid)
            for dependency in self.dependencies[nodeid].dependencies:
                dag.add_edge(self.nodeid_to_item[dependency], item)

        labels = {}
        for item in self.items:
            assert item.cls and issubclass(item.cls, BaseChallenge)
            data = item.cls.info.model_dump()

            node_name = get_name(item)
            data["name"] = node_name
            labels[item] = data

        # only build the tree if it's specified in the env and is a whole run
        if BUILD_SKILL_TREE:
            # graph_spring_layout(dag, labels)
            graph_interactive_network(dag, labels, html_graph_path="")

        # Sort based on the dependencies
        return networkx.topological_sort(dag)

    def register_result(self, item: Item, result: Any) -> None:
        """Register a result of a test."""
        nodeid = clean_nodeid(item.nodeid)
        self.results[nodeid].register_result(result)

    def get_failed(self, item: Item) -> Any:
        """Get a list of unfulfilled dependencies for a test."""
        nodeid = clean_nodeid(item.nodeid)
        failed = []
        for dependency in self.dependencies[nodeid].dependencies:
            result = self.results[dependency]
            if not result.success:
                failed.append(dependency)
        return failed

    def get_missing(self, item: Item) -> Any:
        """Get a list of missing dependencies for a test."""
        nodeid = clean_nodeid(item.nodeid)
        return self.dependencies[nodeid].unresolved
