"""Guards shared by the external API's tests."""

import pytest

from backend.copilot.tools import TOOL_REGISTRY

_METHODS_BEFORE = pytest.StashKey[dict[str, frozenset[str]]]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item):
    item.stash[_METHODS_BEFORE] = _methods_on_instances()
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item):
    """Fail a test that leaves a method override on a shared tool instance.

    v1, v2 and MCP all run the instances in ``TOOL_REGISTRY``. Undoing
    ``monkeypatch.setattr(tool, "_execute", fake)`` sets the original bound
    method back onto the instance, where it shadows every later patch of the
    tool's class: the next test that patches ``type(tool)`` runs the real
    tool. Patch an instance with ``mocker.patch.object``, which removes the
    attribute again, or patch the class. Checked once every fixture, the
    ``monkeypatch`` undo included, has finished.
    """
    try:
        return (yield)
    finally:
        _fail_on_methods_left(item.stash.get(_METHODS_BEFORE, {}))


def _fail_on_methods_left(before: dict[str, frozenset[str]]) -> None:
    left = {
        name: attrs - before.get(name, frozenset())
        for name, attrs in _methods_on_instances().items()
    }
    left = {name: attrs for name, attrs in left.items() if attrs}
    # Removed before failing, so only the test that left them fails.
    for name, attrs in left.items():
        for attr in attrs:
            delattr(TOOL_REGISTRY[name], attr)
    if left:
        pytest.fail(
            "Methods left on shared tool instances: "
            f"{ {name: sorted(attrs) for name, attrs in left.items()} }"
        )


def _methods_on_instances() -> dict[str, frozenset[str]]:
    return {
        name: frozenset(
            attr
            for attr, value in vars(tool).items()
            if callable(value) and callable(getattr(type(tool), attr, None))
        )
        for name, tool in TOOL_REGISTRY.items()
    }
