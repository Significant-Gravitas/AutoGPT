import sys

from prisma import Json

from backend.util.json import SafeJson


class UnserializableScalar:
    __slots__ = ()

    def __str__(self) -> str:
        raise RuntimeError("cannot stringify")


def test_deeply_nested_container_does_not_become_repr():
    """A recursion failure must not store a repr string in a JSON column."""
    depth = sys.getrecursionlimit() + 200
    nested: list[object] = []
    current = nested
    for _ in range(depth):
        child: list[object] = []
        current.append(child)
        current = child

    result = SafeJson({"value": nested})
    list_result = SafeJson(nested)

    assert isinstance(result, Json)
    assert isinstance(list_result, Json)
    assert result.data == {}
    assert list_result.data == []


def test_scalar_serialization_error_returns_empty_object():
    result = SafeJson(UnserializableScalar())

    assert result.data == {}
