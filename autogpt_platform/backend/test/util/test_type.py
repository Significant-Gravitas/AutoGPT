from backend.util.type import convert


def test_type_conversion():
    assert convert(5.5, int) == 5
    assert convert("5.5", int) == 5
    assert convert([1, 2, 3], int) == 3

    assert convert("5.5", float) == 5.5
    assert convert(5, float) == 5.0

    assert convert("True", bool) is True
    assert convert("False", bool) is False

    assert convert(5, str) == "5"
    assert convert({"a": 1, "b": 2}, str) == '{"a": 1, "b": 2}'
    assert convert([1, 2, 3], str) == "[1, 2, 3]"

    assert convert("5", list) == ["5"]
    assert convert((1, 2, 3), list) == [1, 2, 3]
    assert convert({1, 2, 3}, list) == [1, 2, 3]

    assert convert("5", dict) == {"value": 5}
    assert convert('{"a": 1, "b": 2}', dict) == {"a": 1, "b": 2}
    assert convert([1, 2, 3], dict) == {0: 1, 1: 2, 2: 3}
    assert convert((1, 2, 3), dict) == {0: 1, 1: 2, 2: 3}

    from typing import List

    assert convert("5", List[int]) == [5]
    assert convert("[5,4,2]", List[int]) == [5, 4, 2]
    assert convert([5, 4, 2], List[str]) == ["5", "4", "2"]
