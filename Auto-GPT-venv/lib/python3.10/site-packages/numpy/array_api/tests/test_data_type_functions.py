import pytest

from numpy import array_api as xp


@pytest.mark.parametrize(
    "from_, to, expected",
    [
        (xp.int8, xp.int16, True),
        (xp.int16, xp.int8, False),
        (xp.bool, xp.int8, False),
        (xp.asarray(0, dtype=xp.uint8), xp.int8, False),
    ],
)
def test_can_cast(from_, to, expected):
    """
    can_cast() returns correct result
    """
    assert xp.can_cast(from_, to) == expected
