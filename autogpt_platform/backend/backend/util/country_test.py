import pytest

from backend.util.country import ISO_3166_ALPHA2, country_code


def test_every_assigned_code_and_nothing_else():
    assert len(ISO_3166_ALPHA2) == 249
    assert {"US", "GB", "IN", "DE", "SS", "BQ"} <= ISO_3166_ALPHA2


@pytest.mark.parametrize(
    "value,expected",
    [
        ("US", "US"),
        (" in ", "IN"),
        (None, None),
        ("", None),
        ("IN, US", None),
        ("USA", None),
        ("XK", None),
        ("ZZ", None),
        ("EU", None),
        ("T1", None),
    ],
)
def test_country_code_accepts_only_assigned_codes(value, expected):
    assert country_code(value) == expected
