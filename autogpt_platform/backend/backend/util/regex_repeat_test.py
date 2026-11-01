import pytest

from backend.util.regex_repeat import largest_repeat_count

LIMIT = 1000


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("plain text", 1),
        (r"a+b*c?", 1),
        ("a{5}", 5),
        ("a{2,7}", 7),
        ("a{3,}", 3),
        ("a{,9}", 9),
        ("a{2,7}?", 7),
        ("x{1}y{20}z{3}", 20),
        ("(ab){20}", 20),
        ("(a{10}){20}", 200),
        ("((a{10}){10}){10}", 1000),
        ("(?:a{5}|b{50})c{2}", 50),
        (r"\d{3}-\d{4}", 4),
        (r"\p{L}{3}", 3),
        (r"\N{BULLET}", 1),
        (r"\{99999}", 1),
        ("[{]99999}", 1),
        ("[]{]99999}", 1),
        (r"[\]{]99999}", 1),
        ("[^]{]99999}", 1),
        ("a{0005}", 5),
        ("(a{20}", 20),
    ],
)
def test_largest_repeat_count(pattern, expected):
    assert largest_repeat_count(pattern, LIMIT) == expected


@pytest.mark.parametrize(
    "pattern",
    [
        "a{1001}",
        "a{30000000}",
        "a{2,30000000}",
        "a{30000000,}",
        "ok{2}then{5000}",
        "(a{100}){100}",
        "((a{10}){10}){11}",
        "[ab]{5000}",
        r"\p{L}{5000}",
        "a{" + "9" * 5000 + "}",
    ],
)
def test_largest_repeat_count_caps_counts_over_the_limit(pattern):
    assert largest_repeat_count(pattern, LIMIT) == LIMIT + 1
