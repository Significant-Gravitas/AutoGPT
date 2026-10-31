import pytest
import regex

from backend.util import user_regex
from backend.util.user_regex import MAX_COMPILED_ITEMS, compile_user_regex

# Each compiles to just over the limit, through a different part of the count.
OVER_THE_LIMIT = [
    f"a{{{MAX_COMPILED_ITEMS}}}",
    f"a{{{MAX_COMPILED_ITEMS},}}",
    f"(?:a{{{MAX_COMPILED_ITEMS // 2}}})+",
    f"(?:(?:a{{10}}){{10}}){{{MAX_COMPILED_ITEMS // 100}}}",
    f"(?x)a{{{' '.join(str(MAX_COMPILED_ITEMS))}}}",
    f"(?i)[a-z]{{{MAX_COMPILED_ITEMS}}}",
    f"(?=a{{{MAX_COMPILED_ITEMS}}})",
]


@pytest.mark.parametrize("pattern", OVER_THE_LIMIT)
def test_a_pattern_over_the_limit_is_refused_before_it_is_compiled(
    mocker, pattern: str
):
    compile_pattern = mocker.patch.object(user_regex.regex, "compile")

    with pytest.raises(ValueError, match="repeats too much"):
        compile_user_regex(pattern)

    compile_pattern.assert_not_called()


@pytest.mark.parametrize(
    "pattern, text",
    [
        (f"a{{{MAX_COMPILED_ITEMS - 10}}}", "a" * (MAX_COMPILED_ITEMS - 10)),
        (r"^\d{4}-\d{2}-\d{2}$", "2026-10-31"),
        (r"^.{1,1000000}$", "any line"),
        ("|".join(f"word{index}" for index in range(500)), "word499"),
        (r"(?:colour){e<=1}", "color"),
        (r"(\d)(?1)", "42"),
    ],
)
def test_a_pattern_within_the_limit_compiles_and_matches(pattern: str, text: str):
    assert compile_user_regex(pattern).search(text, timeout=1.0)


def test_the_compiled_pattern_is_not_cached(mocker):
    compile_pattern = mocker.spy(user_regex.regex, "compile")

    compile_user_regex(r"\d+", regex.IGNORECASE)

    compile_pattern.assert_called_once_with(
        r"\d+", regex.IGNORECASE, cache_pattern=False
    )
