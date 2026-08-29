"""Tests for TextFormatter resource-exhaustion protections (GHSA-ppw9-h7rv-gwq9)."""

import asyncio

import pytest

from backend.util.text import TextFormatter


@pytest.fixture
def fmt() -> TextFormatter:
    return TextFormatter(autoescape=False)


# --- Normal usage should still work ---


def test_simple_variable_substitution(fmt: TextFormatter):
    result = asyncio.run(fmt.format_string("Hello, {{ name }}!", {"name": "Alice"}))
    assert result == "Hello, Alice!"


def test_loop(fmt: TextFormatter):
    result = asyncio.run(
        fmt.format_string(
            "{% for item in items %}{{ item }} {% endfor %}",
            {"items": ["a", "b", "c"]},
        )
    )
    assert result == "a b c "


def test_conditional(fmt: TextFormatter):
    result = asyncio.run(
        fmt.format_string(
            "{% if x > 5 %}big{% else %}small{% endif %}",
            {"x": 10},
        )
    )
    assert result == "big"


def test_safe_range(fmt: TextFormatter):
    result = asyncio.run(
        fmt.format_string("{% for i in range(5) %}{{ i }}{% endfor %}")
    )
    assert result == "01234"


def test_small_exponent(fmt: TextFormatter):
    result = asyncio.run(fmt.format_string("{{ 2**10 }}"))
    assert result == "1024"


def test_safe_string_repeat(fmt: TextFormatter):
    result = asyncio.run(fmt.format_string("{{ 'ab' * 3 }}"))
    assert result == "ababab"


# --- Resource-exhaustion attacks should be blocked ---


def test_huge_exponent_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ 999999999**999999999 }}"))


def test_large_base_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ 999999999**100 }}"))


def test_negative_exponent_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ 2**-99999 }}"))


def test_huge_range_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ range(999999999) | list }}"))


def test_large_string_repeat_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ 'A' * 100000 }}"))


def test_large_list_repeat_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ [0] * 999999999 }}"))


def test_large_tuple_repeat_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ (0,) * 999999999 }}"))


def test_pow_function_blocked(fmt: TextFormatter):
    """pow() builtin should also be capped."""
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ pow(2, 999999) }}"))


def test_nested_exponentiation_blocked(fmt: TextFormatter):
    """{{ 2 ** (2 ** 100) }} — inner result exceeds MAX_EXPONENT as base."""
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ 2 ** (2 ** 100) }}"))


# --- Edge cases ---


def test_moderate_range_allowed(fmt: TextFormatter):
    """range(100) should work fine — well under the limit."""
    result = asyncio.run(fmt.format_string("{{ range(100) | list | length }}"))
    assert result == "100"


def test_boundary_range(fmt: TextFormatter):
    """range(10000) is exactly at the limit — should work."""
    result = asyncio.run(fmt.format_string("{{ range(10000) | list | length }}"))
    assert result == "10000"


def test_over_boundary_range(fmt: TextFormatter):
    """range(10001) exceeds the limit — should fail."""
    with pytest.raises((ValueError, OverflowError)):
        asyncio.run(fmt.format_string("{{ range(10001) | list }}"))
