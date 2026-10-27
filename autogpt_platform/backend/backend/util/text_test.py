"""Tests for TextFormatter resource-exhaustion protections (GHSA-ppw9-h7rv-gwq9)."""

import asyncio

import pytest

from backend.util.text import (
    MAX_FORMAT_FIELD_SIZE,
    TextFormatter,
    _check_percent_format,
)


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


# --- The format filter's field sizes are capped ---


@pytest.mark.asyncio
async def test_format_filter_pads_within_the_cap(fmt: TextFormatter):
    result = await fmt.format_string(
        "{{ 5 | format('%" + str(MAX_FORMAT_FIELD_SIZE) + ".3f') }}"
    )
    assert len(result) == MAX_FORMAT_FIELD_SIZE
    assert result.endswith("5.000")


@pytest.mark.asyncio
async def test_huge_format_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string(
            "{{ 0 | format('%" + str(MAX_FORMAT_FIELD_SIZE + 1) + "f') }}"
        )


@pytest.mark.asyncio
async def test_huge_format_precision_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string(
            "{{ 0 | format('%." + str(MAX_FORMAT_FIELD_SIZE + 1) + "f') }}"
        )


@pytest.mark.asyncio
async def test_huge_format_width_blocked_in_a_later_field(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string(
            "{{ 0 | format('%%%" + str(MAX_FORMAT_FIELD_SIZE + 1) + "f') }}"
        )


# --- The same cap covers the other ways a template can format a string ---

TOO_WIDE = str(MAX_FORMAT_FIELD_SIZE + 1)


@pytest.mark.asyncio
async def test_percent_operator_pads_within_the_cap(fmt: TextFormatter):
    assert await fmt.format_string("{{ '%5d' % 1 }}") == "    1"


@pytest.mark.asyncio
async def test_huge_percent_operator_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '%" + TOO_WIDE + "d' % 1 }}")


@pytest.mark.asyncio
async def test_str_format_pads_within_the_cap(fmt: TextFormatter):
    assert await fmt.format_string("{{ '{:5}'.format(1) }}") == "    1"


@pytest.mark.asyncio
async def test_huge_str_format_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '{:" + TOO_WIDE + "}'.format(1) }}")


@pytest.mark.asyncio
async def test_huge_str_format_precision_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '{:." + TOO_WIDE + "f}'.format(1.5) }}")


@pytest.mark.asyncio
async def test_huge_nested_str_format_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '{:{w}}'.format(1, w=" + TOO_WIDE + ") }}")


@pytest.mark.asyncio
async def test_huge_format_map_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '{n:" + TOO_WIDE + "}'.format_map({'n': 1}) }}")


@pytest.mark.asyncio
async def test_huge_safe_markup_format_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ ('{:" + TOO_WIDE + "}' | safe).format(1) }}")


@pytest.mark.asyncio
async def test_star_percent_width_pads_within_the_cap(fmt: TextFormatter):
    assert await fmt.format_string("{{ '%*d' % (5, 1) }}") == "    1"


@pytest.mark.asyncio
async def test_huge_star_percent_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '%*d' % (" + TOO_WIDE + ", 1) }}")


@pytest.mark.asyncio
async def test_huge_star_percent_precision_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '%d %.*f' % (1, " + TOO_WIDE + ", 1.5) }}")


@pytest.mark.asyncio
async def test_huge_keyed_percent_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '%(n)" + TOO_WIDE + "d' % {'n': 1} }}")


@pytest.mark.asyncio
async def test_huge_str_format_width_with_newline_fill_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '{:\n>" + TOO_WIDE + "}'.format(1) }}")


@pytest.mark.asyncio
async def test_nested_key_percent_width_pads_within_the_cap(fmt: TextFormatter):
    assert await fmt.format_string("{{ '%(a(b))5d' % {'a(b)': 1} }}") == "    1"


@pytest.mark.asyncio
async def test_huge_nested_key_percent_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '%(a(b))" + TOO_WIDE + "d' % {'a(b)': 1} }}")


@pytest.mark.asyncio
async def test_bytes_percent_operator_pads_within_the_cap(fmt: TextFormatter):
    assert await fmt.format_string("{{ '%5d'.encode() % 1 }}") == "b'    1'"


@pytest.mark.asyncio
async def test_huge_bytes_percent_width_blocked(fmt: TextFormatter):
    with pytest.raises((ValueError, OverflowError)):
        await fmt.format_string("{{ '%" + TOO_WIDE + "d'.encode() % 1 }}")


# CPython rejects a literal "%%" that carries a width, but it reads a "*" size
# for it first, so the cap counts the arguments the same way.


def test_star_size_on_a_literal_percent_is_counted_in_the_argument_order():
    with pytest.raises(OverflowError):
        _check_percent_format("%*%%*d", (1, MAX_FORMAT_FIELD_SIZE + 1, 1))


def test_star_sizes_within_the_cap_pass_after_a_literal_percent():
    _check_percent_format("%*%%*d", (1, 5, 1))
