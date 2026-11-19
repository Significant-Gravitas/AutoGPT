"""Tests for TextFormatter resource-exhaustion protections (GHSA-ppw9-h7rv-gwq9)."""

import asyncio
import os
import resource
import sys
from pathlib import Path

import jinja2
import pytest
from markupsafe import Markup

from backend.util import text
from backend.util.mock import MockObject
from backend.util.text import (
    MAX_FORMAT_FIELD_SIZE,
    MAX_RENDER_OUTPUT,
    MAX_RENDER_WORKERS,
    RENDER_MEMORY_LIMIT,
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


# --- A render runs in a worker process, under its memory and time limits ---

# The templates below build gigabytes by many routes, all past the worker's
# memory limit or the output cap. The sizes are template literals on purpose:
# a test that computed them from the limits would pass whatever they were.
# Some routes are refused for memory, some for output, and the slow ones at
# the deadline; any of the three is the guarantee.
REFUSED = "out of memory|output too large|timed out|too large"
# The worker's memory limit is set on Linux only, and /proc is read there too
LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux", reason="needs the worker's memory limit and /proc"
)
HUGE = "2000000000"
# Built while rendering, because Jinja folds a constant value into the
# compiled code, and compiling it costs several times its size
_X = "('x' ~ range(0)|join)"
# A hundred thousand characters, every one an "x"
BIG_VALUE = "{% set a = " + _X + "|center(100000)|replace(' ', 'x') %}"
# Five megabytes, so what multiplies it is what the case is about
MEGABYTES = "{% set a = " + _X + "|center(5000000) %}"
# Most of the memory limit, so what the next operation does with it decides
MOST_OF_THE_LIMIT = "{% set a = " + _X + "|center(400000000) %}"

_LIST_LITERAL = "[" + ",".join(["1"] * 1000) + "]"
_DICT_LITERAL = "{" + ",".join(f"'k{i}':1" for i in range(500)) + "}"
# Items the loop supplies, so each pass really does build another tuple
_LOOP_TUPLE = "(" + ",".join(["i"] * 1000) + ")"
# One Python keeps a single object for, so no pass builds anything
_CONST_TUPLE = "(" + ",".join(["1"] * 626) + ")"


def _rebuilt_every_iteration(literal: str) -> str:
    """A loop that builds ``literal`` again on each of 100,000 passes."""
    return (
        "{% set acc = [] %}{% for i in range(10000) %}{% for j in range(10) %}"
        f"{{% set _ = acc.append({literal}) %}}"
        "{% endfor %}{% endfor %}{{ acc|length }}"
    )


OVERSIZED_TEMPLATES = {
    "center filter": "{{ 'x'|center(" + HUGE + ")|length }}",
    "indent filter": "{{ 'a\nb'|indent(" + HUGE + ")|length }}",
    "join filter": BIG_VALUE + "{{ range(10000)|join(a ~ a)|length }}",
    "join filter over big items": (
        BIG_VALUE + "{% set l = [a] * 10000 %}{{ l|join|length }}"
    ),
    "replace filter": BIG_VALUE + "{{ a|replace('x', a ~ a)|length }}",
    "center method": "{{ 'x'.center(" + HUGE + ")|length }}",
    "ljust method": "{{ 'x'.ljust(" + HUGE + ")|length }}",
    "rjust method": "{{ 'x'.rjust(" + HUGE + ")|length }}",
    "zfill method": "{{ '1'.zfill(" + HUGE + ")|length }}",
    "expandtabs method": "{{ '\t'.expandtabs(" + HUGE + ")|length }}",
    "join method": (BIG_VALUE + "{% set l = [a] * 10000 %}{{ (a~a).join(l)|length }}"),
    "replace method": BIG_VALUE + "{{ a.replace('x', a ~ a)|length }}",
    "translate method": BIG_VALUE + "{{ (a~a).translate({120: a ~ a})|length }}",
    "bytes repeat": "{{ ('x'.encode() * " + HUGE + ")|length }}",
    "tilde concatenation": (
        "{% set a = 'xx' %}"
        + "{% set a = a~a~a~a~a~a~a~a~a~a %}" * 9
        + "{{ a|length }}"
    ),
    "plus concatenation": (
        "{% set a = 'xx' %}"
        + "{% set a = a+a+a+a+a+a+a+a+a+a %}" * 9
        + "{{ a|length }}"
    ),
    "a loop's output": (
        BIG_VALUE + "{% for i in range(10000) %}{{ a }}{{ a }}{% endfor %}"
    ),
    "a loop's literal text": "{% for i in range(10000) %}"
    + "y" * 20000
    + "{% endfor %}",
    "a loop into a set block": (
        BIG_VALUE
        + "{% set b %}{% for i in range(10000) %}{{ a }}{{ a }}{% endfor %}{% endset %}"
        + "{{ b|length }}"
    ),
    "a lazily mapped chain": "{{ range(10000)|map('center', 1000000)|join|length }}",
    "repeated squaring": (
        "{% set x = 10**1000 %}" + "{% set x = x*x %}" * 14 + "{{ x|string|length }}"
    ),
    "list extension": (
        "{% set l = [1] * 10000 %}" + "{{ l.extend(l) }}" * 25 + "{{ l|length }}"
    ),
    "batch filter's fill": "{{ [1]|batch(" + HUGE + ", 'x')|list|length }}",
    "slice filter's count": "{{ [1]|slice(" + HUGE + ")|list|length }}",
    "a container emitted whole": BIG_VALUE + "{% set l = [a] * 10000 %}{{ l }}",
    "a container nested three deep": (
        BIG_VALUE
        + "{% set inner = [a] * 100 %}{% set middle = [inner] * 10 %}"
        + "{{ [middle] * 10 }}"
    ),
    "a list literal rebuilt every iteration": _rebuilt_every_iteration(_LIST_LITERAL),
    "a dict literal rebuilt every iteration": _rebuilt_every_iteration(_DICT_LITERAL),
    "a tuple of loop values rebuilt every iteration": _rebuilt_every_iteration(
        _LOOP_TUPLE
    ),
    "a constant tuple holding a list, rebuilt every iteration": (
        _rebuilt_every_iteration("(1, " + _LIST_LITERAL + ")")
    ),
    "a list nested in a literal, rebuilt every iteration": _rebuilt_every_iteration(
        "[" + _LIST_LITERAL + "]"
    ),
    "a slice taken every iteration": BIG_VALUE + _rebuilt_every_iteration("a[1:]"),
    "a control-character string emitted whole": "{{ [v] }}",
    "the % operator without a precision": MOST_OF_THE_LIMIT + "{{ '%s' % a }}",
    "str.format's receiver": MOST_OF_THE_LIMIT + "{{ a.format() }}",
    "string filter on a container": (
        BIG_VALUE + "{% set l = [a] * 10000 %}{{ l|string|length }}"
    ),
    "the % operator's arguments": (MEGABYTES + "{{ ('%s' * 5000) % ((a,) * 5000) }}"),
    "str.format's arguments": (
        MEGABYTES + "{{ ('{}' * 5000).format(*((a,) * 5000)) }}"
    ),
    "a lazy join's separator": (
        MEGABYTES + "{{ range(10000)|map('string')|join(a)|length }}"
    ),
    "round to a negative precision": "{{ (10**100)|round(-" + HUGE + ") }}",
    "round to a precision through ceil": "{{ 5|round(" + HUGE + ", 'ceil') }}",
    "round to a precision through floor": "{{ 5|round(" + HUGE + ", 'floor') }}",
    "a filter applied once per item": (
        BIG_VALUE + "{% set l = [a] * 10000 %}{{ (l|map('lower')|list)|length }}"
    ),
    "a filter over a caller's list": "{{ (l|map('lower')|list)|length }}",
    "wordwrap filter's wrap string": (
        BIG_VALUE
        + "{% set t = 'a ' * 5000 %}"
        + "{{ (t ~ t)|wordwrap(1, wrapstring=a)|length }}"
    ),
    "sum filter over big items": (
        BIG_VALUE + "{% set l = [[a]] * 10000 %}{{ l|sum(start=[])|string|length }}"
    ),
    "sum filter by attribute": (
        BIG_VALUE
        + "{% set l = [{'v': [a]}] * 10000 %}"
        + "{{ l|sum(attribute='v', start=[])|string|length }}"
    ),
    "markup search over plain text": (
        BIG_VALUE + "{{ ('&' * 10000)|replace('amp'|safe, a)|length }}"
    ),
    "plain replacement into markup": (
        "{% set q = '\"' * 10000 %}"
        "{{ ((('x' * 10000) ~ ('x' * 10000))|safe)|replace('x', q)|length }}"
    ),
}

# The cases whose values arrive from the caller rather than the template
OVERSIZED_VALUES = {
    "a filter over a caller's list": {"l": ["x" * 100_000] * 10_000},
    # `repr` renders each of these as four characters
    "a control-character string emitted whole": {"v": "\x00" * 16_000_000},
}
# These escape what they build
ESCAPING = {"markup search over plain text", "plain replacement into markup"}


@LINUX_ONLY
@pytest.mark.parametrize(
    ("name", "template"), OVERSIZED_TEMPLATES.items(), ids=OVERSIZED_TEMPLATES
)
@pytest.mark.asyncio
async def test_an_oversized_render_is_refused(name: str, template: str):
    # ValueError, not OverflowError or MemoryError: the app's global handlers
    # answer a ValueError with 400 and anything else with 500 and a logged
    # error, and an oversized template is the caller's
    formatter = TextFormatter(autoescape=name in ESCAPING)
    with pytest.raises(ValueError, match=REFUSED):
        await formatter.format_string(template, OVERSIZED_VALUES.get(name), timeout=3)


# Twenty items sharing one 5 MB field that a join by another attribute never
# reads, so the join is small whatever the items weigh
ONE_PAYLOAD_TWENTY_TIMES = {"l": [{"name": "a", "payload": "p" * 5_000_000}] * 20}

WORKING_TEMPLATES = {
    "substitution": ("Hello {{ name }}!", {"name": "World"}, "Hello World!"),
    "loop": ("{% for i in range(3) %}{{ i }},{% endfor %}", {}, "0,1,2,"),
    "center filter": ("{{ 'x'|center(9) }}|", {}, "    x    |"),
    "indent filter": ("{{ 'a\nb'|indent(2) }}", {}, "a\n  b"),
    "join filter": ("{{ ['a','b','c']|join('-') }}", {}, "a-b-c"),
    "replace filter": ("{{ 'aaa'|replace('a','bb') }}", {}, "bbbbbb"),
    "replace filter with numbers": ("{{ 123|replace(1, 2) }}", {}, "223"),
    "a join by attribute": (
        "{{ l|join(',', attribute='name') }}",
        ONE_PAYLOAD_TWENTY_TIMES,
        ",".join("a" * 20),
    ),
    "a lazy join by attribute": (
        "{{ l|select|join(',', attribute='name') }}",
        ONE_PAYLOAD_TWENTY_TIMES,
        ",".join("a" * 20),
    ),
    "sum filter": ("{{ [1, 2, 3]|sum }}", {}, "6"),
    "sum by attribute": ("{{ [{'n': 1}, {'n': 2}]|sum(attribute='n') }}", {}, "3"),
    "sum of large integers": (
        "{% set x = 10**1000 %}"
        + "{% set x = x*x %}" * 4
        + "{{ ([x] * 10000)|sum == x * 10000 }}",
        {},
        "True",
    ),
    "map filter": ("{{ ['a','b']|map('upper')|join }}", {}, "AB"),
    "select filter": ("{{ [1,2,3]|select('odd')|list|join(',') }}", {}, "1,3"),
    "urlencode filter": ("{{ 'a b'|urlencode }}", {}, "a%20b"),
    "wordwrap filter": ("{{ 'aaa bbb'|wordwrap(3) }}", {}, "aaa\nbbb"),
    "wordwrap with a wrap string": (
        "{{ 'aaa bbb ccc'|wordwrap(3, wrapstring='<br>') }}",
        {},
        "aaa<br>bbb<br>ccc",
    ),
    "filesizeformat filter": ("{{ 1000|filesizeformat }}", {}, "1.0 kB"),
    "batch filter": ("{{ [1,2,3]|batch(2, 'x')|list|length }}", {}, "2"),
    "slice filter": ("{{ [1,2,3]|slice(2)|list|length }}", {}, "2"),
    "round filter": ("{{ 12345|round(-3) }}", {}, "12000"),
    "round filter on a float": ("{{ 1.5|round(100000000) }}", {}, "1.5"),
    "round filter through floor": ("{{ 42.55|round(1, 'floor') }}", {}, "42.5"),
    "a tuple a template builds": (
        "{% set t = (1, 2, 3) %}{{ t|join('-') }}",
        {},
        "1-2-3",
    ),
    "a tuple of one": ("{{ (5,)|length }}", {}, "1"),
    "slices": (
        "{% set s = 'abcdef' %}{{ s[1:4] }}{{ s[::2] }}{{ s[:i] }}",
        {"i": 2},
        "bcdaceab",
    ),
    "tuple targets in a for": (
        "{% for a, b in [(1,2),(3,4)] %}{{ a }}{{ b }}{% endfor %}",
        {},
        "1234",
    ),
    "round filter through ceil": ("{{ 5|round(-10, 'ceil') }}", {}, "10000000000.0"),
    "string filter": ("{{ [1,2]|string }}", {}, "[1, 2]"),
    "a container emitted whole": ("{{ [1,'a'] }}", {}, "[1, 'a']"),
    "a lazy join": ("{{ range(3)|map('string')|join('-') }}", {}, "0-1-2"),
    "center method": ("{{ 'x'.center(5) }}|", {}, "  x  |"),
    "zfill method": ("{{ '7'.zfill(3) }}", {}, "007"),
    "expandtabs method": ("{{ 'a\tb'.expandtabs(4) }}", {}, "a   b"),
    "join method": ("{{ '-'.join(['a','b']) }}", {}, "a-b"),
    "replace method": ("{{ 'aaa'.replace('a','bb') }}", {}, "bbbbbb"),
    "translate method": ("{{ 'abc'.translate({97: 'A'}) }}", {}, "Abc"),
    "list append": (
        (
            "{% set l = [] %}{% for i in range(3) %}{{ l.append(i) or '' }}"
            "{% endfor %}{{ l|join }}"
        ),
        {},
        "012",
    ),
    "tilde concatenation": ("{{ 'a' ~ 1 ~ 'b' }}", {}, "a1b"),
    "plus concatenation": ("{{ 1 + 2 }}", {}, "3"),
    "macro": ("{% macro g(x) %}[{{ x }}]{% endmacro %}{{ g('v') }}", {}, "[v]"),
    "a precision that truncates a big value": (
        "{% set a = 'x'|center(30000000) %}{{ ('%.5s' % a)|length }}",
        {},
        "5",
    ),
    "a megabyte through a filter chain": (
        "{{ (v|trim|upper|lower|trim)|length }}",
        {"v": "z" * 2_000_000},
        "2000000",
    ),
    "a long %-style format string": (
        "{% set f = '%s' * 5000 %}{% set f = f ~ f %}{{ (f % ((1,) * 10000))|length }}",
        {},
        "10000",
    ),
    "a format string with thousands of fields": (
        "{{ f % v }}",
        {"f": "%s" * 5_000, "v": ("",) * 5_000},
        "",
    ),
    "a bound format method called many times": (
        (
            "{% set s = '{:.1}' %}{% set g = s.format %}"
            "{% for i in range(20) %}{{ g(v) }}{% endfor %}"
        ),
        {"v": "x" * 3_000},
        "x" * 20,
    ),
    "a filter that hands its value back": (
        "{{ v|trim }}",
        {"v": "x" * 3_000},
        "x" * 3_000,
    ),
    "a constant tuple in a loop": (
        f"{{% for i in range(10000) %}}{{% set t = {_CONST_TUPLE} %}}{{% endfor %}}ok",
        {},
        "ok",
    ),
    "ten thousand tags": ("{{ v }}" * 10_000, {"v": ""}, ""),
    "a ten-thousand-character tag": (
        '{{ "' + "a" * 9_900 + '"|length }}',
        {},
        "9900",
    ),
}


@pytest.mark.parametrize(
    ("template", "values", "expected"),
    WORKING_TEMPLATES.values(),
    ids=WORKING_TEMPLATES,
)
@pytest.mark.asyncio
async def test_a_template_within_the_limits_still_renders(
    fmt: TextFormatter, template: str, values: dict, expected: str
):
    assert await fmt.format_string(template, values) == expected


# Under autoescape these escape tens of megabytes, which the old in-process
# budget refused and the worker renders
ESCAPING_TEMPLATES = {
    "a markup separator": (
        "{{ (l|join(''|safe))|length }}",
        {"l": ["&" * 11_000_000]},
        "55000000",
    ),
    "markup's join method": (
        "{{ ('x'|safe).join(l)|length }}",
        {"l": ["&" * 11_000_000]},
        "55000000",
    ),
    "a lazy join with a markup item": (
        "{{ ([m] + l)|select|join|length }}",
        {"l": ["&" * 11_000_000], "m": Markup("x")},
        "55000001",
    ),
    "a join": ("{{ (l|join)|length }}", {"l": ["&" * 12_000_000]}, "12000000"),
    "a lazy join": (
        "{{ (l|select|join)|length }}",
        {"l": ["&" * 12_000_000]},
        "12000000",
    ),
    "a replace that does not escape": (
        "{{ v|replace('&', '') }}",
        {"v": "&" * 12_000_000},
        "",
    ),
    "a markup replacement": (
        "{{ v|replace('x', m)|length }}",
        {"v": "x", "m": Markup("&" * 11_000_000)},
        "11000000",
    ),
    "a markup value": (
        "{{ v|safe|replace('&'|safe, '') }}",
        {"v": "&" * 11_000_000},
        "",
    ),
    "a replace on escaped text": (
        "{{ (v|replace('amp'|safe, 'y'))|length }}",
        {"v": "&" * 12_500_000},
        "37500000",
    ),
}


@pytest.mark.parametrize(
    ("template", "values", "expected"),
    ESCAPING_TEMPLATES.values(),
    ids=ESCAPING_TEMPLATES,
)
@pytest.mark.asyncio
async def test_a_template_that_escapes_megabytes_still_renders(
    template: str, values: dict, expected: str
):
    formatter = TextFormatter(autoescape=True)
    assert await formatter.format_string(template, values) == expected


@pytest.mark.parametrize("autoescape", [False, True])
@pytest.mark.parametrize(
    "template",
    [
        "{{ 123|replace(1, 2) }}",
        "{{ 'a1b1'|replace(1, 'x', 1) }}",
        "{{ 'a&b'|replace('&'|safe, '<') }}",
        "{{ ('a&b'|safe)|replace('&', '<') }}",
    ],
)
@pytest.mark.asyncio
async def test_the_replace_filter_answers_as_jinja_does(
    autoescape: bool, template: str
):
    plain = jinja2.Environment(autoescape=autoescape).from_string(template).render()
    assert await TextFormatter(autoescape=autoescape).format_string(template) == plain


@pytest.mark.asyncio
async def test_an_escaped_value_counts_towards_the_output_cap():
    # Escaping is what reaches the output, and it expands "&" fivefold, so a
    # value under the cap before it is escaped need not be under it after
    value = "&" * 12_500_000
    assert len(value) < MAX_RENDER_OUTPUT
    with pytest.raises(ValueError, match="output too large"):
        await TextFormatter(autoescape=True).format_string("{{ v }}", {"v": value})


@pytest.mark.asyncio
async def test_the_output_cap_is_exact(fmt: TextFormatter):
    at_cap = "{{ 'x'|center(" + str(MAX_RENDER_OUTPUT) + ") }}"
    assert len(await fmt.format_string(at_cap)) == MAX_RENDER_OUTPUT
    with pytest.raises(ValueError, match="output too large"):
        await fmt.format_string("{{ 'x'|center(" + str(MAX_RENDER_OUTPUT + 1) + ") }}")


@LINUX_ONLY
@pytest.mark.parametrize(
    "oversized",
    ["tilde concatenation", "a loop's literal text", "center filter"],
)
@pytest.mark.asyncio
async def test_the_limits_hold_for_an_autoescaping_formatter(oversized: str):
    # `TextFormatter()` escapes by default, and Jinja joins and emits through
    # a different pair of functions when it does
    escaping = TextFormatter()
    assert (
        await escaping.format_string("a<b{{ c }}{{ d ~ e }}", {"c": "<i>", "d": "&"})
        == "a<b&lt;i&gt;&amp;"
    )
    with pytest.raises(ValueError, match=REFUSED):
        await escaping.format_string(OVERSIZED_TEMPLATES[oversized], timeout=3)


@LINUX_ONLY
@pytest.mark.asyncio
async def test_the_memory_limit_binds_the_worker_and_not_the_caller(
    fmt: TextFormatter,
):
    # Two gigabytes, far past the worker's limit: it is refused there, and
    # this process never holds any of it
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with pytest.raises(ValueError, match="out of memory"):
        await fmt.format_string("{{ 'x'|center(" + HUGE + ")|length }}")
    grown_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak_before
    assert grown_kb < RENDER_MEMORY_LIMIT // 1024
    # Most of the limit still renders, so the limit is the worker's own
    assert await fmt.format_string(MOST_OF_THE_LIMIT + "{{ a|length }}") == (
        "400000000"
    )


@LINUX_ONLY
@pytest.mark.asyncio
async def test_a_refused_render_leaves_rendering_working(fmt: TextFormatter):
    with pytest.raises(ValueError, match="out of memory"):
        await fmt.format_string(OVERSIZED_TEMPLATES["center filter"])
    assert await fmt.format_string("{{ v }}!", {"v": "still here"}) == "still here!"


@LINUX_ONLY
@pytest.mark.asyncio
async def test_a_render_that_fills_the_heap_leaves_the_next_its_whole_limit(
    fmt: TextFormatter,
):
    # A hundred thousand small strings, which the allocator keeps once freed,
    # so the next render on that worker would start most of its limit down
    with pytest.raises(ValueError, match="out of memory"):
        await fmt.format_string(OVERSIZED_TEMPLATES["a slice taken every iteration"])
    big = "{{ (" + _X + "|center(400000000))|length }}"
    assert await fmt.format_string(big) == "400000000"


@pytest.mark.parametrize("method", ["ceil", "floor"])
@pytest.mark.asyncio
async def test_a_round_whose_power_of_ten_underflows_is_refused(
    fmt: TextFormatter, method: str
):
    # Jinja divides by that power of ten, which is zero
    with pytest.raises(ValueError, match="division"):
        await fmt.format_string("{{ 5|round(-1000, '" + method + "') }}")


@pytest.mark.parametrize(
    ("template", "refusal"),
    [
        ("{{ ", "end of template"),
        ("{{ 1 / 0 }}", "division by zero"),
        ("{{ " + "1+" * 2_000 + "1 }}", "recursion"),
    ],
    ids=["a syntax error", "a division by zero", "an expression too deep to parse"],
)
@pytest.mark.asyncio
async def test_a_template_error_is_the_callers(
    fmt: TextFormatter, template: str, refusal: str
):
    with pytest.raises(ValueError, match=refusal):
        await fmt.format_string(template)


@pytest.mark.asyncio
async def test_a_builtin_error_keeps_its_type(fmt: TextFormatter):
    # As before the worker: what is not a template refusal is not a 400
    with pytest.raises(TypeError, match="unsupported operand"):
        await fmt.format_string("{{ 1 + 'a' }}")


def test_an_error_raised_while_handling_a_memory_error_is_out_of_memory():
    # What Jinja raises when it cannot rewrite a MemoryError's traceback
    try:
        try:
            raise MemoryError
        except MemoryError:
            raise SystemError("error return without exception set")
    except SystemError as handled:
        assert text._ran_out_of_memory(handled)
    assert not text._ran_out_of_memory(SystemError("an interpreter error"))


@pytest.mark.asyncio
async def test_a_builtin_error_its_type_cannot_rebuild_is_a_runtime_error(
    fmt: TextFormatter,
):
    # UnicodeEncodeError takes five arguments, and the worker sends one
    with pytest.raises(RuntimeError, match="UnicodeEncodeError"):
        await fmt.format_string("{{ v.encode('utf-8') }}", {"v": "\udcff"})


@pytest.mark.asyncio
async def test_a_render_interrupted_mid_answer_leaves_no_answer_behind(
    fmt: TextFormatter, monkeypatch: pytest.MonkeyPatch
):
    await fmt.format_string("warm")

    def interrupted(self, maxlength=None):
        raise MemoryError

    # The first answer stays unread on its worker's socket
    with monkeypatch.context() as patched:
        patched.setattr(text.Connection, "recv_bytes", interrupted)
        with pytest.raises(MemoryError):
            await fmt.format_string("first")
    assert await fmt.format_string("second") == "second"


@pytest.mark.asyncio
async def test_renders_waiting_for_a_worker_leave_the_shared_executor_free(
    fmt: TextFormatter,
):
    # More waiting renders than the loop's default executor has threads
    waiting = (os.cpu_count() or 1) + 4 + MAX_RENDER_WORKERS
    renders = [
        asyncio.create_task(fmt.format_string(NESTED_LOOPS, timeout=1))
        for _ in range(waiting)
    ]
    await asyncio.sleep(0.2)
    try:
        await asyncio.wait_for(asyncio.to_thread(int), timeout=0.5)
    finally:
        for render in renders:
            render.cancel()
        await asyncio.gather(*renders, return_exceptions=True)
    assert await fmt.format_string("{{ 2 }}") == "2"


@pytest.mark.asyncio
async def test_markup_and_nested_values_reach_the_worker_intact():
    values = {"m": Markup("<b>x</b>"), "d": {"l": [1, {"k": "<"}]}, "t": (1, 2)}
    rendered = await TextFormatter().format_string(
        "{{ m }}{{ d.l[1].k }}{{ d.l[0] + t[1] }}", values
    )
    assert rendered == "<b>x</b>&lt;3"


@pytest.mark.asyncio
async def test_an_attribute_pin_value_reaches_the_worker(fmt: TextFormatter):
    # What a graph's `_@_` pin builds, which the render pickles to its worker
    value = MockObject(attr="x")
    assert await fmt.format_string("{{ v.attr }}|{{ v.missing }}", {"v": value}) == (
        "x|None"
    )


@pytest.mark.asyncio
async def test_a_value_that_cannot_reach_the_worker_is_the_callers(
    fmt: TextFormatter,
):
    with pytest.raises(ValueError, match="plain data"):
        await fmt.format_string("{{ v }}", {"v": lambda: 1})


@pytest.mark.asyncio
async def test_concurrent_renders_each_get_their_own_answer(fmt: TextFormatter):
    # More renders than workers, so some wait for one to come free
    answers = await asyncio.gather(
        *(fmt.format_string("{{ i * 3 }}", {"i": i}) for i in range(24))
    )
    assert answers == [str(i * 3) for i in range(24)]


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("hello", "hello"),
        # Jinja's lexer splits on \r\n, \r and \n, drops one trailing
        # newline of any of the three and rejoins on \n
        ("hello\n", "hello"),
        ("hello\n\n", "hello\n"),
        ("hello\r\n", "hello"),
        ("hello\r", "hello"),
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("a\r\nb\r\n", "a\nb"),
        ("\r\n", ""),
        ("a\n\r\n", "a\n"),
        ("a\r\r\nb", "a\n\nb"),
        ("<b>&", "<b>&"),
        ("{ not a tag }", "{ not a tag }"),
    ],
)
@pytest.mark.parametrize("autoescape", [False, True])
@pytest.mark.asyncio
async def test_a_template_with_no_tag_comes_back_as_it_came(
    template: str, expected: str, autoescape: bool
):
    assert await TextFormatter(autoescape).format_string(template) == expected


@pytest.mark.asyncio
async def test_a_long_prompt_renders_wherever_its_tag_sits(fmt: TextFormatter):
    # The caller `llm.py` renders a prompt before the model call can compress
    # it, and a model window reaches millions of characters
    for prompt in ("y" * 4_000_000 + "{{ v }}", "{{ v }}" + "y" * 4_000_000):
        assert len(await fmt.format_string(prompt, {"v": "z"})) == 4_000_001


# --- The deadline is met, and the event loop is free while a render runs ---

# Four nested loops are 10^16 iterations in 156 bytes of template: more than
# any deadline allows, and no single operation of it is over a limit.
NESTED_LOOPS = (
    "{% for a in range(10000) %}{% for b in range(10000) %}"
    "{% for c in range(10000) %}{% for d in range(10000) %}"
    "{% endfor %}{% endfor %}{% endfor %}{% endfor %}"
)
# The same with no call inside the loops, so nothing but the loop itself runs
NESTED_LOOPS_OVER_A_LIST = (
    "{% set s = range(10000)|list %}"
    "{% for a in s %}{% for b in s %}{% for c in s %}{% for d in s %}"
    "{% endfor %}{% endfor %}{% endfor %}{% endfor %}"
)
# A filter Jinja applies item by item runs inside one loop of its own
ITEM_BY_ITEM = (
    "{% set l = [1] * 10000 %}"
    + "{{ l.extend(l) or '' }}" * 10
    + "{{ l|map('abs')|map('abs')|map('abs')|map('abs')|map('abs')|list|length }}"
)
# A thousand tags of forty concatenations each, slow to compile
SLOW_TO_COMPILE = ("{{ " + " ~ ".join(["'a'"] * 40) + " }}") * 1_000


@pytest.mark.parametrize(
    "template", [NESTED_LOOPS, NESTED_LOOPS_OVER_A_LIST, ITEM_BY_ITEM]
)
@pytest.mark.asyncio
async def test_a_render_past_its_deadline_is_refused(fmt: TextFormatter, template: str):
    with pytest.raises(ValueError, match="timed out"):
        await fmt.format_string(template, timeout=0.5)


@LINUX_ONLY
@pytest.mark.asyncio
async def test_a_render_past_its_deadline_is_killed(fmt: TextFormatter):
    # A worker waits idle before the render starts, so the first sample sees
    # the one it takes even if this loop then pauses until the kill
    await fmt.format_string("warm")
    seen = _render_workers()
    render = asyncio.create_task(fmt.format_string(NESTED_LOOPS, timeout=2))
    while not render.done():
        seen |= _render_workers()
        await asyncio.wait([render], timeout=0.05)
    with pytest.raises(ValueError, match="timed out"):
        await render
    # The worker that ran it is gone, rather than still spinning, and the
    # next render gets a worker that answers
    assert len(seen - _render_workers()) == 1
    assert await fmt.format_string("{{ 1 + 1 }}") == "2"


@pytest.mark.parametrize("template", [NESTED_LOOPS, SLOW_TO_COMPILE])
@pytest.mark.asyncio
async def test_another_task_runs_while_a_template_renders(
    fmt: TextFormatter, template: str
):
    served = 0

    async def serve_other_requests():
        nonlocal served
        while True:
            await asyncio.sleep(0.01)
            served += 1

    other = asyncio.create_task(serve_other_requests())
    try:
        try:
            await fmt.format_string(template, timeout=0.5)
        except ValueError:
            pass
    finally:
        other.cancel()
    # Only that the render let go of the loop at all: how many turns a
    # 10 ms timer gets inside half a second is the runner's to decide
    assert served > 0


def _render_workers() -> set[int]:
    """The pids of this process's live render workers."""
    pids = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            ppid = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[1])
            command = (proc / "cmdline").read_bytes()
        except (OSError, IndexError, ValueError):
            continue
        if ppid == os.getpid() and b"_serve_renders" in command:
            pids.add(int(proc.name))
    return pids
