"""Tests for TextFormatter resource-exhaustion protections (GHSA-ppw9-h7rv-gwq9)."""

import asyncio
from time import monotonic

import pytest

from backend.util import text
from backend.util.text import (
    MAX_FORMAT_FIELD_SIZE,
    MAX_FORMAT_STRING_LENGTH,
    MAX_RENDER_BYTES,
    MAX_TEMPLATE_LENGTH,
    MAX_TEMPLATE_TAG_LENGTH,
    MAX_TEMPLATE_TAGS,
    REMOVED_FILTERS,
    RENDER_YIELD_INTERVAL,
    SIZE_CHECKED_FILTERS,
    SIZE_CHECKED_METHODS,
    TextFormatter,
    _check_percent_format,
    _render_budget,
    _RenderBudget,
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


# --- One render may build only so much, however it builds it ---

# Each template below builds a 200 MB value by a different route, so every
# check has a case of its own. The sizes are template literals on purpose: a
# test that computed them from the limits would pass whatever the limits were.
# "would exceed" is the refusal an operation gets before it builds anything,
# because its result size follows from its arguments; "rendering exceeded" is
# the one a value gets charged for after the fact.
HUGE = "200000000"
BIG_VALUE = "{% set a = 'x' * 10000 %}"
# Five megabytes, built within the budget, so what multiplies it is
# what the case is about
MEGABYTES = "{% set a = 'x'|center(5000000) %}"
BEFORE_IT_ALLOCATES = "would exceed"
AFTER_IT_IS_BUILT = "rendering exceeded"

OVERSIZED_TEMPLATES = {
    "center filter": (
        "{{ 'x'|center(" + HUGE + ")|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "indent filter": (
        "{{ 'a\nb'|indent(" + HUGE + ")|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "join filter": (
        BIG_VALUE + "{{ range(10000)|join(a ~ a)|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "join filter over big items": (
        BIG_VALUE + "{% set l = [a] * 10000 %}{{ l|join|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "replace filter": (
        BIG_VALUE + "{{ a|replace('x', a ~ a)|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "center method": (
        "{{ 'x'.center(" + HUGE + ")|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "ljust method": (
        "{{ 'x'.ljust(" + HUGE + ")|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "rjust method": (
        "{{ 'x'.rjust(" + HUGE + ")|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "zfill method": (
        "{{ '1'.zfill(" + HUGE + ")|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "expandtabs method": (
        "{{ '\t'.expandtabs(" + HUGE + ")|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "join method": (
        BIG_VALUE + "{% set l = [a] * 10000 %}{{ (a~a).join(l)|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "replace method": (
        BIG_VALUE + "{{ a.replace('x', a ~ a)|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "translate method": (
        BIG_VALUE + "{{ (a~a).translate({120: a ~ a})|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "bytes repeat": (
        "{{ ('x'.encode() * " + HUGE + ")|length }}",
        "Sequence repeat too large",
    ),
    "tilde concatenation": (
        "{% set a = 'xx' %}"
        + "{% set a = a~a~a~a~a~a~a~a~a~a %}" * 8
        + "{{ a|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "plus concatenation": (
        "{% set a = 'xx' %}"
        + "{% set a = a+a+a+a+a+a+a+a+a+a %}" * 8
        + "{{ a|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "a loop's output": (
        BIG_VALUE + "{% for i in range(10000) %}{{ a }}{{ a }}{% endfor %}",
        AFTER_IT_IS_BUILT,
    ),
    "a loop's literal text": (
        "{% for i in range(10000) %}" + "y" * 20000 + "{% endfor %}",
        AFTER_IT_IS_BUILT,
    ),
    "a loop into a set block": (
        BIG_VALUE
        + "{% set b %}{% for i in range(10000) %}{{ a }}{{ a }}{% endfor %}{% endset %}"
        + "{{ b|length }}",
        AFTER_IT_IS_BUILT,
    ),
    "a lazily mapped chain": (
        "{{ range(10000)|map('center', 1000000)|join|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "repeated squaring": (
        "{% set x = 10**1000 %}" + "{% set x = x*x %}" * 14 + "{{ x|string|length }}",
        "Integer too large",
    ),
    "list extension": (
        "{% set l = [1] * 10000 %}" + "{{ l.extend(l) }}" * 25 + "{{ l|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "batch filter's fill": (
        "{{ [1]|batch(" + HUGE + ", 'x')|list|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "slice filter's count": (
        "{{ [1]|slice(" + HUGE + ")|list|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "a container emitted whole": (
        BIG_VALUE + "{% set l = [a] * 10000 %}{{ l }}",
        AFTER_IT_IS_BUILT,
    ),
    "string filter on a container": (
        BIG_VALUE + "{% set l = [a] * 10000 %}{{ l|string|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "the % operator's arguments": (
        MEGABYTES + "{{ ('%s' * 5000) % ((a,) * 5000) }}",
        BEFORE_IT_ALLOCATES,
    ),
    "str.format's arguments": (
        MEGABYTES + "{{ ('{}' * 5000).format(*((a,) * 5000)) }}",
        BEFORE_IT_ALLOCATES,
    ),
    "a lazy join's separator": (
        MEGABYTES + "{{ range(10000)|map('string')|join(a)|length }}",
        BEFORE_IT_ALLOCATES,
    ),
    "round to a negative precision": (
        "{{ (10**100)|round(-" + HUGE + ") }}",
        "Integer too large",
    ),
    "a filter applied once per item": (
        BIG_VALUE + "{% set l = [a] * 10000 %}{{ (l|map('lower')|list)|length }}",
        AFTER_IT_IS_BUILT,
    ),
    "a filter over a caller's list": (
        "{{ (l|map('lower')|list)|length }}",
        AFTER_IT_IS_BUILT,
    ),
    "a long %-style format string": (
        "{% set f = '%s' * 5000 %}{% set f = f ~ f %}{{ (f % ()) }}",
        "Format string too long",
    ),
}


# The one case whose list arrives from the caller rather than the template
OVERSIZED_VALUES = {"a filter over a caller's list": {"l": ["x" * 10_000] * 6_000}}


@pytest.mark.parametrize(
    ("name", "template", "refusal"),
    [(name, *case) for name, case in OVERSIZED_TEMPLATES.items()],
    ids=OVERSIZED_TEMPLATES,
)
@pytest.mark.asyncio
async def test_an_oversized_render_is_refused(
    fmt: TextFormatter, name: str, template: str, refusal: str
):
    # ValueError, not OverflowError: the app's global handlers answer a
    # ValueError with 400 and anything else with 500 and a logged error, and
    # an oversized template is the caller's
    with pytest.raises(ValueError, match=refusal):
        await fmt.format_string(template, OVERSIZED_VALUES.get(name))


WORKING_TEMPLATES = {
    "substitution": ("Hello {{ name }}!", {"name": "World"}, "Hello World!"),
    "loop": ("{% for i in range(3) %}{{ i }},{% endfor %}", {}, "0,1,2,"),
    "center filter": ("{{ 'x'|center(9) }}|", {}, "    x    |"),
    "indent filter": ("{{ 'a\nb'|indent(2) }}", {}, "a\n  b"),
    "join filter": ("{{ ['a','b','c']|join('-') }}", {}, "a-b-c"),
    "replace filter": ("{{ 'aaa'|replace('a','bb') }}", {}, "bbbbbb"),
    "map filter": ("{{ ['a','b']|map('upper')|join }}", {}, "AB"),
    "select filter": ("{{ [1,2,3]|select('odd')|list|join(',') }}", {}, "1,3"),
    "urlencode filter": ("{{ 'a b'|urlencode }}", {}, "a%20b"),
    "wordwrap filter": ("{{ 'aaa bbb'|wordwrap(3) }}", {}, "aaa\nbbb"),
    "filesizeformat filter": ("{{ 1000|filesizeformat }}", {}, "1.0 kB"),
    "batch filter": ("{{ [1,2,3]|batch(2, 'x')|list|length }}", {}, "2"),
    "slice filter": ("{{ [1,2,3]|slice(2)|list|length }}", {}, "2"),
    "round filter": ("{{ 12345|round(-3) }}", {}, "12000"),
    "round filter on a float": ("{{ 1.5|round(100000000) }}", {}, "1.5"),
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
    "a megabyte through a filter chain": (
        "{{ (v|trim|upper|lower|trim)|length }}",
        {"v": "z" * 2_000_000},
        "2000000",
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


# A filter Jinja applies item by item runs inside one loop of its own, with
# no body for a checkpoint to sit at the top of
ITEM_BY_ITEM = (
    "{% set l = [1] * 10000 %}"
    + "{{ l.extend(l) or '' }}" * 7
    + "{{ l|map('abs')|map('abs')|map('abs')|map('abs')|map('abs')|list|length }}"
)


@pytest.mark.parametrize(
    "template", [NESTED_LOOPS, NESTED_LOOPS_OVER_A_LIST, ITEM_BY_ITEM]
)
@pytest.mark.asyncio
async def test_a_render_past_its_deadline_is_refused(fmt: TextFormatter, template: str):
    with pytest.raises(ValueError, match="timed out"):
        await fmt.format_string(template, timeout=0.5)


@pytest.mark.asyncio
async def test_another_task_runs_while_a_template_renders(fmt: TextFormatter):
    served = 0

    async def serve_other_requests():
        nonlocal served
        while True:
            await asyncio.sleep(0.01)
            served += 1

    other = asyncio.create_task(serve_other_requests())
    try:
        with pytest.raises(ValueError, match="timed out"):
            await fmt.format_string(NESTED_LOOPS, timeout=0.5)
    finally:
        other.cancel()
    assert served > 5


@pytest.mark.asyncio
async def test_a_render_without_a_timeout_still_meets_its_size_budget(
    fmt: TextFormatter,
):
    with pytest.raises(ValueError, match=BEFORE_IT_ALLOCATES):
        await fmt.format_string(OVERSIZED_TEMPLATES["center filter"][0], timeout=None)


@pytest.mark.asyncio
async def test_the_checkpoint_refuses_a_render_past_its_deadline(fmt: TextFormatter):
    token = _render_budget.set(_RenderBudget(MAX_RENDER_BYTES, monotonic() - 1))
    try:
        with pytest.raises(Exception) as refusal:
            await fmt.env.checkpoint()
    finally:
        _render_budget.reset(token)
    assert type(refusal.value).__name__ == "_RenderDeadlineExceeded"


@pytest.mark.asyncio
async def test_the_checkpoint_gives_the_event_loop_a_turn(fmt: TextFormatter):
    ran = False

    async def other_work():
        nonlocal ran
        ran = True

    token = _render_budget.set(_RenderBudget(MAX_RENDER_BYTES, monotonic() + 60))
    try:
        # A render that has just started is not yet due to give up the loop
        task = asyncio.create_task(other_work())
        await fmt.env.checkpoint()
        assert ran is False
        await task
        # Once it has held the loop for the interval, the next one does
        await asyncio.sleep(RENDER_YIELD_INTERVAL * 2)
        ran = False
        task = asyncio.create_task(other_work())
        await fmt.env.checkpoint()
        assert ran is True
        await task
    finally:
        _render_budget.reset(token)


# These two run against a budget small enough that a modest template reaches
# it, because the value a lazy filter or an in-place method builds is only
# charged after it exists: no argument of theirs says how big it will be.


@pytest.mark.asyncio
async def test_what_a_lazy_filter_builds_is_charged(
    fmt: TextFormatter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(text, "MAX_RENDER_BYTES", 5_000)
    with pytest.raises(ValueError, match="exceed"):
        # `list` answers a render with a coroutine, so the list of 2,000
        # one-character strings is only there once that is awaited
        await fmt.format_string("{{ v|list|length }}", {"v": "x" * 2_000})


@pytest.mark.asyncio
async def test_what_a_method_adds_to_its_own_container_is_charged(
    fmt: TextFormatter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(text, "MAX_RENDER_BYTES", 5_000)
    with pytest.raises(ValueError, match="exceed"):
        await fmt.format_string(
            "{% set l = [] %}"
            "{% for i in range(10000) %}{{ l.append(i) or '' }}{% endfor %}"
        )


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
    with pytest.raises(ValueError, match="exceed"):
        await escaping.format_string(OVERSIZED_TEMPLATES[oversized][0])


@pytest.mark.asyncio
async def test_a_template_longer_than_the_bound_is_refused(fmt: TextFormatter):
    with pytest.raises(ValueError, match="Template too long"):
        await fmt.format_string("x" * (MAX_TEMPLATE_LENGTH + 1))
    assert await fmt.format_string("x" * MAX_TEMPLATE_LENGTH) == (
        "x" * MAX_TEMPLATE_LENGTH
    )


@pytest.mark.asyncio
async def test_a_template_with_too_many_tags_is_refused(fmt: TextFormatter):
    # Compiling happens before the budget and the deadline exist, and what it
    # costs follows from the number of tags rather than the length
    with pytest.raises(ValueError, match="too many tags"):
        await fmt.format_string("{{ v }}" * (MAX_TEMPLATE_TAGS + 1), {"v": ""})
    assert await fmt.format_string("{{ v }}" * MAX_TEMPLATE_TAGS, {"v": ""}) == ""


@pytest.mark.asyncio
async def test_comments_count_towards_the_tag_cap(fmt: TextFormatter):
    # A comment is a tag the lexer still walks, and it renders to nothing, so
    # neither the tag count nor the output would otherwise see it
    with pytest.raises(ValueError, match="too many tags"):
        await fmt.format_string("{# c #}" * (MAX_TEMPLATE_TAGS + 1))
    assert await fmt.format_string("a{# c #}b") == "ab"


@pytest.mark.asyncio
async def test_a_single_tag_longer_than_the_bound_is_refused(fmt: TextFormatter):
    # The tag count says nothing about what one tag holds, and the length cap
    # is set where a model's prompt lives, so one tag could carry all of it
    one_long_tag = "{{ " + "1+" * (MAX_TEMPLATE_TAG_LENGTH // 2) + "1 }}"
    with pytest.raises(ValueError, match="tag too long"):
        await fmt.format_string(one_long_tag)
    # A tag at the bound still compiles, as long as it is not what Jinja's
    # own parser recurses on
    flat = '{{ "' + "a" * (MAX_TEMPLATE_TAG_LENGTH - 100) + '"|length }}'
    assert await fmt.format_string(flat) == str(MAX_TEMPLATE_TAG_LENGTH - 100)


@pytest.mark.asyncio
async def test_a_long_prompt_renders_wherever_its_tag_sits(fmt: TextFormatter):
    # The bound measures a tag, not the distance to the next one, so text on
    # either side of it is not what gets counted
    for prompt in ("y" * 4_000_000 + "{{ v }}", "{{ v }}" + "y" * 4_000_000):
        assert len(await fmt.format_string(prompt, {"v": "z"})) == 4_000_001


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("hello", "hello"),
        # Jinja's lexer splits on \r\n, \r and \n, drops one trailing
        # newline of any of the three and rejoins on \n; the shortcut stands
        # in for that lexer, so it has to answer the same way
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
async def test_a_template_with_no_tag_costs_no_budget(
    fmt: TextFormatter, monkeypatch: pytest.MonkeyPatch
):
    # Nothing is built and nothing is emitted: the answer is the string that
    # came in, so compiling it and charging its text would both be waste
    monkeypatch.setattr(text, "MAX_RENDER_BYTES", 5_000)
    assert await fmt.format_string("y" * 10_000) == "y" * 10_000


@pytest.mark.asyncio
async def test_a_long_prompt_with_one_tag_still_renders(fmt: TextFormatter):
    # The caller `llm.py` renders a prompt before the model call can compress
    # it, and a model window reaches millions of characters, so length alone
    # must not be what refuses one
    prompt = "y" * 4_000_000 + "{{ v }}"
    assert len(await fmt.format_string(prompt, {"v": "z"})) == 4_000_001


@pytest.mark.asyncio
async def test_a_format_string_within_the_bound_still_works(fmt: TextFormatter):
    within = "%s" * (MAX_FORMAT_STRING_LENGTH // 2)
    assert (
        await fmt.format_string(
            "{{ f % v }}", {"f": within, "v": ("",) * (MAX_FORMAT_STRING_LENGTH // 2)}
        )
        == ""
    )


@pytest.mark.asyncio
async def test_what_str_format_builds_is_charged(
    fmt: TextFormatter, monkeypatch: pytest.MonkeyPatch
):
    # The field check bounds one field; this is the result the call returns
    monkeypatch.setattr(text, "MAX_RENDER_BYTES", 5_000)
    with pytest.raises(ValueError, match="exceed"):
        await fmt.format_string("{{ '{}'.format(v) }}", {"v": "x" * 3_000})


@pytest.mark.asyncio
async def test_a_bound_format_method_is_charged_per_call(
    fmt: TextFormatter, monkeypatch: pytest.MonkeyPatch
):
    # Jinja builds one formatter per lookup, so a method bound once and called
    # many times shares one. Each call renders a single character here, and a
    # running total that spanned the calls would refuse the render for 60,000
    # characters that never exist at one time.
    monkeypatch.setattr(text, "MAX_RENDER_BYTES", 50_000)
    bound_once = (
        "{% set s = '{:.1}' %}{% set g = s.format %}"
        "{% for i in range(20) %}{{ g(v) }}{% endfor %}"
    )
    assert await fmt.format_string(bound_once, {"v": "x" * 3_000}) == "x" * 20


@pytest.mark.asyncio
async def test_a_filter_that_hands_its_value_back_costs_nothing(
    fmt: TextFormatter, monkeypatch: pytest.MonkeyPatch
):
    # `str.strip()` answers with the same object when it changes nothing, so
    # the budget pays for the value once, at the output, and not again here
    monkeypatch.setattr(text, "MAX_RENDER_BYTES", 5_000)
    assert await fmt.format_string("{{ v|trim }}", {"v": "x" * 3_000}) == "x" * 3_000


# Two tails, because the walk has two reasons not to search twice and each
# shape exercises one: with nothing to find, the first opening ends the walk;
# with a closing at the far end, the cursor skips every opening inside the tag
# that was just measured. Either way one search is all it takes.
@pytest.mark.parametrize("closing_delimiter", ["", "}}"])
@pytest.mark.asyncio
async def test_unclosed_tags_are_walked_once(
    fmt: TextFormatter, monkeypatch: pytest.MonkeyPatch, closing_delimiter: str
):
    # Searching for a closing delimiter from every opening costs one scan of
    # the tail per opening, which is quadratic in the template's size and is
    # inside every cap. Counting the searches says so without timing anything.
    searches = 0
    closing = text._JINJA_TAG_END

    class CountingSearches:
        def search(self, *args: object, **kwargs: object) -> object:
            nonlocal searches
            searches += 1
            return closing.search(*args, **kwargs)

    monkeypatch.setattr(text, "_JINJA_TAG_END", CountingSearches())
    template = "{{" * 5_000 + "y" * 100_000 + closing_delimiter
    with pytest.raises(ValueError, match="tag too long"):
        await fmt.format_string(template)
    assert searches == 1


@pytest.mark.asyncio
async def test_the_tag_walk_measures_what_it_did_before(fmt: TextFormatter):
    # The cursor must not change any answer, only the work taken to reach it
    assert text._longest_tag("") == 0
    assert text._longest_tag("no tags here") == 0
    assert text._longest_tag("{{ v }}") == 7
    assert text._longest_tag("{{ v }}{% if v %}y{% endif %}") == len("{% endif %}")
    assert text._longest_tag("a{# c #}b") == 7
    # An opening inside the tag just measured belongs to that tag
    assert text._longest_tag("{{ {{ v }}") == 10
    # An unclosed opening runs to the end
    assert text._longest_tag("ab{{ cde") == 6


@pytest.mark.asyncio
async def test_an_expression_too_deep_to_parse_is_the_callers_error(
    fmt: TextFormatter,
):
    # Jinja's parser recurses per operator, so a nested expression inside one
    # tag raises RecursionError while compiling. It is the caller's template,
    # so it answers the way every other refusal does.
    with pytest.raises(ValueError, match="recursion"):
        await fmt.format_string("{{ " + "1+" * 2_000 + "1 }}")


# --- A new way to build a value is classified, never allowed by omission ---

# Reviewed as bounded by the length of what they are given, so the budget can
# charge their result instead of working its size out beforehand. A filter
# Jinja adds, or one we add, lands in neither set and fails this test.
BOUNDED_FILTERS = frozenset(
    {
        "abs",
        "attr",
        "capitalize",
        "count",
        "d",
        "default",
        "dictsort",
        "e",
        "escape",
        "filesizeformat",
        "first",
        "float",
        "forceescape",
        "groupby",
        "int",
        "items",
        "last",
        "length",
        "list",
        "lower",
        "map",
        "max",
        "min",
        "random",
        "reject",
        "rejectattr",
        "reverse",
        "safe",
        "select",
        "selectattr",
        "sort",
        "striptags",
        "sum",
        "title",
        "trim",
        "truncate",
        "unique",
        "upper",
        "urlencode",
        "wordcount",
        "wordwrap",
    }
)

# The same for the methods a template can call on a str, bytes or bytearray
BOUNDED_METHODS = frozenset(
    {
        "append",
        "capitalize",
        "casefold",
        "clear",
        "copy",
        "count",
        "decode",
        "encode",
        "endswith",
        "find",
        "fromhex",
        "hex",
        "index",
        "insert",
        "isalnum",
        "isalpha",
        "isascii",
        "isdecimal",
        "isdigit",
        "isidentifier",
        "islower",
        "isnumeric",
        "isprintable",
        "isspace",
        "istitle",
        "isupper",
        "lower",
        "lstrip",
        "maketrans",
        "partition",
        "pop",
        "remove",
        "removeprefix",
        "removesuffix",
        "reverse",
        "rfind",
        "rindex",
        "rpartition",
        "rsplit",
        "rstrip",
        "split",
        "splitlines",
        "startswith",
        "strip",
        "swapcase",
        "title",
        "upper",
    }
)


def test_every_filter_is_classified(fmt: TextFormatter):
    assert set(fmt.env.filters) == set(SIZE_CHECKED_FILTERS | BOUNDED_FILTERS)
    assert SIZE_CHECKED_FILTERS.isdisjoint(BOUNDED_FILTERS)
    assert REMOVED_FILTERS.isdisjoint(fmt.env.filters)


def test_every_text_method_is_classified():
    callables = {
        name
        for kind in (str, bytes, bytearray)
        for name in dir(kind)
        if not name.startswith("_") and callable(getattr(kind, name))
    }
    assert callables - SIZE_CHECKED_METHODS == BOUNDED_METHODS


def test_a_loop_body_checks_the_budget_before_it_runs(fmt: TextFormatter):
    source = fmt.env.compile("{% for i in range(3) %}{{ i }}{% endfor %}", raw=True)
    assert "await environment.checkpoint()" in source[source.index("async for ") :]


def test_concatenation_goes_through_the_checked_join(fmt: TextFormatter):
    assert "environment.join_checked(" in fmt.env.compile("{{ a ~ b }}", raw=True)


def test_an_output_expression_is_left_to_the_render(fmt: TextFormatter):
    # Jinja evaluates a wholly constant output expression while compiling the
    # template, where no limit of ours can see it; the environment's finalize
    # takes the context, which is what stops that.
    assert "center" in fmt.env.compile("{{ 'x'|center(9) }}", raw=True)
