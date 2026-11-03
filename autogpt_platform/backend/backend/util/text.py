import asyncio
import inspect
import logging
import re
import types
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Collection,
    Iterable,
    Iterator,
)
from contextvars import ContextVar
from functools import update_wrapper
from time import monotonic
from typing import Any

from jinja2 import BaseLoader, pass_context
from jinja2.async_utils import auto_aiter
from jinja2.compiler import CodeGenerator, CompilerExit, Frame
from jinja2.exceptions import TemplateError
from jinja2.filters import make_attrgetter
from jinja2.nodes import Concat, Const, Dict, List, Node, Tuple
from jinja2.runtime import Context
from jinja2.sandbox import (
    SandboxedEnvironment,
    SandboxedEscapeFormatter,
    SandboxedFormatter,
)
from jinja2.utils import _PassArg
from markupsafe import Markup, escape, soft_str

logger = logging.getLogger(__name__)

# Resource limits for template rendering
MAX_EXPONENT = 1000  # Max allowed exponent in ** operations
MAX_RANGE = 10_000  # Max items from range()
MAX_SEQUENCE_REPEAT = 10_000  # Max length from sequence * int
TEMPLATE_RENDER_TIMEOUT = 10  # Seconds before render is killed
MAX_FORMAT_FIELD_SIZE = 1_000  # Max width or precision of a formatted field
MAX_TEMPLATE_LENGTH = 8_000_000  # Max characters of template one render may compile
MAX_TEMPLATE_TAGS = 10_000  # Max tags in one template
MAX_TEMPLATE_TAG_LENGTH = 10_000  # Max characters of any one tag
MAX_FORMAT_STRING_LENGTH = 10_000  # Max characters of one %-style format string
MAX_RENDER_BYTES = 50_000_000  # Max bytes of values one render may build
MAX_RENDER_INT_BITS = 100_000  # Widest integer a template may compute
RENDER_YIELD_INTERVAL = 0.005  # Max seconds a render may hold the event loop

# Filters a template may not reach at all
REMOVED_FILTERS = frozenset({"pprint", "tojson", "urlize", "xmlattr"})

# The rest of one %-style conversion after its optional "(key)": flags, a
# width and a precision that are digits or "*", a length modifier and the
# conversion type, e.g. "-12.4f" -> ("12", "4", "f"); "*d" -> ("*", "", "d")
_PERCENT_CONVERSION = re.compile(r"[-+ #0]*(\*|\d*)(?:\.(\*|\d*))?[hlL]?([\s\S])")
# The size fields of one str.format() spec, with any fill character, e.g.
# "<12.4f" -> ("12", "4"). A type's own spec language (a datetime's "%Y")
# doesn't match and isn't checked.
_FORMAT_SPEC_SIZES = re.compile(
    r"^(?:[\s\S]?[<>=^])?[-+ ]?z?#?0?(\d*)[,_]?(?:\.(\d+))?[a-zA-Z%]?$"
)
# What makes a string a template at all: an expression, a block or a comment
_JINJA_TAG = re.compile(r"\{\{|\{%|\{#")
_JINJA_TAG_END = re.compile(r"\}\}|%\}|#\}")
# Jinja's lexer splits on these and rejoins on "\n" (jinja2/lexer.py), so a
# template that skips the lexer has to normalise them itself
_NEWLINE = re.compile(r"\r\n|\r")


def format_filter_for_jinja2(value, format_string=None):
    if not format_string:
        return value
    _check_percent_format(format_string)
    return format_string % float(value)


def _check_percent_format(
    format_string: str, args: Any = (), budget: "_RenderBudget | None" = None
) -> None:
    """Refuse a %-style format string with a field over the size cap.

    A ``*`` width or precision takes its size from ``args``, in the order the
    % operator reads them; a size that isn't there is left for the operator
    to reject. Given a budget, the text every conversion would render is
    checked as well, because the operator builds all of it in one step.
    """
    # The walk below is one uninterruptible step, so the string it walks is
    # bounded first; every format string a template writes is short
    if len(format_string) > MAX_FORMAT_STRING_LENGTH:
        raise OverflowError(
            f"Format string too long (max {MAX_FORMAT_STRING_LENGTH} characters)"
        )
    values = args if isinstance(args, tuple) else (args,)
    mapping = args if isinstance(args, dict) else None
    position = 0
    rendered = len(format_string)
    for key, width, precision, conversion in _percent_conversions(format_string):
        resolved: list[Any] = []
        for size in (width, precision):
            if size == "*":
                # Taken even by a literal "%%", so later sizes keep their place
                size = values[position] if position < len(values) else 0
                position += 1
            resolved.append(size)
        sizes = [s if isinstance(s, (str, int)) else 0 for s in resolved]
        if conversion != "%":
            if mapping is not None:
                # Without a "(key)" the whole mapping is what renders
                size = _text_size(mapping.get(key) if key is not None else mapping)
            else:
                size = _text_size(values[position]) if position < len(values) else 0
            rendered += _truncated_size(size, resolved[1], conversion)
            position += 1
        _check_field_size(*sizes)
    if budget is not None:
        budget.check(rendered)


def _truncated_size(size: int, precision: Any, conversion: str) -> int:
    """What a conversion renders, where its precision truncates the text.

    Only the text conversions truncate; for a numeric one the precision
    says how much it renders, not how little.
    """
    if conversion not in ("s", "r", "a"):
        return size
    if isinstance(precision, str) and precision.isdigit():
        return min(size, int(precision))
    if isinstance(precision, int) and not isinstance(precision, bool):
        return min(size, abs(precision))
    return size


def _percent_conversions(
    format_string: str,
) -> Iterator[tuple[str | None, str, str, str]]:
    """Each %-style conversion as (key, width, precision, type).

    A "(key)" may hold balanced parentheses, as the % operator allows, so
    it's scanned by depth rather than matched up to the first ")". Yielded
    one at a time, so a refusal costs one conversion and not the whole list.
    """
    position = format_string.find("%")
    while position != -1:
        position += 1
        key = None
        if format_string.startswith("(", position):
            depth, start = 1, position + 1
            position = start
            while position < len(format_string) and depth:
                depth += {"(": 1, ")": -1}.get(format_string[position], 0)
                position += 1
            key = format_string[start : position - 1]
        rest = _PERCENT_CONVERSION.match(format_string, position)
        if rest is None:
            break
        width, precision, conversion = rest.groups()
        yield key, width, precision, conversion
        position = format_string.find("%", rest.end())


def _check_format_spec(format_spec: str) -> None:
    """Refuse a str.format() spec with a field over the size cap."""
    sizes = _FORMAT_SPEC_SIZES.match(format_spec)
    if sizes:
        _check_field_size(sizes.group(1), sizes.group(2))


def _check_field_size(width: str | int | None, precision: str | int | None) -> None:
    # The width and precision decide how much formatting allocates
    if max(abs(int(width or 0)), abs(int(precision or 0))) > MAX_FORMAT_FIELD_SIZE:
        raise OverflowError(
            f"Format field too large (max {MAX_FORMAT_FIELD_SIZE} characters)"
        )


class TextFormatter:
    def __init__(self, autoescape: bool = True):
        self.env = _RestrictedEnvironment(
            loader=BaseLoader(),
            autoescape=autoescape,
            enable_async=True,
            # Taking the context in `finalize` is what leaves an output
            # expression to the render instead of to compilation
            finalize=_charge_rendered_value,
        )
        self.env.globals.clear()

        # Replace range with a safe capped version
        self.env.globals["range"] = _safe_range

        # Instead of clearing all filters, just remove potentially unsafe ones
        for f in REMOVED_FILTERS:
            if f in self.env.filters:
                del self.env.filters[f]

        self.env.filters["format"] = format_filter_for_jinja2
        for name in list(self.env.filters):
            self.env.filters[name] = _budgeted_filter(name, self.env.filters[name])

    async def format_string(
        self,
        template_str: str,
        values=None,
        *,
        timeout: float | None = TEMPLATE_RENDER_TIMEOUT,
        **kwargs,
    ) -> str:
        """Render a Jinja2 template within one budget of size and time.

        The budget is spent inside the render, at every operation that builds
        a value and at the top of every loop, macro and block body, which is
        where the render also gives the event loop a turn;
        ``asyncio.wait_for`` stays as a backstop. Compiling the template
        reaches none of those checks, so it answers to bounds of its own: a
        template with no tag is returned as Jinja's lexer would have left
        it, and one that has tags is bounded by how many there are, how long
        the longest runs, and the length of the whole. Within those bounds
        it still runs for seconds, so it runs in a thread, inside the same
        deadline and leaving the event loop its turns.

        Raises ``ValueError`` for anything the caller's template is refused
        for, so the API answers 400 rather than 500.
        """
        if len(template_str) > MAX_TEMPLATE_LENGTH:
            raise ValueError(
                f"Template too long (max {MAX_TEMPLATE_LENGTH} characters)"
            )
        if not _JINJA_TAG.search(template_str):
            # Nothing to render, so answer with what Jinja's lexer would have
            # made of it: newlines normalised, one trailing newline dropped
            return _NEWLINE.sub("\n", template_str).removesuffix("\n")
        tags = (
            template_str.count("{{")
            + template_str.count("{%")
            + template_str.count("{#")
        )
        if tags > MAX_TEMPLATE_TAGS:
            raise ValueError(f"Template has too many tags (max {MAX_TEMPLATE_TAGS})")
        if _longest_tag(template_str) > MAX_TEMPLATE_TAG_LENGTH:
            raise ValueError(
                f"Template tag too long (max {MAX_TEMPLATE_TAG_LENGTH} characters)"
            )
        # The deadline is wall-clock, so a render that gives the loop a turn
        # is charged for the time other tasks take with it
        budget = _RenderBudget(
            MAX_RENDER_BYTES, None if timeout is None else monotonic() + timeout
        )
        token = _render_budget.set(budget)
        try:
            coro = self._compile_and_render(template_str, values or {}, kwargs)
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        except (TimeoutError, _RenderDeadlineExceeded):
            raise ValueError(
                f"Template rendering timed out after {timeout}s "
                "(expression too complex)"
            )
        except (OverflowError, RecursionError) as refused:
            raise ValueError(str(refused)) from refused
        except TemplateError as e:
            raise ValueError(e) from e
        finally:
            _render_budget.reset(token)

    async def _compile_and_render(
        self, template_str: str, values: dict, kwargs: dict
    ) -> str:
        """Compile off the event loop, then render on it.

        Lexing, parsing and code generation are one uninterruptible step
        each, so the deadline can only reach them from outside; the thread
        is what leaves the loop a turn while they run.
        """
        template = await asyncio.to_thread(self.env.from_string, template_str)
        return await template.render_async(values, **kwargs)


def _longest_tag(template_str: str) -> int:
    """How many characters the longest tag in the template runs to.

    The tag count says nothing about what one tag holds, and the length cap
    is set where a model's prompt lives, so one tag could carry all of it. A
    tag that is never closed runs to the end, which is what Jinja's lexer
    reads before it gives up, so it is measured the same way.

    One pass: the cursor skips an opening inside the tag just measured, so
    no part of the template is searched twice, and the first unclosed
    opening ends the walk. Searching from every opening instead costs one
    scan of the tail per opening.
    """
    longest = 0
    cursor = 0
    for opening in _JINJA_TAG.finditer(template_str):
        if opening.start() < cursor:
            continue
        closing = _JINJA_TAG_END.search(template_str, opening.end())
        if closing is None:
            return max(longest, len(template_str) - opening.start())
        cursor = closing.end()
        longest = max(longest, cursor - opening.start())
    return longest


def _safe_range(*args: int) -> range:
    """range() replacement that caps the number of items to prevent DoS."""
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise OverflowError(f"range() too large ({len(r)} items, max {MAX_RANGE})")
    return r


class _RenderDeadlineExceeded(Exception):
    """Raised inside a render that has run past its deadline."""


class _RenderBudget:
    """What one render may still spend: bytes of values to build, and time.

    Mutated on every template operation, so it stays a plain slotted object
    rather than a model.
    """

    __slots__ = ("deadline", "remaining", "yield_at")

    def __init__(self, remaining: int, deadline: float | None) -> None:
        self.remaining = remaining
        self.deadline = deadline
        self.yield_at = monotonic() + RENDER_YIELD_INTERVAL

    def check(self, size: int) -> None:
        """Refuse an operation before it builds a result this size."""
        if size > self.remaining:
            raise OverflowError(
                f"Template rendering would exceed {MAX_RENDER_BYTES} bytes"
            )

    def spend(self, size: int) -> None:
        self.remaining -= size
        if self.remaining < 0:
            raise OverflowError(f"Template rendering exceeded {MAX_RENDER_BYTES} bytes")

    def charge(self, result: Any, *inputs: Any) -> Any:
        """Spend what ``result`` costs, unless it IS one of its inputs.

        An operation that hands its argument back costs nothing. One that
        answers with a new value has allocated it, whatever its inputs were
        worth, which is what an operation applied once per item does.
        """
        if any(result is value for value in inputs):
            return result
        produced = _value_size(result)
        if produced > 0:
            self.spend(produced)
        return result


# Set for the whole render by `format_string`, and read by every check below.
# The object is created once and mutated in place, so the copy of the context
# that `asyncio.wait_for` hands its task counts against the same budget.
_render_budget: ContextVar[_RenderBudget | None] = ContextVar(
    "template_render_budget", default=None
)


def _current_budget() -> _RenderBudget:
    """The running render's budget, or a fresh one outside ``format_string``.

    An environment driven directly still gets every per-operation limit; only
    the total is per call then.
    """
    return _render_budget.get() or _RenderBudget(MAX_RENDER_BYTES, None)


def _value_size(value: Any) -> int:
    """About how many bytes ``value`` holds, for the budget to charge.

    A container is counted by the references it holds, not by what they
    point at, which is charged where it is built. A lazy sequence such as
    ``range`` counts for nothing: it allocates nothing either.
    """
    if isinstance(value, (str, bytes, bytearray)):
        return len(value)
    if isinstance(value, dict):
        # An entry holds both its key and its value
        return len(value) * 2 * _REFERENCE_BYTES
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) * _REFERENCE_BYTES
    if _is_plain_int(value):
        return value.bit_length() // 8
    return 0


_REFERENCE_BYTES = 8


def _text_size(value: Any, escaped: bool = False) -> int:
    """About how many characters ``value`` renders as.

    A container renders as its items' text, not as the references it holds,
    which is what `_value_size` measures, so the walk goes all the way down.
    ``escaped`` adds what autoescaping that text would grow it by.
    """
    return _walk_text_size(value, escaped, {})


def _walk_text_size(value: Any, escaped: bool, measured: dict[int, int]) -> int:
    """``_text_size`` over one structure, measuring each container once.

    Measuring a container once is what bounds the walk: the budget pays to
    build each one, while the references to it are free and can repeat. Its
    size is counted per reference, because that is what the text holds. One
    reached from inside itself counts for nothing, as ``repr`` renders it.
    """
    if isinstance(value, (str, bytes, bytearray)):
        return len(value) + (_escaped_growth(value) if escaped else 0)
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        already = measured.get(id(value))
        if already is not None:
            return already
        measured[id(value)] = 0
        total = _walk_items(value, escaped, measured) + 2
        measured[id(value)] = total
        return total
    if _is_plain_int(value):
        return value.bit_length() // 3
    return 0


def _walk_items(value: Any, escaped: bool, measured: dict[int, int]) -> int:
    """What a container's items render as, a dict's keys and values both.

    A dict is walked key by key rather than over the pairs ``items()`` makes,
    because each pair is freed as the walk moves on and the next one can be
    built at the same address, which would read as the pair just measured.

    Every item carries the separator beside it, and a string the quotes
    ``repr`` renders it in, which escaping turns into five characters each.
    """
    quoted = 12 if escaped else 4

    def rendered(item: Any) -> int:
        beside = quoted if isinstance(item, (str, bytes, bytearray)) else 2
        return _walk_text_size(item, escaped, measured) + beside

    if isinstance(value, dict):
        return sum(rendered(k) + rendered(v) for k, v in value.items())
    return sum(rendered(i) for i in value)


def _escaped_growth(value: Any) -> int:
    """How many characters escaping adds to the text, without building it."""
    if not isinstance(value, str):
        return 0
    return (
        4 * value.count("&")
        + 3 * value.count("<")
        + 3 * value.count(">")
        + 4 * value.count('"')
        + 4 * value.count("'")
    )


def _check_deadline() -> None:
    """Refuse work that has run past the render's deadline.

    `checkpoint` runs at the top of a body; a filter Jinja applies item by
    item inside one loop of its own reaches this instead.
    """
    budget = _render_budget.get()
    if (
        budget is not None
        and budget.deadline is not None
        and monotonic() > budget.deadline
    ):
        raise _RenderDeadlineExceeded()


@pass_context
def _charge_rendered_value(context: Context, value: Any) -> Any:
    """Count an expression the template renders against the budget.

    Jinja calls `str()` on what this returns and escapes it where the
    template autoescapes, so the charge is the text that reaches the output
    rather than the references the value holds.

    Taking the context is also what stops Jinja finalizing — and so
    evaluating — an output expression while it compiles the template.
    """
    escaped = context.eval_ctx.autoescape and not hasattr(value, "__html__")
    _current_budget().spend(_text_size(value, escaped))
    return value


class _BudgetedCodeGenerator(CodeGenerator):
    """Code generator that makes a render answer for its own size and time.

    Jinja's output path is left as it is except where a template can spend
    without passing any of the checks below: the top of every body gets a
    checkpoint, a literal chunk is counted as the render emits it, and ``~``
    goes through a join that is refused before it allocates.
    """

    def blockvisit(self, nodes: Iterable[Node], frame: Frame) -> None:
        # Jinja's own blockvisit with the checkpoint as the body's first
        # statement, so every loop iteration and every macro call reaches it.
        try:
            self.writeline("pass")
            if self.environment.is_async:
                self.writeline("await environment.checkpoint()")
            for node in nodes:
                self.visit(node, frame)
        except CompilerExit:
            pass

    def visit_Concat(self, node: Concat, frame: Frame) -> None:
        if frame.eval_ctx.volatile:
            join = "(markup_join if context.eval_ctx.volatile else str_join)"
        elif frame.eval_ctx.autoescape:
            join = "markup_join"
        else:
            join = "str_join"
        self.write(f"environment.join_checked({join}, (")
        for child in node.nodes:
            self.visit(child, frame)
            self.write(", ")
        self.write("))")

    def visit_List(self, node: List, frame: Frame) -> None:
        self._charged_literal(super().visit_List, node, frame)

    def visit_Dict(self, node: Dict, frame: Frame) -> None:
        self._charged_literal(super().visit_Dict, node, frame)

    def visit_Tuple(self, node: Tuple, frame: Frame) -> None:
        # A tuple being assigned to is a target, and a call around it would
        # not be one
        if node.ctx == "store":
            super().visit_Tuple(node, frame)
        else:
            self._charged_literal(super().visit_Tuple, node, frame)

    def visit_Const(self, node: Const, frame: Frame) -> None:
        # Jinja folds a list or dict of constants to one Const, and emits it
        # as a literal that the line rebuilds every time it runs
        if isinstance(node.value, (list, dict, tuple)):
            self._charged_literal(super().visit_Const, node, frame)
        else:
            super().visit_Const(node, frame)

    def _charged_literal(
        self, visit: Callable[[Any, Frame], None], node: Any, frame: Frame
    ) -> None:
        self.write("environment.charge_literal(")
        visit(node, frame)
        self.write(")")

    def _output_const_repr(self, group: Iterable[Any]) -> str:
        return f"environment.charge_data({super()._output_const_repr(group)})"


class _RestrictedEnvironment(SandboxedEnvironment):
    """SandboxedEnvironment with computational complexity limits.

    Prevents resource-exhaustion attacks such as ``{{ 999999999**999999999 }}``
    or ``{{ range(999999999) | list }}`` by intercepting dangerous builtins.
    Every other operation that builds a value is refused before it allocates
    where its size follows from the arguments, and charged to the render's
    budget where it does not; the deadline is read inside the render, at the
    seams below.
    """

    code_generator_class = _BudgetedCodeGenerator

    # Tell Jinja2 to route these operators through call_binop()
    intercepted_binops = frozenset(["**", "*", "%", "+"])

    async def checkpoint(self) -> None:
        """Enforce the deadline and give the event loop a turn.

        Emitted at the top of every generated loop, macro and block body, so
        that a render which does no I/O of its own still reaches both.
        """
        budget = _render_budget.get()
        if budget is None:
            return
        now = monotonic()
        if budget.deadline is not None and now > budget.deadline:
            raise _RenderDeadlineExceeded()
        if now >= budget.yield_at:
            budget.yield_at = now + RENDER_YIELD_INTERVAL
            await asyncio.sleep(0)

    def charge_data(self, data: str) -> str:
        """Count a literal chunk of template text the render emits."""
        _current_budget().spend(len(data))
        return data

    def charge_literal(self, value: list | dict | tuple) -> list | dict | tuple:
        """Count a list, dict or tuple the template builds from a literal.

        Jinja compiles one to a plain Python literal, so nothing else sees
        it being built, however many times a loop runs it.
        """
        _current_budget().spend(_value_size(value))
        return value

    def join_checked(
        self, join: Callable[[tuple[Any, ...]], str], pieces: tuple[Any, ...]
    ) -> str:
        """``~`` concatenation, refused before it allocates."""
        budget = _current_budget()
        budget.check(sum(map(_text_size, pieces)))
        return budget.charge(join(pieces), *pieces)

    def wrap_str_format(self, value: Any) -> Callable[..., str] | None:
        """``SandboxedEnvironment.wrap_str_format`` with formatters that
        refuse a field over the size cap, so ``str.format`` and
        ``format_map`` get the same limit as the ``format`` filter."""
        if not isinstance(
            value, (types.MethodType, types.BuiltinMethodType)
        ) or value.__name__ not in ("format", "format_map"):
            return None

        f_self: Any = value.__self__
        if not isinstance(f_self, str):
            return None

        str_type: type[str] = type(f_self)
        is_format_map = value.__name__ == "format_map"
        formatter: SandboxedFormatter
        if isinstance(f_self, Markup):
            formatter = _SizeCheckedEscapeFormatter(self, escape=f_self.escape)
        else:
            formatter = _SizeCheckedFormatter(self)
        vformat = formatter.vformat

        def wrapper(*args: Any, **kwargs: Any) -> str:
            # Jinja builds one formatter per lookup, so a method bound once
            # and called many times would otherwise share a running total
            formatter.rendered = 0
            if is_format_map:
                if kwargs:
                    raise TypeError("format_map() takes no keyword arguments")
                if len(args) != 1:
                    raise TypeError(
                        f"format_map() takes exactly one argument ({len(args)} given)"
                    )
                kwargs = args[0]
                args = ()
            budget = _current_budget()
            # vformat walks and copies the receiver, as the % operator does
            budget.check(_value_size(f_self))
            return budget.charge(str_type(vformat(f_self, args, kwargs)), f_self)

        return update_wrapper(wrapper, value)

    def call(
        __self,  # noqa: N805 – Jinja2 convention
        __context,
        __obj,
        *args,
        **kwargs,
    ):
        # Intercept pow() to cap the exponent
        if __obj is pow and len(args) >= 2:
            base, exp = args[0], args[1]
            if isinstance(exp, (int, float)) and abs(exp) > MAX_EXPONENT:
                raise OverflowError(f"Exponent too large (max {MAX_EXPONENT})")
            if isinstance(base, (int, float)) and abs(base) > MAX_EXPONENT:
                raise OverflowError(
                    f"Base too large for exponentiation (max {MAX_EXPONENT})"
                )

        _check_deadline()
        budget = _current_budget()
        owner = _method_owner(__obj)
        args = _check_method_size(owner, __obj, args, kwargs, budget) or args
        if isinstance(owner, _MUTABLE_CONTAINERS):
            # A method that grows its own container answers with None, so the
            # container itself is what has to be measured
            before = _value_size(owner)
            result = super().call(__context, __obj, *args, **kwargs)
            budget.spend(max(0, _value_size(owner) - before))
            return budget.charge(result, owner, *args)
        return budget.charge(
            super().call(__context, __obj, *args, **kwargs), owner, *args
        )

    def call_binop(self, context, operator, left, right):
        budget = _current_budget()
        # Intercept %-style formatting, of bytes too: a template can make them
        if operator == "%" and isinstance(left, str):
            _check_percent_format(left, right, budget)
        if operator == "%" and isinstance(left, (bytes, bytearray)):
            _check_percent_format(left.decode("latin-1"), right, budget)
        # Intercept the ** (power) operator
        if operator == "**":
            if isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT:
                raise OverflowError(f"Exponent too large (max {MAX_EXPONENT})")
            if isinstance(left, (int, float)) and abs(left) > MAX_EXPONENT:
                raise OverflowError(
                    f"Base too large for exponentiation (max {MAX_EXPONENT})"
                )
        # Intercept sequence repetition via * (strings, bytes, lists, tuples)
        if operator == "*":
            if isinstance(left, _SEQUENCES) and isinstance(right, int):
                if len(left) * right > MAX_SEQUENCE_REPEAT:
                    raise OverflowError(
                        f"Sequence repeat too large (max {MAX_SEQUENCE_REPEAT} items)"
                    )
            if isinstance(right, _SEQUENCES) and isinstance(left, int):
                if len(right) * left > MAX_SEQUENCE_REPEAT:
                    raise OverflowError(
                        f"Sequence repeat too large (max {MAX_SEQUENCE_REPEAT} items)"
                    )
            if (
                isinstance(left, int)
                and isinstance(right, int)
                and left.bit_length() + right.bit_length() > MAX_RENDER_INT_BITS
            ):
                raise OverflowError(
                    f"Integer too large (max {MAX_RENDER_INT_BITS} bits)"
                )
        # Concatenation builds the two sides into one value
        if operator == "+":
            budget.check(_value_size(left) + _value_size(right))
        return budget.charge(
            super().call_binop(context, operator, left, right), left, right
        )


_SEQUENCES = (str, bytes, bytearray, list, tuple)
_MUTABLE_CONTAINERS = (list, set, dict, bytearray)


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _method_owner(obj: Any) -> Any:
    """The object a bound method belongs to, or None for anything else."""
    if isinstance(obj, (types.MethodType, types.BuiltinMethodType)):
        return obj.__self__
    return None


def _check_method_size(
    owner: Any,
    obj: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
) -> tuple[Any, ...] | None:
    """Refuse a method call whose result size follows from its arguments.

    A check answers with the arguments to call it with where it had to
    replace one, as a lazy iterable has to be replaced to be bounded.
    """
    if owner is None:
        return None
    check = _METHOD_SIZE_CHECKS.get(obj.__name__)
    return check(owner, args, kwargs, budget) if check is not None else None


def _budgeted_filter(name: str, filter_func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a Jinja filter so its result is refused or charged.

    The piped value is the first argument the filter itself declares; Jinja
    puts a context, eval context or environment in front of that for the
    filters that ask for one, and the check is handed it as ``passed``.
    """
    offset = 0 if _PassArg.from_obj(filter_func) is None else 1
    check = _FILTER_SIZE_CHECKS.get(name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _check_deadline()
        own = args[offset:]
        budget = _current_budget()
        if check is not None:
            own = check(own, kwargs, budget, args[0] if offset else None) or own
            args = (*args[:offset], *own)
        result = filter_func(*args, **kwargs)
        if inspect.isawaitable(result):
            # The async variant of a filter answers with a coroutine, so the
            # value it builds is only there to charge once that is awaited
            return _charge_awaited(result, budget, own)
        return budget.charge(result, *own)

    return update_wrapper(wrapper, filter_func)


async def _charge_awaited(awaitable: Any, budget: _RenderBudget, inputs: tuple) -> Any:
    return budget.charge(await awaitable, *inputs)


def _check_padding(
    owner: Any, args: tuple[Any, ...], kwargs: dict[str, Any], budget: _RenderBudget
) -> None:
    """ljust/rjust/center/zfill: the width is the whole result."""
    budget.check(_int_arg(args, kwargs, 0, "width", 0) - len(owner))


def _check_expandtabs(
    owner: Any, args: tuple[Any, ...], kwargs: dict[str, Any], budget: _RenderBudget
) -> None:
    tabs = (
        owner.count(b"\t")
        if isinstance(owner, (bytes, bytearray))
        else owner.count("\t")
    )
    budget.check(_int_arg(args, kwargs, 0, "tabsize", 8) * tabs)


def _check_str_join(
    owner: Any, args: tuple[Any, ...], kwargs: dict[str, Any], budget: _RenderBudget
) -> tuple[Any, ...]:
    items = args[0] if args else ()
    return (_bounded_join_input(items, owner, budget), *args[1:])


def _check_str_replace(
    owner: Any, args: tuple[Any, ...], kwargs: dict[str, Any], budget: _RenderBudget
) -> None:
    if len(args) >= 2:
        _check_replace_sizes(
            owner, args[0], args[1], _int_arg(args, kwargs, 2, "count", -1), budget
        )


def _check_translate(
    owner: Any, args: tuple[Any, ...], kwargs: dict[str, Any], budget: _RenderBudget
) -> None:
    table = args[0] if args else None
    if isinstance(table, dict):
        widest = max((_value_size(v) for v in table.values()), default=1)
        budget.check(len(owner) * widest - len(owner))


def _check_container_extend(
    owner: Any, args: tuple[Any, ...], kwargs: dict[str, Any], budget: _RenderBudget
) -> None:
    """extend/update: the argument says how many items it adds."""
    budget.check(sum(_value_size(a) for a in args if isinstance(a, Collection)))


def _check_center_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> None:
    budget.check(_int_arg(own, kwargs, 1, "width", 80) - _value_size(own[0]))


def _check_indent_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> None:
    lines = str(own[0]).count("\n") + 1
    width = _own_arg(own, kwargs, 1, "width", 4)
    # The width is a count of spaces or the string to indent with; measure it
    # rather than build it, since building it is what has to be refused
    indent_size = width if _is_plain_int(width) else len(str(width))
    budget.check(indent_size * lines)


def _check_join_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> tuple[Any, ...]:
    own = _projected(own, kwargs, 2, passed.environment)
    separator = _own_arg(own, kwargs, 1, "d", "")
    return (_bounded_join_input(own[0], separator, budget), *own[1:])


def _check_string_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> None:
    """string renders its value as the text a container's items hold."""
    budget.check(_text_size(own[0]))


def _check_batch_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> None:
    """batch pads its last row with as many references as a row is wide."""
    if _own_arg(own, kwargs, 2, "fill_with", None) is not None:
        budget.check(_int_arg(own, kwargs, 1, "linecount", 0) * _REFERENCE_BYTES)


def _check_slice_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> None:
    """slice builds one list per slice, however many the template asks for."""
    budget.check(_int_arg(own, kwargs, 1, "slices", 0) * _REFERENCE_BYTES)


def _check_round_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> None:
    """``round`` raises ten to its precision: to a negative one on an integer,
    and to either sign of one on the ``ceil`` and ``floor`` paths."""
    precision = _int_arg(own, kwargs, 1, "precision", 0)
    digits = -precision if _is_plain_int(own[0]) and precision < 0 else 0
    if _own_arg(own, kwargs, 2, "method", "common") in ("ceil", "floor"):
        # Going down it is a float, and one that has underflowed is divided by
        if precision < 0 and 10.0**precision == 0:
            raise OverflowError("round() precision too small for ceil or floor")
        digits = max(digits, precision)
    if digits * _BITS_PER_DIGIT > MAX_RENDER_INT_BITS:
        raise OverflowError(f"Integer too large (max {MAX_RENDER_INT_BITS} bits)")


_BITS_PER_DIGIT = 4


def _check_replace_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> None:
    old = _own_arg(own, kwargs, 1, "old", _ABSENT)
    new = _own_arg(own, kwargs, 2, "new", _ABSENT)
    if old is _ABSENT or new is _ABSENT:
        return
    count = _own_arg(own, kwargs, 3, "count", None)
    if passed.autoescape and not hasattr(own[0], "__html__"):
        budget.check(_text_size(own[0], True))
    value, old, new = _replace_operands(passed.autoescape, own[0], old, new)
    _check_replace_sizes(value, old, new, count, budget)


_ABSENT = object()


def _replace_operands(autoescape: bool, value: Any, old: Any, new: Any) -> tuple:
    """The text, search and replacement Jinja's ``replace`` filter works on.

    It stringifies all three, and under autoescape escapes the text when
    either argument is markup, as ``do_replace`` does.
    """
    if not autoescape:
        return str(value), str(old), str(new)
    if hasattr(old, "__html__") or (
        hasattr(new, "__html__") and not hasattr(value, "__html__")
    ):
        value = escape(value)
    else:
        value = soft_str(value)
    return value, soft_str(old), soft_str(new)


def _check_wordwrap_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> None:
    """wordwrap puts its wrap string between every two lines it makes."""
    text = own[0]
    if not isinstance(text, str):
        return
    width = _own_arg(own, kwargs, 1, "width", 79)
    wrapstring = _own_arg(own, kwargs, 3, "wrapstring", None)
    wrap_size = (
        len(passed.newline_sequence) if wrapstring is None else _text_size(wrapstring)
    )
    budget.check(len(text) + _wordwrap_line_count(text, width) * wrap_size)


def _wordwrap_line_count(text: str, width: Any) -> int:
    """At most how many lines wordwrap can make of ``text``.

    Every line holds a character of the input, two lines in a row take more
    than ``width`` of it between them, and each paragraph is wrapped on its own.
    """
    lines = len(text) + 1
    if isinstance(width, (int, float)) and width >= 1:
        paragraphs = sum(map(text.count, _LINE_BREAKS)) + 1
        lines = min(lines, int(2 * len(text) / width) + paragraphs)
    return lines


# What str.splitlines splits on, so wordwrap wraps each piece on its own
_LINE_BREAKS = (
    "\n",
    "\r",
    "\v",
    "\f",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
)


def _check_sum_filter(
    own: tuple[Any, ...],
    kwargs: dict[str, Any],
    budget: _RenderBudget,
    passed: Any,
) -> tuple[Any, ...]:
    """sum adds every item onto its start, so the result holds all of them."""
    own = _projected(own, kwargs, 1, passed)
    start = _own_arg(own, kwargs, 2, "start", 0)
    return (_bounded_sum_input(own[0], _value_size(start), budget), *own[1:])


async def _bounded_sum_input(
    items: Any, total: int, budget: _RenderBudget
) -> AsyncIterator[Any]:
    async for item in auto_aiter(items):
        # Adding an integer grows the sum by a bit or so, not by the item
        if not _is_plain_int(item):
            total += _value_size(item)
            budget.check(total)
        yield item


def _projected(
    own: tuple[Any, ...], kwargs: dict[str, Any], index: int, environment: Any
) -> tuple[Any, ...]:
    """``own`` with the filter's ``attribute`` already applied to its items.

    A check then measures, and the filter is handed, what the attribute picks
    out of each item rather than the whole item.
    """
    attribute = _own_arg(own, kwargs, index, "attribute", None)
    if attribute is None:
        return own
    getter = make_attrgetter(environment, attribute)
    items = own[0]
    if isinstance(items, AsyncIterable):
        projected: Any = _amap(getter, items)
    else:
        projected = map(getter, items)
    if len(own) > index:
        own = (*own[:index], None, *own[index + 1 :])
    else:
        kwargs["attribute"] = None
    return (projected, *own[1:])


async def _amap(
    func: Callable[[Any], Any], items: AsyncIterable[Any]
) -> AsyncIterator[Any]:
    async for item in items:
        yield func(item)


def _bounded_join_input(value: Any, separator: Any, budget: _RenderBudget) -> Any:
    """A join's items, with what joining them will cost accounted for.

    A join copies every item and every separator into one string. A set of
    items of known length can be measured before the join runs; a lazy
    iterable's length is only known once it has been consumed, so its items
    are handed over one at a time against a running total instead.
    """
    if isinstance(value, AsyncIterable):
        return _bounded_aiter(value, _text_size(separator), budget)
    if not isinstance(value, Collection):
        # A sync iterable the join would consume anyway, so it can be listed
        value = budget.charge(list(value))
    separators = _text_size(separator) * len(value)
    budget.check(sum(map(_text_size, value)) + separators)
    return value


async def _bounded_aiter(
    items: AsyncIterable[Any], cost: int, budget: _RenderBudget
) -> AsyncIterator[Any]:
    total = 0
    async for item in items:
        total += cost + _text_size(item)
        budget.check(total)
        yield item


def _check_replace_sizes(
    value: Any, old: Any, new: Any, count: Any, budget: _RenderBudget
) -> None:
    if isinstance(value, Markup):
        # Markup.replace escapes its arguments before it searches or inserts
        budget.check(_text_size(old, True) + _text_size(new, True))
        old, new = escape(old), escape(new)
    occurrences = value.count(old) if old else len(value) + 1
    if _is_plain_int(count) and 0 <= count < occurrences:
        occurrences = count
    budget.check(occurrences * (_value_size(new) - _value_size(old)))


def _int_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: int
) -> int:
    value = _own_arg(args, kwargs, index, name, default)
    return value if _is_plain_int(value) else default


def _own_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: Any
) -> Any:
    """An argument a filter or method may have been given either way."""
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


# Methods whose result can be far larger than what they are called on, so the
# size is worked out from the arguments before the call allocates it. Every
# other method of a str, bytes or sequence is bounded by its own operands and
# is charged afterwards, in `call`. `format` and `format_map` never reach
# `call`: Jinja wraps them where they are looked up, and `wrap_str_format`
# above gives them `_SizeCheckedFormatter`.
_METHOD_SIZE_CHECKS: dict[str, Callable[..., Any]] = {
    "center": _check_padding,
    "expandtabs": _check_expandtabs,
    "extend": _check_container_extend,
    "join": _check_str_join,
    "ljust": _check_padding,
    "replace": _check_str_replace,
    "rjust": _check_padding,
    "translate": _check_translate,
    "update": _check_container_extend,
    "zfill": _check_padding,
}

# The same for filters. `format` checks its own format string, in
# `format_filter_for_jinja2`.
_FILTER_SIZE_CHECKS: dict[str, Callable[..., Any]] = {
    "batch": _check_batch_filter,
    "center": _check_center_filter,
    "indent": _check_indent_filter,
    "join": _check_join_filter,
    "replace": _check_replace_filter,
    "round": _check_round_filter,
    "slice": _check_slice_filter,
    "string": _check_string_filter,
    "sum": _check_sum_filter,
    "wordwrap": _check_wordwrap_filter,
}
SIZE_CHECKED_FILTERS = frozenset(_FILTER_SIZE_CHECKS) | {"format"}
SIZE_CHECKED_METHODS = frozenset(_METHOD_SIZE_CHECKS) | {"format", "format_map"}


class _SizeCheckedFormatter(SandboxedFormatter):
    """A sandboxed str.format() that refuses a field over the size cap.

    Nested sizes such as ``{:{width}}`` are resolved before ``format_field``
    runs, so the check sees the width that would apply, and the running total
    of what the fields render to is checked as they are formatted.
    """

    def __init__(self, env: Any, **kwargs: Any) -> None:
        super().__init__(env, **kwargs)
        self.rendered = 0

    def format_field(self, value: Any, format_spec: str) -> Any:
        _check_format_spec(format_spec)
        self.rendered += _text_size(value)
        _current_budget().check(self.rendered)
        return super().format_field(value, format_spec)


class _SizeCheckedEscapeFormatter(_SizeCheckedFormatter, SandboxedEscapeFormatter):
    pass


# ---------------------------------------------------------------------------
# CamelCase splitting
# ---------------------------------------------------------------------------

# Map of split forms back to their canonical compound terms.
# Mirrors the frontend exception list in frontend/src/lib/utils.ts.
_CAMELCASE_EXCEPTIONS: dict[str, str] = {
    "Auto GPT": "AutoGPT",
    "Auto Pilot": "AutoPilot",
    "Open AI": "OpenAI",
    "You Tube": "YouTube",
    "Git Hub": "GitHub",
    "Linked In": "LinkedIn",
    "Tik Tok": "TikTok",
    "All Quiet": "AllQuiet",
    "Any Search": "AnySearch",
}

_CAMELCASE_EXCEPTION_RE = re.compile(
    "|".join(re.escape(k) for k in _CAMELCASE_EXCEPTIONS),
)


def split_camelcase(text: str) -> str:
    """Split CamelCase into separate words.

    Uses a single-pass character-by-character algorithm to avoid any
    regex backtracking concerns (guaranteed O(n) time).

    After splitting, known compound terms are restored via an exception
    list (e.g. ``"YouTube"`` stays ``"YouTube"`` instead of becoming
    ``"You Tube"``).  The list mirrors the frontend mapping in
    ``frontend/src/lib/utils.ts``.

    Examples::

        >>> split_camelcase("AITextGeneratorBlock")
        'AI Text Generator Block'
        >>> split_camelcase("OAuth2Block")
        'OAuth2 Block'
        >>> split_camelcase("YouTubeBlock")
        'YouTube Block'
    """
    if len(text) <= 1:
        return text

    parts: list[str] = []
    prev = 0
    for i in range(1, len(text)):
        # Insert split between lowercase/digit and uppercase: "camelCase" -> "camel|Case"
        if (text[i - 1].islower() or text[i - 1].isdigit()) and text[i].isupper():
            parts.append(text[prev:i])
            prev = i
        # Insert split between uppercase run (2+ chars) and uppercase+lowercase:
        # "AIText" -> "AI|Text".  Requires at least 3 consecutive uppercase chars
        # before the lowercase so that the left part keeps 2+ uppercase chars
        # (mirrors the original regex r"([A-Z]{2,})([A-Z][a-z])").
        elif (
            i >= 2
            and text[i - 2].isupper()
            and text[i - 1].isupper()
            and text[i].islower()
            and (i - 1 - prev) >= 2  # left part must retain at least 2 upper chars
        ):
            parts.append(text[prev : i - 1])
            prev = i - 1

    parts.append(text[prev:])
    result = " ".join(parts)

    # Restore known compound terms that should not be split.
    result = _CAMELCASE_EXCEPTION_RE.sub(
        lambda m: _CAMELCASE_EXCEPTIONS[m.group()], result
    )
    return result
