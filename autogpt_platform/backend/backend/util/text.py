import asyncio
import logging
import re
import types
from collections.abc import Callable
from functools import update_wrapper
from typing import Any

from jinja2 import BaseLoader
from jinja2.exceptions import TemplateError
from jinja2.sandbox import (
    SandboxedEnvironment,
    SandboxedEscapeFormatter,
    SandboxedFormatter,
)
from markupsafe import Markup

logger = logging.getLogger(__name__)

# Resource limits for template rendering
MAX_EXPONENT = 1000  # Max allowed exponent in ** operations
MAX_RANGE = 10_000  # Max items from range()
MAX_SEQUENCE_REPEAT = 10_000  # Max length from sequence * int
TEMPLATE_RENDER_TIMEOUT = 10  # Seconds before render is killed
MAX_FORMAT_FIELD_SIZE = 1_000  # Max width or precision of a formatted field

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


def format_filter_for_jinja2(value, format_string=None):
    if not format_string:
        return value
    _check_percent_format(format_string)
    return format_string % float(value)


def _check_percent_format(format_string: str, args: Any = ()) -> None:
    """Refuse a %-style format string with a field over the size cap.

    A ``*`` width or precision takes its size from ``args``, in the order the
    % operator reads them; a size that isn't there is left for the operator
    to reject.
    """
    values = args if isinstance(args, tuple) else (args,)
    position = 0
    for key, width, precision, conversion in _percent_conversions(format_string):
        sizes: list[str | int | None] = []
        for size in (width, precision):
            if size == "*":
                # Taken even by a literal "%%", so later sizes keep their place
                size = values[position] if position < len(values) else 0
                position += 1
            sizes.append(size if isinstance(size, (str, int)) else 0)
        if conversion != "%":
            position += 1
        _check_field_size(*sizes)


def _percent_conversions(
    format_string: str,
) -> list[tuple[str | None, str, str, str]]:
    """Each %-style conversion as (key, width, precision, type).

    A "(key)" may hold balanced parentheses, as the % operator allows, so
    it's scanned by depth rather than matched up to the first ")".
    """
    conversions = []
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
        conversions.append((key, *rest.groups()))
        position = format_string.find("%", rest.end())
    return conversions


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
            loader=BaseLoader(), autoescape=autoescape, enable_async=True
        )
        self.env.globals.clear()

        # Replace range with a safe capped version
        self.env.globals["range"] = _safe_range

        # Instead of clearing all filters, just remove potentially unsafe ones
        unsafe_filters = ["pprint", "tojson", "urlize", "xmlattr"]
        for f in unsafe_filters:
            if f in self.env.filters:
                del self.env.filters[f]

        self.env.filters["format"] = format_filter_for_jinja2

    async def format_string(
        self,
        template_str: str,
        values=None,
        *,
        timeout: float | None = TEMPLATE_RENDER_TIMEOUT,
        **kwargs,
    ) -> str:
        """Render a Jinja2 template with resource limits.

        Uses Jinja2's native async rendering (``render_async``) with
        ``asyncio.wait_for`` as a defense-in-depth timeout.
        """
        try:
            template = self.env.from_string(template_str)
            coro = template.render_async(values or {}, **kwargs)
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        except TimeoutError:
            raise ValueError(
                f"Template rendering timed out after {timeout}s "
                "(expression too complex)"
            )
        except TemplateError as e:
            raise ValueError(e) from e


def _safe_range(*args: int) -> range:
    """range() replacement that caps the number of items to prevent DoS."""
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise OverflowError(f"range() too large ({len(r)} items, max {MAX_RANGE})")
    return r


class _RestrictedEnvironment(SandboxedEnvironment):
    """SandboxedEnvironment with computational complexity limits.

    Prevents resource-exhaustion attacks such as ``{{ 999999999**999999999 }}``
    or ``{{ range(999999999) | list }}`` by intercepting dangerous builtins.
    """

    # Tell Jinja2 to route these operators through call_binop()
    intercepted_binops = frozenset(["**", "*", "%"])

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
            if is_format_map:
                if kwargs:
                    raise TypeError("format_map() takes no keyword arguments")
                if len(args) != 1:
                    raise TypeError(
                        f"format_map() takes exactly one argument ({len(args)} given)"
                    )
                kwargs = args[0]
                args = ()
            return str_type(vformat(f_self, args, kwargs))

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
        return super().call(__context, __obj, *args, **kwargs)

    def call_binop(self, context, operator, left, right):
        # Intercept %-style formatting, of bytes too: a template can make them
        if operator == "%" and isinstance(left, str):
            _check_percent_format(left, right)
        if operator == "%" and isinstance(left, (bytes, bytearray)):
            _check_percent_format(left.decode("latin-1"), right)
        # Intercept the ** (power) operator
        if operator == "**":
            if isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT:
                raise OverflowError(f"Exponent too large (max {MAX_EXPONENT})")
            if isinstance(left, (int, float)) and abs(left) > MAX_EXPONENT:
                raise OverflowError(
                    f"Base too large for exponentiation (max {MAX_EXPONENT})"
                )
        # Intercept sequence repetition via * (strings, lists, tuples)
        if operator == "*":
            if isinstance(left, (str, list, tuple)) and isinstance(right, int):
                if len(left) * right > MAX_SEQUENCE_REPEAT:
                    raise OverflowError(
                        f"Sequence repeat too large (max {MAX_SEQUENCE_REPEAT} items)"
                    )
            if isinstance(right, (str, list, tuple)) and isinstance(left, int):
                if len(right) * left > MAX_SEQUENCE_REPEAT:
                    raise OverflowError(
                        f"Sequence repeat too large (max {MAX_SEQUENCE_REPEAT} items)"
                    )
        return super().call_binop(context, operator, left, right)


class _SizeCheckedFormatter(SandboxedFormatter):
    """A sandboxed str.format() that refuses a field over the size cap.

    Nested sizes such as ``{:{width}}`` are resolved before ``format_field``
    runs, so the check sees the width that would apply.
    """

    def format_field(self, value: Any, format_spec: str) -> Any:
        _check_format_spec(format_spec)
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
