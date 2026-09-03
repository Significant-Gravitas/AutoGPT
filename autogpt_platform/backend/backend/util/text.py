import asyncio
import logging
import re

from jinja2 import BaseLoader
from jinja2.exceptions import TemplateError
from jinja2.sandbox import SandboxedEnvironment

logger = logging.getLogger(__name__)

# Resource limits for template rendering
MAX_EXPONENT = 1000  # Max allowed exponent in ** operations
MAX_RANGE = 10_000  # Max items from range()
MAX_SEQUENCE_REPEAT = 10_000  # Max length from sequence * int
TEMPLATE_RENDER_TIMEOUT = 10  # Seconds before render is killed


def format_filter_for_jinja2(value, format_string=None):
    if format_string:
        return format_string % float(value)
    return value


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
    intercepted_binops = frozenset(["**", "*"])

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
