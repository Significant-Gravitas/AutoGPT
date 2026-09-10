"""Helpers for injecting user-provided variables into sandboxed code.

Strategy: serialize the variables to JSON, base64-encode that JSON text, and
pass the result through an environment variable (the data channel), then
prepend a small *constant* snippet that base64-decodes and deserializes that
env var into named variables inside the sandbox. No user data ever enters the
code string, so there is no code-injection surface -- the same principle as
parameterized SQL queries.

The base64 step (rather than passing the JSON text directly) exists because
the env var crosses a request boundary into E2B's sandbox before it's read
back out: it travels as a string value nested inside a JSON request body, and
by the time a real astral character (e.g. an emoji) comes back out through
that path, `json.dumps`'s default `ensure_ascii=True` escaping can end up
re-interpreted as literal `\\uXXXX` Python/JS source escapes rather than JSON
escapes. Python and JS source parsers don't recombine adjacent surrogate
escapes into one supplementary character the way a JSON parser does, so a
perfectly valid, well-formed emoji can come out the other side as two lone
surrogate codepoints and crash on encode -- this is a real, reproduced
failure mode, not a hypothetical (see PR discussion). Base64 text is pure
ASCII with no backslashes or unicode escapes for any downstream text-handling
layer to misinterpret, so it closes the whole bug class rather than special-
casing individual malformed inputs.
"""

import base64
import json
import keyword
from enum import Enum
from typing import Any


class ProgrammingLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "js"
    BASH = "bash"
    R = "r"
    JAVA = "java"


# Env var name used to carry the serialized variables into the sandbox.
VARIABLES_ENV_KEY = "AGPT_VARIABLES"

# Cap the serialized payload to stay well under typical OS environment limits
# (which range from ~128 KB to a couple MB). Keeps failures clear instead of
# surfacing as cryptic sandbox startup errors.
MAX_VARIABLES_PAYLOAD_BYTES = 64 * 1024


class UnsupportedLanguageError(ValueError):
    """Raised when variable injection is requested for an unsupported language."""


# Constant prefixes. They reference only the env var (data), never user values.
_PYTHON_PREFIX = (
    "import base64 as _agpt_b64, json as _agpt_json, os as _agpt_os\n"
    "globals().update(_agpt_json.loads(_agpt_b64.b64decode("
    f'_agpt_os.environ["{VARIABLES_ENV_KEY}"]).decode("utf-8")))\n'
)
_JAVASCRIPT_PREFIX = (
    "Object.assign(globalThis, JSON.parse(Buffer.from("
    f"process.env.{VARIABLES_ENV_KEY}, 'base64').toString('utf-8')));\n"
)

_PREFIX_BY_LANGUAGE = {
    ProgrammingLanguage.PYTHON: _PYTHON_PREFIX,
    ProgrammingLanguage.JAVASCRIPT: _JAVASCRIPT_PREFIX,
}


def build_variable_injection(
    variables: dict[str, Any],
    language: ProgrammingLanguage,
) -> tuple[dict[str, str], str]:
    """Build the env vars and code prefix needed to expose `variables`.

    Returns a tuple of:
      - envs: env-var dict to pass to the sandbox (empty if no variables)
      - prefix: code to prepend so the variables exist as named variables
                (empty string if no variables)

    Raises UnsupportedLanguageError if `language` has no injection strategy.
    """
    if not variables:
        return {}, ""

    prefix = _PREFIX_BY_LANGUAGE.get(language)
    if prefix is None:
        raise UnsupportedLanguageError(
            f"Variable injection is not supported for {language.value}. "
            "Supported languages: python, js."
        )

    _validate_keys(variables)

    try:
        serialized = json.dumps(variables)
    except (TypeError, ValueError) as e:
        bad_keys = [k for k, v in variables.items() if not _is_json_serializable(v)]
        raise ValueError(
            f"Variable value is not serializable for key(s): {', '.join(bad_keys)}"
        ) from e

    serialized_bytes = serialized.encode("utf-8")
    if len(serialized_bytes) > MAX_VARIABLES_PAYLOAD_BYTES:
        raise ValueError(
            "Variables payload is too large "
            f"(max {MAX_VARIABLES_PAYLOAD_BYTES // 1024} KB). "
            "Pass large data through a file or upstream block instead."
        )

    envs = {VARIABLES_ENV_KEY: base64.b64encode(serialized_bytes).decode("ascii")}
    return envs, prefix


def _validate_keys(variables: dict[str, Any]) -> None:
    """Reject keys that can't be used as a variable name in the sandbox.

    Each key becomes a global variable, so it must be a valid identifier that
    isn't a language keyword. Dunder names are rejected because they can shadow
    builtins/internals, and `_agpt_`-prefixed names would collide with the
    deserialization snippet's own imports.
    """
    invalid = [
        key
        for key in variables
        if not key.isidentifier()
        or keyword.iskeyword(key)
        or key.startswith("__")
        or key.startswith("_agpt_")
    ]
    if invalid:
        raise ValueError(
            "Invalid variable name(s): "
            f"{', '.join(repr(k) for k in invalid)}. "
            "Names must be valid identifiers, not language keywords, and not "
            "start with '__' or '_agpt_'."
        )


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
