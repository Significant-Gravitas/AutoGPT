"""Helpers for injecting user-provided variables into sandboxed code.

Strategy: serialize the variables to JSON and pass them through an environment
variable (the data channel), then prepend a small *constant* snippet that
deserializes that env var into named variables inside the sandbox. No user data
ever enters the code string, so there is no code-injection surface -- the same
principle as parameterized SQL queries.
"""

import json
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


class UnsupportedLanguageError(ValueError):
    """Raised when variable injection is requested for an unsupported language."""


# Constant prefixes. They reference only the env var (data), never user values.
_PYTHON_PREFIX = (
    "import json as _agpt_json, os as _agpt_os\n"
    f'globals().update(_agpt_json.loads(_agpt_os.environ["{VARIABLES_ENV_KEY}"]))\n'
)
_JAVASCRIPT_PREFIX = (
    f"Object.assign(globalThis, JSON.parse(process.env.{VARIABLES_ENV_KEY}));\n"
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

    coerced = {key: _coerce_value(value) for key, value in variables.items()}
    envs = {VARIABLES_ENV_KEY: json.dumps(coerced)}
    return envs, prefix


def _coerce_value(value: Any) -> Any:
    """Parse a string value as JSON so types survive (42 -> int, true -> bool).

    The builder's dynamic-dict widget stores every value as a string. Trying
    json.loads recovers the intended type; values that aren't valid JSON (e.g.
    `hello`) are kept as plain strings. Non-string values are passed through.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
