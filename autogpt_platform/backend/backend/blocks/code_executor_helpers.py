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

Payloads over `MAX_ENV_PAYLOAD_BYTES` are written to a JSON file in the
sandbox instead, with its path in `AGPT_VARIABLES_FILE`. An env var can't
carry them: Linux caps a single env string at 128 KiB (MAX_ARG_STRLEN), and
the sandbox's env is inherited by every process the user's code starts, so an
oversized one makes each `subprocess`/`pip`/`!cmd` call fail with E2BIG.
"""

import base64
import json
import keyword
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ProgrammingLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "js"
    BASH = "bash"
    R = "r"
    JAVA = "java"


# Env var carrying the base64-encoded JSON payload into the sandbox.
VARIABLES_ENV_KEY = "AGPT_VARIABLES"
# Env var naming the JSON file that carries a payload too big for the env.
VARIABLES_FILE_ENV_KEY = "AGPT_VARIABLES_FILE"
# Outside the working directory, so it isn't picked up as an output file.
VARIABLES_FILE_PATH = "/tmp/agpt/variables.json"

# Largest JSON payload sent through the env var. Base64 grows it by a third,
# which keeps the env string under Linux's 128 KiB per-string limit.
MAX_ENV_PAYLOAD_BYTES = 64 * 1024
# Largest JSON payload accepted at all; bigger data belongs in a file the code
# downloads itself.
MAX_VARIABLES_PAYLOAD_BYTES = 10 * 1024 * 1024


class UnsupportedLanguageError(ValueError):
    """Raised when variable injection is requested for an unsupported language."""


class VariableInjection(BaseModel):
    """What the sandbox needs so the code sees `variables` as named values."""

    envs: dict[str, str] = {}
    """Env vars to set for the run."""
    files: dict[str, bytes] = {}
    """Files to write into the sandbox before the run, by absolute path."""
    prefix: str = ""
    """Constant code to prepend to the user's code."""


# Constant prefixes. They read the payload from the file or env var named by
# the env (data), never from user values in the code string.
_PYTHON_PREFIX = (
    "import base64 as _agpt_b64, json as _agpt_json, os as _agpt_os\n"
    "globals().update(_agpt_json.loads("
    f'open(_agpt_os.environ["{VARIABLES_FILE_ENV_KEY}"], "rb").read() '
    f'if "{VARIABLES_FILE_ENV_KEY}" in _agpt_os.environ '
    f'else _agpt_b64.b64decode(_agpt_os.environ["{VARIABLES_ENV_KEY}"])))\n'
)
# `require` is missing in ES-module kernels, where `process.getBuiltinModule`
# (Node 22.3+) stands in for it.
_JAVASCRIPT_PREFIX = (
    "Object.assign(globalThis, JSON.parse("
    f"process.env.{VARIABLES_FILE_ENV_KEY} ? "
    "(typeof require === 'function' ? require('fs') : "
    "process.getBuiltinModule('fs'))"
    f".readFileSync(process.env.{VARIABLES_FILE_ENV_KEY}, 'utf-8') : "
    f"Buffer.from(process.env.{VARIABLES_ENV_KEY}, 'base64').toString('utf-8')));\n"
)

_PREFIX_BY_LANGUAGE = {
    ProgrammingLanguage.PYTHON: _PYTHON_PREFIX,
    ProgrammingLanguage.JAVASCRIPT: _JAVASCRIPT_PREFIX,
}


def build_variable_injection(
    variables: dict[str, Any],
    language: ProgrammingLanguage,
) -> VariableInjection:
    """Build the env vars, files and code prefix needed to expose `variables`.

    Up to `MAX_ENV_PAYLOAD_BYTES` the payload travels base64-encoded in the
    `AGPT_VARIABLES` env var, as it always has. Above that it is written to
    `VARIABLES_FILE_PATH` as plain JSON, named by `AGPT_VARIABLES_FILE`.

    Raises UnsupportedLanguageError if `language` has no injection strategy,
    and ValueError for bad names, unserializable values or a payload over
    `MAX_VARIABLES_PAYLOAD_BYTES`.
    """
    if not variables:
        return VariableInjection()

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
    if language is ProgrammingLanguage.JAVASCRIPT:
        # JSON.parse rejects the NaN/Infinity that json.dumps writes; turn them
        # into null, as JavaScript's own JSON.stringify does.
        serialized = json.dumps(json.loads(serialized, parse_constant=lambda _: None))

    serialized_bytes = serialized.encode("utf-8")
    size = len(serialized_bytes)
    if size > MAX_VARIABLES_PAYLOAD_BYTES:
        raise ValueError(
            f"Variables payload is too large ({size / 1024 / 1024:.1f} MB, "
            f"max {MAX_VARIABLES_PAYLOAD_BYTES // 1024 // 1024} MB). "
            "Put the data in a file or at a URL and read or download it from "
            "your code instead."
        )
    if size > MAX_ENV_PAYLOAD_BYTES:
        return VariableInjection(
            envs={VARIABLES_FILE_ENV_KEY: VARIABLES_FILE_PATH},
            files={VARIABLES_FILE_PATH: serialized_bytes},
            prefix=prefix,
        )
    return VariableInjection(
        envs={VARIABLES_ENV_KEY: base64.b64encode(serialized_bytes).decode("ascii")},
        prefix=prefix,
    )


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
