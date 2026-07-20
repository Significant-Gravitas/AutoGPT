"""Tests for the Execute Code block's variable-injection helper.

The helper serializes user-provided variables to JSON, passes them via an
environment variable, and prepends a constant snippet that deserializes them
into named variables inside the sandbox. Keeping user data in the env var (the
data channel) rather than the code string (the code channel) avoids code
injection -- analogous to parameterized SQL queries.
"""

import base64
import json
import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.code_executor import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ExecuteCodeBlock,
    ProgrammingLanguage,
)
from backend.blocks.code_executor_helpers import (
    MAX_VARIABLES_PAYLOAD_BYTES,
    VARIABLES_ENV_KEY,
    UnsupportedLanguageError,
    build_variable_injection,
)
from backend.executor.utils import ExecutionContext


def _b64_json(variables: dict) -> str:
    """Match build_variable_injection's own encoding for test assertions."""
    return base64.b64encode(
        json.dumps(variables, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")


def _decode_env_payload(envs: dict) -> dict:
    return json.loads(base64.b64decode(envs[VARIABLES_ENV_KEY]).decode("utf-8"))


def _execution_context() -> ExecutionContext:
    ids = {k: str(uuid.uuid4()) for k in ("user", "graph", "exec", "node", "nexec")}
    return ExecutionContext(
        user_id=ids["user"],
        graph_id=ids["graph"],
        graph_exec_id=ids["exec"],
        graph_version=1,
        node_id=ids["node"],
        node_exec_id=ids["nexec"],
    )


def _make_input(**overrides) -> ExecuteCodeBlock.Input:
    data: dict = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "code": "print('hi')",
        "language": ProgrammingLanguage.PYTHON.value,
    }
    data.update(overrides)
    return ExecuteCodeBlock.Input.model_validate(data)


async def _run(block: ExecuteCodeBlock, input_data: ExecuteCodeBlock.Input):
    return [
        item
        async for item in block.run(
            input_data,
            credentials=TEST_CREDENTIALS,
            execution_context=_execution_context(),
        )
    ]


class TestBuildVariableInjection:
    def test_empty_variables_returns_noop(self):
        """No variables -> no env var, no prepended code (don't touch anything)."""
        envs, prefix = build_variable_injection({}, ProgrammingLanguage.PYTHON)
        assert envs == {}
        assert prefix == ""

    def test_python_serializes_to_env_and_unpacks_to_globals(self):
        variables = {"x": 42, "name": "Blake", "items": [1, 2, 3]}
        envs, prefix = build_variable_injection(variables, ProgrammingLanguage.PYTHON)

        # Data travels in the env var, JSON-encoded then base64-encoded.
        assert envs == {VARIABLES_ENV_KEY: _b64_json(variables)}

        # Prefix is constant code that reads the env var as data and unpacks it.
        assert "b64decode" in prefix
        assert "json.loads" in prefix
        assert "globals().update" in prefix
        assert VARIABLES_ENV_KEY in prefix
        # Crucially: no user data is embedded in the code string.
        assert "Blake" not in prefix
        assert "42" not in prefix

    def test_javascript_serializes_to_env_and_unpacks_to_globalthis(self):
        variables = {"x": 42, "name": "Blake"}
        envs, prefix = build_variable_injection(
            variables, ProgrammingLanguage.JAVASCRIPT
        )

        assert envs == {VARIABLES_ENV_KEY: _b64_json(variables)}
        assert "base64" in prefix
        assert "JSON.parse" in prefix
        assert "Object.assign(globalThis" in prefix
        assert "process.env" in prefix
        assert "Blake" not in prefix

    def test_malicious_value_cannot_break_out_of_code_channel(self):
        """A value that looks like code stays inert: it's only ever JSON data."""
        variables = {"evil": "'); import os; os.system('rm -rf /'); ('"}
        envs, prefix = build_variable_injection(variables, ProgrammingLanguage.PYTHON)
        # The dangerous string lives only in the env payload, never in the code.
        assert "os.system" not in prefix
        assert envs[VARIABLES_ENV_KEY] == _b64_json(variables)

    @pytest.mark.parametrize(
        "language",
        [
            ProgrammingLanguage.BASH,
            ProgrammingLanguage.R,
            ProgrammingLanguage.JAVA,
        ],
    )
    def test_unsupported_languages_raise(self, language):
        with pytest.raises(UnsupportedLanguageError):
            build_variable_injection({"x": 1}, language)

    def test_non_serializable_value_raises_clear_error_with_key(self):
        with pytest.raises(ValueError, match="bad"):
            build_variable_injection(
                {"ok": 1, "bad": {1, 2, 3}}, ProgrammingLanguage.PYTHON
            )

    @pytest.mark.parametrize(
        "bad_key",
        ["my var", "2x", "for", "__builtins__", "_agpt_json"],
    )
    def test_invalid_variable_names_raise(self, bad_key):
        with pytest.raises(ValueError, match="Invalid variable name"):
            build_variable_injection({bad_key: 1}, ProgrammingLanguage.PYTHON)

    def test_oversized_payload_raises(self):
        big = {"data": "x" * (MAX_VARIABLES_PAYLOAD_BYTES + 1)}
        with pytest.raises(ValueError, match="too large"):
            build_variable_injection(big, ProgrammingLanguage.PYTHON)

    def test_lone_surrogate_is_sanitized_not_crashed_on(self):
        """A lone (unpaired) UTF-16 surrogate -- e.g. from malformed emoji data
        returned by an upstream block -- must not raise. It's a valid Python
        `str` character but not a valid standalone Unicode scalar value, so it
        gets replaced rather than passed through to whatever consumes the env
        var downstream.
        """
        variables = {"note": "Holidays \ud83c"}
        envs, _ = build_variable_injection(variables, ProgrammingLanguage.PYTHON)

        assert "\ud83c" not in _decode_env_payload(envs)["note"]
        # The payload must be safe to actually use as an env var value (it's
        # base64/ASCII regardless, but assert this explicitly since it's the
        # property that actually matters for the env-var transport).
        os.environ["TEST_AGPT_VARIABLES_SANITIZE_CHECK"] = envs[VARIABLES_ENV_KEY]
        del os.environ["TEST_AGPT_VARIABLES_SANITIZE_CHECK"]

    def test_real_emoji_survives_sanitization_untouched(self):
        """Properly-paired surrogates (real emoji) are valid Unicode and must
        not be altered by the lone-surrogate sanitization.
        """
        variables = {"note": "Holidays \U0001f385\U0001f3fb"}
        envs, _ = build_variable_injection(variables, ProgrammingLanguage.PYTHON)

        assert _decode_env_payload(envs) == variables

    def test_lone_surrogate_sanitized_inside_nested_structures(self):
        variables = {
            "items": [{"label": "Holidays \ud83c"}, "plain"],
            "meta": {"tag": "x \udfff y"},
            "nested_bad_key": {"\ud83c_key": "value"},
            "tuple_val": ("tuple \udfff",),
        }
        envs, _ = build_variable_injection(variables, ProgrammingLanguage.PYTHON)

        decoded = json.dumps(_decode_env_payload(envs))
        assert "\ud83c" not in decoded
        assert "\udfff" not in decoded

    def test_real_emoji_does_not_reappear_as_literal_escapes_in_payload(self):
        """Regression test for the gap a maintainer identified in review: a
        well-formed, validly-paired emoji must not come back out as raw
        `\\uXXXX`-style escape text anywhere in the env payload. That text
        form is what broke downstream -- E2B's transport nests this payload
        as a string value inside a JSON request body, and if the value ever
        gets re-interpreted using Python/JS *source*-literal escape rules
        (which don't recombine adjacent surrogate escapes the way a JSON
        parser does) instead of JSON's own escape rules, two lone surrogate
        codepoints come out instead of one real character, and encoding that
        to UTF-8 crashes. Base64 has no backslashes or escapes at all, so
        there's nothing left for any downstream layer to misparse.
        """
        variables = {"note": "Holidays \U0001f385\U0001f3fb"}
        envs, _ = build_variable_injection(variables, ProgrammingLanguage.PYTHON)

        payload = envs[VARIABLES_ENV_KEY]
        assert "\\u" not in payload
        # Base64 alphabet only -- safe no matter how a downstream layer
        # re-serializes or re-parses this value.
        assert all(c.isalnum() or c in "+/=" for c in payload)
        assert _decode_env_payload(envs) == variables


class TestExecuteCodeBlockRun:
    """run() should inject variables: prefix the code and pass the env var."""

    async def test_run_prefixes_python_code_and_passes_envs(self):
        block = ExecuteCodeBlock()
        mock = AsyncMock(return_value=([], "", "", "", "sandbox_id", []))
        with patch.object(block, "execute_code", mock):
            await _run(
                block, _make_input(code="print(name)", variables={"name": "blake"})
            )

        kwargs = mock.call_args.kwargs
        # The user's code is prefixed with the deserialize snippet.
        assert kwargs["code"].endswith("print(name)")
        assert "globals().update" in kwargs["code"]
        # Variables travel via the env var, JSON-encoded then base64-encoded.
        assert kwargs["envs"] == {VARIABLES_ENV_KEY: _b64_json({"name": "blake"})}

    async def test_run_prefixes_javascript_code_and_passes_envs(self):
        block = ExecuteCodeBlock()
        mock = AsyncMock(return_value=([], "", "", "", "sandbox_id", []))
        with patch.object(block, "execute_code", mock):
            await _run(
                block,
                _make_input(
                    code="console.log(name)",
                    language=ProgrammingLanguage.JAVASCRIPT.value,
                    variables={"name": "blake"},
                ),
            )

        kwargs = mock.call_args.kwargs
        assert kwargs["code"].endswith("console.log(name)")
        assert "Object.assign(globalThis" in kwargs["code"]
        assert kwargs["envs"] == {VARIABLES_ENV_KEY: _b64_json({"name": "blake"})}

    async def test_run_without_variables_sends_no_envs_and_unmodified_code(self):
        block = ExecuteCodeBlock()
        mock = AsyncMock(return_value=([], "", "", "", "sandbox_id", []))
        with patch.object(block, "execute_code", mock):
            await _run(block, _make_input(code="print('hi')"))

        kwargs = mock.call_args.kwargs
        assert kwargs["code"] == "print('hi')"
        assert kwargs["envs"] == {}

    async def test_run_yields_all_outputs_when_present(self):
        block = ExecuteCodeBlock()
        mock = AsyncMock(
            return_value=([], "42", "stdout text", "stderr text", "sandbox_id", [])
        )
        # process_execution_results parses E2B-specific result objects; patch it so
        # this test only exercises run()'s own output-forwarding branches.
        with patch.object(block, "execute_code", mock), patch.object(
            block, "process_execution_results", return_value=({"text": "42"}, [])
        ):
            outputs = dict(
                await _run(
                    block,
                    _make_input(code="print(name)", variables={"name": "blake"}),
                )
            )

        assert outputs["main_result"] == {"text": "42"}
        assert outputs["response"] == "42"
        assert outputs["stdout_logs"] == "stdout text"
        assert outputs["stderr_logs"] == "stderr text"
        assert outputs["files"] == []

    async def test_run_unsupported_language_with_variables_yields_error(self):
        block = ExecuteCodeBlock()
        mock = AsyncMock()
        with patch.object(block, "execute_code", mock):
            outputs = await _run(
                block,
                _make_input(
                    code="echo hi",
                    language=ProgrammingLanguage.BASH.value,
                    variables={"name": "blake"},
                ),
            )

        assert any(name == "error" for name, _ in outputs)
        mock.assert_not_called()
