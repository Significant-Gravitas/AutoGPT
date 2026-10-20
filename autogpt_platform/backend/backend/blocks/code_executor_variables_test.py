"""End-to-end tests for how ExecuteCodeBlock hands `variables` to the code.

Only the sandbox SDK and output storage are replaced. The real block and
executor write files, pass env vars, and execute their prefix with real Python
and Node interpreters.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b_code_interpreter.models import Execution, Logs

from backend.blocks.code_executor import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ExecuteCodeBlock,
    ProgrammingLanguage,
)
from backend.blocks.code_executor_helpers import (
    MAX_ENV_PAYLOAD_BYTES,
    MAX_VARIABLES_PAYLOAD_BYTES,
    VARIABLES_FILE_ENV_KEY,
    build_variable_injection,
)
from backend.data.execution import ExecutionContext

PYTHON = ProgrammingLanguage.PYTHON
JAVASCRIPT = ProgrammingLanguage.JAVASCRIPT

_NODE = shutil.which("node")
_LANGUAGES = [
    PYTHON,
    pytest.param(
        JAVASCRIPT,
        marks=pytest.mark.skipif(_NODE is None, reason="node is not installed"),
    ),
]

TRICKY = {
    "quotes": "say \"hi\", it's 'fine'",
    "backslashes": "C:\\Users\\me\\new\\table \\u0041 \\\\ trailing\\",
    "newlines": "line 1\nline 2\r\n\ttabbed",
    "unicode": "🎅🏻 café 中文 Ελληνικά \u2028\u2029 \u0000",
    "json_text": json.dumps({"msg": 'he said "no"\\n', "emoji": "😀"}),
    "nested": {"list": ['"a"', "\\", None, True, 1.5, {"deep": "ünï"}]},
}

# A realistic AutoPilot payload: ~1 MB of notifications full of the same
# characters that are hard to escape.
LARGE = {
    "notifications": [
        {"id": i, "title": f'Note "{i}"', "body": "C:\\path ✓ 😀 " * 40}
        for i in range(1_800)
    ]
}


class LocalSandbox:
    """Runs what the block would send to E2B, locally."""

    def __init__(self, workdir: Path, javascript_module: bool = False):
        self.workdir = workdir
        self.javascript_module = javascript_module
        self.sandbox_id = "local"
        self.calls: list[dict] = []
        self.input_files: dict[str, bytes] = {}
        self.create = AsyncMock(return_value=self)
        self.dispose = AsyncMock()
        self.files = MagicMock(write=AsyncMock(side_effect=self.write_file))
        self.commands = MagicMock(
            run=AsyncMock(return_value=MagicMock(stdout="2026-10-19T00:00:00"))
        )

    async def write_file(self, path: str, data: bytes):
        self.input_files[path] = data
        (self.workdir / Path(path).name).write_bytes(data)

    async def run_code(self, code: str, *, language: str, envs=None, **_):
        env = {**os.environ, **(envs or {})}
        for sandbox_path in self.input_files:
            local = self.workdir / Path(sandbox_path).name
            env = {k: str(local) if v == sandbox_path else v for k, v in env.items()}
        self.calls.append({"envs": envs or {}, "files": self.input_files.copy()})
        node_args = ["--input-type=module"] if self.javascript_module else []
        cmd = (
            [sys.executable, "-c", code]
            if language == PYTHON.value
            else [str(_NODE), *node_args, "-e", code]
        )
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=20
        )
        if proc.returncode != 0:
            lines = proc.stderr.strip().splitlines()
            raise RuntimeError(next((ln for ln in lines if "Error" in ln), lines[-1]))
        return Execution(logs=Logs(stdout=[proc.stdout], stderr=[proc.stderr]))


def _dump_code(language: ProgrammingLanguage, names: list[str]) -> str:
    if language is PYTHON:
        return f"import json\nprint(json.dumps({{k: globals()[k] for k in {names!r}}}))"
    return f"console.log(JSON.stringify({{{', '.join(names)}}}))"


async def _run(
    tmp_path: Path,
    language,
    variables: dict,
    code: str | None = None,
    *,
    javascript_module: bool = False,
):
    block = ExecuteCodeBlock()
    sandbox = LocalSandbox(tmp_path, javascript_module)
    input_data = ExecuteCodeBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "language": language.value,
            "variables": variables,
            "code": code or _dump_code(language, list(variables)),
        }
    )
    context = ExecutionContext(user_id=str(uuid.uuid4()), graph_exec_id="gexec")
    with (
        patch(
            "backend.blocks.code_executor.create_sandbox",
            sandbox.create,
        ),
        patch("backend.blocks.code_executor.kill_sandbox", sandbox.dispose),
        patch(
            "backend.blocks.code_executor.extract_and_store_sandbox_files",
            AsyncMock(return_value=[]),
        ),
    ):
        outputs = dict(
            [
                item
                async for item in block.run(
                    input_data, credentials=TEST_CREDENTIALS, execution_context=context
                )
            ]
        )
    return outputs, sandbox


def _read_back(outputs: dict) -> dict:
    assert "error" not in outputs, outputs["error"]
    return json.loads(outputs["stdout_logs"])


@pytest.mark.parametrize("language", _LANGUAGES)
async def test_tricky_values_reach_the_code_unchanged(tmp_path, language):
    """Quotes, backslashes, newlines, unicode and JSON-in-a-string must come
    out exactly as they went in. (Already true since #13576 base64-encoded the
    payload; kept as a guard.)"""
    outputs, _ = await _run(tmp_path, language, TRICKY)
    assert _read_back(outputs) == TRICKY


@pytest.mark.parametrize("language", _LANGUAGES)
async def test_large_payload_reaches_the_code_unchanged(tmp_path, language):
    """A ~1 MB payload used to be refused at 64 KB."""
    assert len(json.dumps(LARGE).encode()) > 1_000_000
    outputs, sandbox = await _run(tmp_path, language, LARGE)
    assert _read_back(outputs) == LARGE
    # Too big for an env var: one env string over 128 KiB makes every
    # subprocess the user's code starts fail with E2BIG.
    assert all(len(v) < 100_000 for v in sandbox.calls[0]["envs"].values())


@pytest.mark.parametrize("language", _LANGUAGES)
async def test_large_payload_path_is_exposed_to_the_code(tmp_path, language):
    code = (
        "import json, os\nprint(json.dumps(len(json.load(open(os.environ['AGPT_VARIABLES_FILE']))['notifications'])))"
        if language is PYTHON
        else "console.log(JSON.stringify(JSON.parse(require('fs').readFileSync(process.env.AGPT_VARIABLES_FILE, 'utf-8')).notifications.length))"
    )
    outputs, _ = await _run(tmp_path, language, LARGE, code)
    assert _read_back(outputs) == len(LARGE["notifications"])


@pytest.mark.parametrize("large", [False, True])
async def test_non_finite_numbers_do_not_break_javascript(tmp_path, large):
    """JSON has no NaN/Infinity; Python's json writes them anyway and
    JavaScript's JSON.parse rejects them. Spreadsheet data is full of NaN."""
    if _NODE is None:
        pytest.skip("node is not installed")
    variables = {
        "rows": [1.5, math.nan, math.inf, -math.inf, {"nested": math.nan}],
        "text": "NaN Infinity -Infinity",
        "padding": "x" * (MAX_ENV_PAYLOAD_BYTES if large else 0),
    }
    outputs, _ = await _run(tmp_path, JAVASCRIPT, variables)
    # The same values JavaScript's own JSON.stringify would produce.
    assert _read_back(outputs) == {
        **variables,
        "rows": [1.5, None, None, None, {"nested": None}],
    }


@pytest.mark.parametrize("large", [False, True])
async def test_non_finite_numbers_survive_in_python(tmp_path, large):
    variables = {
        "rows": [1.5, math.nan, math.inf, -math.inf],
        "padding": "x" * (MAX_ENV_PAYLOAD_BYTES if large else 0),
    }
    code = "print(repr(rows))"
    outputs, _ = await _run(tmp_path, PYTHON, variables, code)
    assert "error" not in outputs, outputs["error"]
    assert outputs["stdout_logs"].strip() == "[1.5, nan, inf, -inf]"


async def test_small_payload_still_uses_the_env_var(tmp_path):
    """Code that reads AGPT_VARIABLES itself keeps working for every payload
    size that worked before."""
    code = (
        "import base64, json, os\n"
        "print(json.dumps(json.loads(base64.b64decode(os.environ['AGPT_VARIABLES']))))"
    )
    outputs, sandbox = await _run(tmp_path, PYTHON, TRICKY, code)
    assert _read_back(outputs) == TRICKY
    assert sandbox.calls[0]["files"] == {}


@pytest.mark.parametrize("language", _LANGUAGES)
async def test_large_payload_does_not_break_subprocesses(tmp_path, language):
    code = (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'print(42)'], check=True)"
        if language is PYTHON
        else "console.log(require('child_process').execFileSync("
        "process.execPath, ['-e', 'console.log(42)'], {encoding: 'utf-8'}).trim())"
    )
    outputs, _ = await _run(tmp_path, language, LARGE, code)
    assert _read_back(outputs) == 42


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
async def test_large_payload_in_javascript_module(tmp_path):
    outputs, _ = await _run(tmp_path, JAVASCRIPT, LARGE, javascript_module=True)
    assert _read_back(outputs) == LARGE


@pytest.mark.parametrize("language", [PYTHON, JAVASCRIPT])
@pytest.mark.parametrize(
    "size",
    [
        MAX_ENV_PAYLOAD_BYTES - 1,
        MAX_ENV_PAYLOAD_BYTES,
        MAX_ENV_PAYLOAD_BYTES + 1,
        MAX_VARIABLES_PAYLOAD_BYTES,
    ],
)
def test_serialized_size_boundaries(language, size):
    variables = {"data": "x" * (size - len(json.dumps({"data": ""})))}
    injection = build_variable_injection(variables, language)
    assert bool(injection.files) == (size > MAX_ENV_PAYLOAD_BYTES)
    assert (VARIABLES_FILE_ENV_KEY in injection.envs) == bool(injection.files)
    if injection.files:
        assert json.loads(next(iter(injection.files.values()))) == variables


@pytest.mark.parametrize("language", _LANGUAGES)
async def test_over_limit_does_not_start_sandbox(tmp_path, language):
    variables = {"data": "x" * (MAX_VARIABLES_PAYLOAD_BYTES + 1)}
    outputs, sandbox = await _run(tmp_path, language, variables)
    assert "max 10 MB" in outputs["error"]
    sandbox.create.assert_not_awaited()
    sandbox.files.write.assert_not_awaited()


async def test_file_upload_failure_stops_execution_and_disposes_sandbox(tmp_path):
    with patch.object(
        LocalSandbox, "write_file", AsyncMock(side_effect=OSError("upload failed"))
    ):
        outputs, sandbox = await _run(tmp_path, PYTHON, LARGE)
    assert outputs["error"] == "upload failed"
    assert sandbox.calls == []
    sandbox.dispose.assert_awaited_once_with(sandbox)
