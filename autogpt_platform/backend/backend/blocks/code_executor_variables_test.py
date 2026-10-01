"""End-to-end tests for how ExecuteCodeBlock hands `variables` to the code.

`execute_code` is replaced by a local stand-in that does what the E2B sandbox
does with what the block sends it: write any files, set the env vars, and run
the prefixed code with a real Python or Node interpreter. That way these tests
exercise the real serializer *and* the real in-sandbox prefix together, which
is where the data has to come out unchanged.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.blocks.code_executor import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ExecuteCodeBlock,
    ProgrammingLanguage,
)
from backend.executor.utils import ExecutionContext

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

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.calls: list[dict] = []

    async def execute_code(self, *, code, language, envs=None, files=None, **_):
        env = {**os.environ, **(envs or {})}
        for sandbox_path, data in (files or {}).items():
            local = self.workdir / Path(sandbox_path).name
            local.write_bytes(data)
            # The prefix finds the file through the env var, so point it here.
            env = {k: str(local) if v == sandbox_path else v for k, v in env.items()}
        self.calls.append({"envs": envs or {}, "files": files or {}})
        cmd = (
            [sys.executable, "-c", code]
            if language is PYTHON
            else [str(_NODE), "-e", code]
        )
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            lines = proc.stderr.strip().splitlines()
            raise RuntimeError(next((ln for ln in lines if "Error" in ln), lines[-1]))
        return [], "", proc.stdout, proc.stderr, "local", []


def _dump_code(language: ProgrammingLanguage, names: list[str]) -> str:
    if language is PYTHON:
        return f"import json\nprint(json.dumps({{k: globals()[k] for k in {names!r}}}))"
    return f"console.log(JSON.stringify({{{', '.join(names)}}}))"


async def _run(tmp_path: Path, language, variables: dict, code: str | None = None):
    block = ExecuteCodeBlock()
    sandbox = LocalSandbox(tmp_path)
    input_data = ExecuteCodeBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "language": language.value,
            "variables": variables,
            "code": code or _dump_code(language, list(variables)),
        }
    )
    context = ExecutionContext(user_id=str(uuid.uuid4()), graph_exec_id="gexec")
    with patch.object(block, "execute_code", sandbox.execute_code):
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


async def test_non_finite_numbers_do_not_break_javascript(tmp_path):
    """JSON has no NaN/Infinity; Python's json writes them anyway and
    JavaScript's JSON.parse rejects them. Spreadsheet data is full of NaN."""
    if _NODE is None:
        pytest.skip("node is not installed")
    variables = {"rows": [1.5, math.nan, math.inf, -math.inf]}
    outputs, _ = await _run(tmp_path, JAVASCRIPT, variables)
    # The same values JavaScript's own JSON.stringify would produce.
    assert _read_back(outputs) == {"rows": [1.5, None, None, None]}


async def test_non_finite_numbers_survive_in_python(tmp_path):
    variables = {"rows": [1.5, math.nan, math.inf]}
    code = "print(repr(rows))"
    outputs, _ = await _run(tmp_path, PYTHON, variables, code)
    assert "error" not in outputs, outputs["error"]
    assert outputs["stdout_logs"].strip() == "[1.5, nan, inf]"


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
