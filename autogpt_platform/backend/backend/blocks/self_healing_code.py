"""
Self-Healing Code Block — runs tests, detects failures, and feeds errors back
to the LLM for automatic fixing. Implements the write-test-fix loop.

Also includes auto unit test generation: given a function/module, generate
a comprehensive pytest test suite.
"""

import logging
import re
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 5


class HealingOperation(str, Enum):
    RUN_TESTS = "run_tests"
    GENERATE_TESTS = "generate_tests"
    ANALYZE_FAILURE = "analyze_failure"


class SelfHealingInput(BlockSchemaInput):
    operation: HealingOperation = SchemaField(
        default=HealingOperation.RUN_TESTS,
        description="Operation: run_tests, generate_tests, or analyze_failure.",
    )
    code: str = SchemaField(
        default="",
        description="Source code to test or generate tests for.",
    )
    test_code: str = SchemaField(
        default="",
        description="Existing test code to run (for RUN_TESTS).",
    )
    test_framework: str = SchemaField(
        default="pytest",
        description="Test framework: 'pytest' or 'unittest'.",
    )
    error_output: str = SchemaField(
        default="",
        description="Test failure output to analyze (for ANALYZE_FAILURE).",
    )
    language: str = SchemaField(
        default="python",
        description="Programming language of the code.",
    )
    module_name: str = SchemaField(
        default="module",
        description="Module/file name for test generation context.",
    )
    retry_count: int = SchemaField(
        default=0,
        description="Current retry attempt number (for tracking healing iterations).",
    )
    max_retries: int = SchemaField(
        default=MAX_RETRY_ATTEMPTS,
        description="Maximum number of healing retry attempts before giving up.",
    )


class SelfHealingOutput(BlockSchemaOutput):
    tests_passed: bool = SchemaField(description="True if all tests passed.")
    test_output: str = SchemaField(description="Full test runner output.")
    failure_summary: str = SchemaField(description="Concise summary of test failures.")
    fix_prompt: str = SchemaField(
        description="Prompt to send to LLM for fixing the failing code."
    )
    generated_tests: str = SchemaField(description="Generated test code (for GENERATE_TESTS).")
    should_retry: bool = SchemaField(description="True if healing should retry.")
    retry_count: int = SchemaField(description="Updated retry count.")
    status: str = SchemaField(description="Operation status message.")


def _run_pytest(code: str, test_code: str) -> tuple[bool, str]:
    """Run pytest in a temp directory and return (passed, output)."""
    with tempfile.TemporaryDirectory(prefix="autogpt_test_") as tmpdir:
        # Write source code
        src_file = Path(tmpdir) / "module.py"
        src_file.write_text(code)

        # Write test code
        test_file = Path(tmpdir) / "test_module.py"
        # Ensure the test imports from the local module
        if "from module import" not in test_code and "import module" not in test_code:
            test_code = "from module import *\n\n" + test_code
        test_file.write_text(test_code)

        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", str(test_file), "-v", "--tb=short", "--no-header"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0
            return passed, output
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out after 60 seconds."
        except FileNotFoundError:
            return False, "pytest not found. Run: pip install pytest"


def _extract_failures(test_output: str) -> str:
    """Extract a concise failure summary from pytest output."""
    lines = test_output.split("\n")
    failure_lines = []
    in_failure = False
    for line in lines:
        if line.startswith("FAILED") or line.startswith("ERROR"):
            failure_lines.append(line)
            in_failure = True
        elif in_failure and line.strip():
            failure_lines.append(line)
        elif "short test summary" in line.lower():
            in_failure = False
    return "\n".join(failure_lines[:50])  # Limit to 50 lines


def _build_fix_prompt(code: str, test_code: str, failure_summary: str, retry_count: int) -> str:
    """Build an LLM prompt for fixing failing code."""
    return f"""The following code has failing tests. This is retry attempt {retry_count + 1}.

## Source Code
```python
{code}
```

## Test Code
```python
{test_code}
```

## Test Failures
```
{failure_summary}
```

## Instructions
Analyze the test failures and fix the source code. Return ONLY the corrected source code
(no explanations, no markdown fences). The code must pass all the tests above.
Focus on:
1. Fixing the specific assertions that are failing
2. Handling edge cases the tests expose
3. Not breaking any currently passing tests
"""


def _generate_test_prompt(code: str, module_name: str, language: str) -> str:
    """Generate a prompt for creating tests."""
    return f"""Generate a comprehensive pytest test suite for the following {language} code.

## Module: {module_name}
```{language}
{code}
```

## Requirements
- Use pytest with descriptive test function names (test_<function>_<scenario>)
- Cover: happy path, edge cases, error conditions, boundary values
- Use pytest.mark.parametrize for data-driven tests where appropriate
- Include docstrings explaining what each test verifies
- Do NOT import the module at the top level; use `from module import *` or specific imports
- Return ONLY the test code, no explanations
"""


class SelfHealingCodeBlock(Block):
    """
    Implements the write-test-fix loop for self-healing code.

    Runs tests, detects failures, generates fix prompts for the LLM,
    and tracks retry attempts. Also generates unit tests from source code.
    Connect the fix_prompt output to an LLM block and feed the result
    back as updated code for the next iteration.
    """

    class Input(SelfHealingInput):
        pass

    class Output(SelfHealingOutput):
        pass

    def __init__(self):
        super().__init__(
            id="c9d0e1f2-a3b4-5678-cdef-901234567890",
            description=(
                "Self-healing code loop: runs tests, detects failures, generates LLM fix prompts, "
                "and tracks retries. Also generates pytest test suites from source code."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.AI},
            input_schema=SelfHealingCodeBlock.Input,
            output_schema=SelfHealingCodeBlock.Output,
            test_input={
                "operation": HealingOperation.RUN_TESTS.value,
                "code": "def add(a, b):\n    return a + b",
                "test_code": "def test_add():\n    assert add(1, 2) == 3",
                "language": "python",
            },
            test_output=[
                ("tests_passed", True),
                ("should_retry", False),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        if input_data.operation == HealingOperation.RUN_TESTS:
            if not input_data.code or not input_data.test_code:
                yield "tests_passed", False
                yield "test_output", ""
                yield "failure_summary", "No code or test code provided."
                yield "fix_prompt", ""
                yield "generated_tests", ""
                yield "should_retry", False
                yield "retry_count", input_data.retry_count
                yield "status", "Missing code or test_code."
                return

            passed, output = _run_pytest(input_data.code, input_data.test_code)
            failure_summary = "" if passed else _extract_failures(output)
            should_retry = (
                not passed and input_data.retry_count < input_data.max_retries
            )
            fix_prompt = ""
            if not passed and should_retry:
                fix_prompt = _build_fix_prompt(
                    input_data.code,
                    input_data.test_code,
                    failure_summary,
                    input_data.retry_count,
                )

            yield "tests_passed", passed
            yield "test_output", output
            yield "failure_summary", failure_summary
            yield "fix_prompt", fix_prompt
            yield "generated_tests", ""
            yield "should_retry", should_retry
            yield "retry_count", input_data.retry_count + (0 if passed else 1)
            yield "status", (
                f"All tests passed." if passed
                else f"Tests failed (attempt {input_data.retry_count + 1}/{input_data.max_retries})."
            )

        elif input_data.operation == HealingOperation.GENERATE_TESTS:
            prompt = _generate_test_prompt(
                input_data.code,
                input_data.module_name,
                input_data.language,
            )
            yield "tests_passed", False
            yield "test_output", ""
            yield "failure_summary", ""
            yield "fix_prompt", prompt
            yield "generated_tests", prompt  # Caller passes this to LLM
            yield "should_retry", False
            yield "retry_count", 0
            yield "status", "Test generation prompt ready. Pass fix_prompt to LLM block."

        elif input_data.operation == HealingOperation.ANALYZE_FAILURE:
            if not input_data.error_output:
                yield "tests_passed", False
                yield "test_output", ""
                yield "failure_summary", "No error output provided."
                yield "fix_prompt", ""
                yield "generated_tests", ""
                yield "should_retry", False
                yield "retry_count", input_data.retry_count
                yield "status", "No error output to analyze."
                return

            failure_summary = _extract_failures(input_data.error_output)
            fix_prompt = _build_fix_prompt(
                input_data.code,
                input_data.test_code,
                failure_summary or input_data.error_output[:2000],
                input_data.retry_count,
            )
            should_retry = input_data.retry_count < input_data.max_retries

            yield "tests_passed", False
            yield "test_output", input_data.error_output
            yield "failure_summary", failure_summary
            yield "fix_prompt", fix_prompt
            yield "generated_tests", ""
            yield "should_retry", should_retry
            yield "retry_count", input_data.retry_count + 1
            yield "status", f"Failure analyzed. Fix prompt generated (attempt {input_data.retry_count + 1})."
