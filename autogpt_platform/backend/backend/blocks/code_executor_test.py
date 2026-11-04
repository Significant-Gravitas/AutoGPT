"""Tests for the Execute Code block's variable-injection helper.

The helper serializes user-provided variables to JSON, passes them via an
environment variable, and prepends a constant snippet that deserializes them
into named variables inside the sandbox. Keeping user data in the env var (the
data channel) rather than the code string (the code channel) avoids code
injection -- analogous to parameterized SQL queries.
"""

import base64
import json
import uuid
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from e2b_code_interpreter.constants import DEFAULT_TIMEOUT as RUN_CODE_TIMEOUT
from pydantic import SecretStr

from backend.blocks.code_executor import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ExecuteCodeBlock,
    ExecuteCodeStepBlock,
    InstantiateCodeSandboxBlock,
    ProgrammingLanguage,
)
from backend.blocks.code_executor_helpers import (
    MAX_ENV_PAYLOAD_BYTES,
    MAX_VARIABLES_PAYLOAD_BYTES,
    VARIABLES_ENV_KEY,
    VARIABLES_FILE_ENV_KEY,
    VARIABLES_FILE_PATH,
    UnsupportedLanguageError,
    build_variable_injection,
)
from backend.data.execution import ExecutionContext
from backend.integrations.credentials_store import e2b_credentials
from backend.util.sandbox_output import (
    OutputLimitedCodeSandbox,
    SandboxOutputLimitError,
)

# A credential with no key of its own: E2B opens the box on the operator's,
# so a block must ask for the pause as it does for the platform's own.
_EMPTY_KEY_CREDENTIALS = TEST_CREDENTIALS.model_copy(update={"api_key": SecretStr("")})


def _b64_json(variables: dict) -> str:
    """Match build_variable_injection's own encoding for test assertions."""
    return base64.b64encode(json.dumps(variables).encode("utf-8")).decode("ascii")


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
        injection = build_variable_injection({}, ProgrammingLanguage.PYTHON)
        assert injection.envs == {}
        assert injection.files == {}
        assert injection.prefix == ""

    def test_python_serializes_to_env_and_unpacks_to_globals(self):
        variables = {"x": 42, "name": "Blake", "items": [1, 2, 3]}
        injection = build_variable_injection(variables, ProgrammingLanguage.PYTHON)
        envs, prefix = injection.envs, injection.prefix

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
        injection = build_variable_injection(variables, ProgrammingLanguage.JAVASCRIPT)
        envs, prefix = injection.envs, injection.prefix

        assert envs == {VARIABLES_ENV_KEY: _b64_json(variables)}
        assert "base64" in prefix
        assert "JSON.parse" in prefix
        assert "Object.assign(globalThis" in prefix
        assert "process.env" in prefix
        assert "Blake" not in prefix

    def test_malicious_value_cannot_break_out_of_code_channel(self):
        """A value that looks like code stays inert: it's only ever JSON data."""
        variables = {"evil": "'); import os; os.system('rm -rf /'); ('"}
        injection = build_variable_injection(variables, ProgrammingLanguage.PYTHON)
        envs, prefix = injection.envs, injection.prefix
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

    def test_payload_over_the_env_limit_goes_to_a_file(self):
        variables = {"data": "x" * MAX_ENV_PAYLOAD_BYTES}
        injection = build_variable_injection(variables, ProgrammingLanguage.PYTHON)

        assert injection.envs == {VARIABLES_FILE_ENV_KEY: VARIABLES_FILE_PATH}
        assert json.loads(injection.files[VARIABLES_FILE_PATH]) == variables
        assert VARIABLES_FILE_ENV_KEY in injection.prefix
        assert "x" * 100 not in injection.prefix

    def test_real_emoji_survives_sanitization_untouched(self):
        """Properly-paired surrogates (real emoji) are valid Unicode and must
        round-trip through the injection helper untouched.
        """
        variables = {"note": "Holidays \U0001f385\U0001f3fb"}
        envs = build_variable_injection(variables, ProgrammingLanguage.PYTHON).envs

        assert _decode_env_payload(envs) == variables

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
        envs = build_variable_injection(variables, ProgrammingLanguage.PYTHON).envs

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
        with (
            patch.object(block, "execute_code", mock),
            patch.object(
                block, "process_execution_results", return_value=({"text": "42"}, [])
            ),
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


class TestExecuteCodeWritesFiles:
    async def test_files_are_written_before_the_code_runs(self):
        block = ExecuteCodeBlock()
        calls: list[str] = []
        execution = MagicMock(error=None, results=[], text="")
        execution.logs.stdout, execution.logs.stderr = [], []

        async def write(*_):
            calls.append("write")

        async def command(*_, **__):
            calls.append("setup")

        async def run_code(*_, **__):
            calls.append("run_code")
            return execution

        sandbox = MagicMock(sandbox_id="sb")
        sandbox.commands.run = AsyncMock(side_effect=command)
        sandbox.files.write = AsyncMock(side_effect=write)
        sandbox.run_code = AsyncMock(side_effect=run_code)
        with patch(
            "backend.blocks.code_executor.create_sandbox",
            AsyncMock(return_value=sandbox),
        ):
            await block.execute_code(
                api_key="k",
                code="print(1)",
                language=ProgrammingLanguage.PYTHON,
                setup_commands=["echo setup"],
                envs={VARIABLES_FILE_ENV_KEY: VARIABLES_FILE_PATH},
                input_files={VARIABLES_FILE_PATH: b"{}"},
            )

        assert calls == ["setup", "write", "run_code"]
        sandbox.files.write.assert_awaited_once_with(VARIABLES_FILE_PATH, b"{}")
        assert sandbox.run_code.await_args.kwargs["envs"] == {
            VARIABLES_FILE_ENV_KEY: VARIABLES_FILE_PATH
        }


class TestCodeSandboxClass:
    """``run_code`` is bounded only on ``OutputLimitedCodeSandbox``."""

    @staticmethod
    def _sandbox() -> MagicMock:
        execution = MagicMock(error=None, results=[], text="")
        execution.logs.stdout, execution.logs.stderr = [], []
        sandbox = MagicMock(sandbox_id="sb")
        sandbox.run_code = AsyncMock(return_value=execution)
        return sandbox

    async def test_a_new_sandbox_is_output_limited(self):
        create = AsyncMock(return_value=self._sandbox())
        with patch("backend.blocks.code_executor.create_sandbox", create):
            await ExecuteCodeBlock().execute_code(
                api_key="k", code="print(1)", language=ProgrammingLanguage.PYTHON
            )

        assert create.await_args.args[0] is OutputLimitedCodeSandbox

    async def test_a_reconnected_sandbox_is_output_limited(self):
        connect = AsyncMock(return_value=self._sandbox())
        with (
            patch("backend.blocks.code_executor.AsyncSandbox") as cls,
            patch("backend.blocks.code_executor.owned_by_user", return_value=True),
            patch("backend.blocks.code_executor.connect_sandbox", connect),
        ):
            cls.get_info = AsyncMock(return_value=MagicMock(metadata={}))
            await ExecuteCodeBlock().execute_code(
                api_key="k",
                code="print(1)",
                language=ProgrammingLanguage.PYTHON,
                sandbox_id="sb-mine",
                execution_context=ExecutionContext(user_id="user-a"),
            )

        assert connect.await_args.args[0] is OutputLimitedCodeSandbox


class TestConnectToExistingSandbox:
    """A sandbox id is caller input; under the platform key any id connects,
    so the box's stamped user decides whether it may be used."""

    def _sandbox(self, stamped: dict) -> AsyncMock:
        sandbox = AsyncMock()
        sandbox.get_info = AsyncMock(return_value=AsyncMock(metadata=stamped))
        return sandbox

    @staticmethod
    def _sdk(cls: MagicMock, box: AsyncMock) -> None:
        """The stamp is read through the static ``get_info`` before any connect."""
        cls.get_info = AsyncMock(return_value=box.get_info.return_value)
        cls.connect = AsyncMock(return_value=box)

    @pytest.mark.parametrize(
        "dispose_sandbox, pause_when_kept",
        [(True, False), (False, True)],
        ids=["disposed", "kept-on-platform-key"],
    )
    async def test_another_users_sandbox_is_refused_and_left_alone(
        self, dispose_sandbox, pause_when_kept
    ):
        block = ExecuteCodeBlock()
        context = ExecutionContext(user_id="user-b", graph_exec_id="gexec-1")
        theirs = self._sandbox(
            {
                "service": "autogpt-platform",
                "autogpt_owner": "user:user-a",
                "autogpt_kind": "code",
                "autogpt_source": "block",
                "autogpt_env": "dev",
                "autogpt_user": "user-a",
            }
        )
        hold = AsyncMock()
        with (
            patch("backend.blocks.code_executor.AsyncSandbox") as cls,
            patch("backend.blocks.code_executor.hold_sandbox", hold),
            patch("backend.util.e2b_network.AsyncSandbox") as by_id,
        ):
            self._sdk(cls, theirs)
            by_id.pause = AsyncMock()
            with pytest.raises(PermissionError, match="does not belong"):
                await block.execute_code(
                    api_key="k",
                    code="print(1)",
                    language=ProgrammingLanguage.PYTHON,
                    sandbox_id="sb-theirs",
                    dispose_sandbox=dispose_sandbox,
                    pause_when_kept=pause_when_kept,
                    execution_context=context,
                )
        cls.get_info.assert_awaited_once_with("sb-theirs", api_key="k")
        # Connecting would resume the other user's box on their bill.
        cls.connect.assert_not_awaited()
        theirs.run_code.assert_not_awaited()
        # Not ours to kill, count or pause, whatever the run would do with its own.
        theirs.kill.assert_not_awaited()
        hold.assert_not_awaited()
        by_id.pause.assert_not_awaited()

    async def test_the_users_own_box_of_another_kind_is_refused(self):
        """A code step continues the boxes the code sandbox blocks make, not a
        box the same user has from another block."""
        block = ExecuteCodeStepBlock()
        context = ExecutionContext(user_id="user-a", graph_exec_id="gexec-1")
        claude_box = self._sandbox(
            {
                "service": "autogpt-platform",
                "autogpt_owner": "user:user-a",
                "autogpt_kind": "claude_code",
                "autogpt_source": "block",
                "autogpt_env": "dev",
                "autogpt_user": "user-a",
            }
        )
        with (
            patch("backend.blocks.code_executor.AsyncSandbox") as cls,
            patch("backend.blocks.code_executor.hold_sandbox", AsyncMock()) as hold,
            patch(
                "backend.blocks.code_executor.release_sandbox", AsyncMock()
            ) as release,
            patch(
                "backend.blocks.code_executor.connect_sandbox", AsyncMock()
            ) as connect,
        ):
            self._sdk(cls, claude_box)
            with pytest.raises(PermissionError, match="not a code sandbox"):
                await block.execute_code(
                    api_key="k",
                    code="print(1)",
                    language=ProgrammingLanguage.PYTHON,
                    sandbox_id="sb-claude",
                    dispose_sandbox=False,
                    pause_when_kept=True,
                    execution_context=context,
                )
        # Refused before the box is counted as held, so nothing pauses it.
        hold.assert_not_awaited()
        release.assert_not_awaited()
        connect.assert_not_awaited()
        claude_box.run_code.assert_not_awaited()

    async def test_the_users_own_code_box_is_reconnected(self):
        block = ExecuteCodeStepBlock()
        context = ExecutionContext(user_id="user-a", graph_exec_id="gexec-1")
        code_box = self._sandbox(
            {
                "service": "autogpt-platform",
                "autogpt_owner": "user:user-a",
                "autogpt_kind": "code",
                "autogpt_source": "block",
                "autogpt_env": "dev",
                "autogpt_user": "user-a",
            }
        )
        code_box.sandbox_id = "sb-code"
        code_box.run_code = AsyncMock(
            return_value=MagicMock(
                error=None,
                results=[],
                text="1",
                logs=MagicMock(stdout=["1\n"], stderr=[]),
            )
        )
        with (
            patch("backend.blocks.code_executor.AsyncSandbox") as cls,
            patch(
                "backend.blocks.code_executor.connect_sandbox",
                AsyncMock(return_value=code_box),
            ) as connect,
        ):
            self._sdk(cls, code_box)
            *_, sandbox_id, _ = await block.execute_code(
                api_key="k",
                code="print(1)",
                language=ProgrammingLanguage.PYTHON,
                sandbox_id="sb-code",
                dispose_sandbox=False,
                execution_context=context,
            )
        connect.assert_awaited_once()
        code_box.run_code.assert_awaited_once()
        assert sandbox_id == "sb-code"

    async def test_step_block_hands_the_caller_to_the_ownership_check(self):
        """The step block always reconnects by id, so without the caller's
        context every run would be refused as not the user's box."""
        block = ExecuteCodeStepBlock()
        context = ExecutionContext(user_id="user-a", graph_exec_id="gexec-1")
        with patch.object(
            block, "execute_code", AsyncMock(return_value=([], "", "", "", "sb", []))
        ) as execute:
            async for _ in block.run(
                ExecuteCodeStepBlock.Input(
                    credentials=TEST_CREDENTIALS_INPUT,
                    sandbox_id="sb-mine",
                    step_code="print(1)",
                ),
                credentials=TEST_CREDENTIALS,
                execution_context=context,
            ):
                pass
        assert execute.await_args.kwargs["execution_context"] is context
        assert execute.await_args.kwargs["sandbox_id"] == "sb-mine"

    async def test_instantiate_block_hands_the_caller_to_the_box_it_creates(self):
        """The box is created for this user: without the context its egress
        credential would name nobody, and the proxy could swap nothing in."""
        block = InstantiateCodeSandboxBlock()
        context = ExecutionContext(user_id="user-a", graph_exec_id="gexec-1")
        with patch.object(
            block, "execute_code", AsyncMock(return_value=([], "", "", "", "sb", []))
        ) as execute:
            async for _ in block.run(
                InstantiateCodeSandboxBlock.Input(
                    credentials=TEST_CREDENTIALS_INPUT, setup_code="print(1)"
                ),
                credentials=TEST_CREDENTIALS,
                execution_context=context,
            ):
                pass
        assert execute.await_args.kwargs["execution_context"] is context

    async def test_without_a_user_no_existing_sandbox_can_be_used(self):
        block = ExecuteCodeBlock()
        box = self._sandbox(
            {
                "service": "autogpt-platform",
                "autogpt_owner": "user:user-a",
                "autogpt_kind": "code",
                "autogpt_source": "block",
                "autogpt_env": "dev",
                "autogpt_user": "user-a",
            }
        )
        with patch("backend.blocks.code_executor.AsyncSandbox") as cls:
            self._sdk(cls, box)
            with pytest.raises(PermissionError):
                await block.execute_code(
                    api_key="k",
                    code="print(1)",
                    language=ProgrammingLanguage.PYTHON,
                    sandbox_id="sb-theirs",
                    execution_context=None,
                )
        cls.connect.assert_not_awaited()


def _executed_box(sandbox_id: str) -> MagicMock:
    execution = MagicMock(error=None, results=[], text="")
    execution.logs.stdout, execution.logs.stderr = [], []
    box = MagicMock(sandbox_id=sandbox_id)
    box.run_code = AsyncMock(return_value=execution)
    return box


class TestKeptSandbox:
    """When a block returns, a disposed box is killed, and a kept box is
    released (paused once no other run holds it) on the platform's E2B key and
    left running on the user's own key."""

    @pytest.mark.parametrize(
        "dispose_sandbox, pause_when_kept, outcome",
        [
            (True, True, "kill"),
            (True, False, "kill"),
            (False, True, "release"),
            (False, False, None),
        ],
    )
    async def test_the_box_is_killed_released_or_left_running(
        self, dispose_sandbox, pause_when_kept, outcome
    ):
        box = _executed_box("sb")
        m = "backend.blocks.code_executor"
        kill, hold, release = AsyncMock(), AsyncMock(), AsyncMock()
        with (
            patch(f"{m}.create_sandbox", AsyncMock(return_value=box)),
            patch(f"{m}.kill_sandbox", kill),
            patch(f"{m}.hold_sandbox", hold),
            patch(f"{m}.release_sandbox", release),
        ):
            await ExecuteCodeBlock().execute_code(
                api_key="k",
                code="print(1)",
                language=ProgrammingLanguage.PYTHON,
                dispose_sandbox=dispose_sandbox,
                pause_when_kept=pause_when_kept,
            )
        kept = outcome == "release"
        assert kill.await_args_list == ([call(box)] if outcome == "kill" else [])
        assert hold.await_args_list == ([call("sb")] if kept else [])
        assert release.await_args_list == (
            [call("sb", box, api_key="k")] if kept else []
        )

    @pytest.mark.parametrize(
        "dispose_sandbox, outcome", [(True, "kill"), (False, "release")]
    )
    async def test_output_past_the_limit_still_kills_or_releases_the_box(
        self, dispose_sandbox, outcome
    ):
        box = _executed_box("sb")
        box.run_code = AsyncMock(side_effect=SandboxOutputLimitError("too much"))
        m = "backend.blocks.code_executor"
        kill, hold, release = AsyncMock(), AsyncMock(), AsyncMock()
        with (
            patch(f"{m}.create_sandbox", AsyncMock(return_value=box)),
            patch(f"{m}.kill_sandbox", kill),
            patch(f"{m}.hold_sandbox", hold),
            patch(f"{m}.release_sandbox", release),
            pytest.raises(SandboxOutputLimitError),
        ):
            await ExecuteCodeBlock().execute_code(
                api_key="k",
                code="print(1)",
                language=ProgrammingLanguage.PYTHON,
                dispose_sandbox=dispose_sandbox,
                pause_when_kept=True,
            )
        assert kill.await_args_list == ([call(box)] if outcome == "kill" else [])
        assert release.await_args_list == (
            [call("sb", box, api_key="k")] if outcome == "release" else []
        )

    @pytest.mark.parametrize("connects", [True, False])
    async def test_a_resumed_box_is_counted_before_it_connects(self, connects):
        """Another run returning during the connect must not pause the box
        under this one, and the box must outlive ``run_code``'s own limit."""
        context = _execution_context()
        user = context.user_id
        stamp = {
            "service": "autogpt-platform",
            "autogpt_owner": f"user:{user}",
            "autogpt_kind": "code",
            "autogpt_source": "block",
            "autogpt_env": "dev",
            "autogpt_user": user,
        }
        box = _executed_box("sb-mine")
        order: list[str] = []
        hold = AsyncMock(side_effect=lambda *_: order.append("hold"))

        async def connect(*args, **kwargs):
            order.append("connect")
            if not connects:
                raise RuntimeError("connect failed")
            return box

        m = "backend.blocks.code_executor"
        release, connect_mock = AsyncMock(), AsyncMock(side_effect=connect)
        with (
            patch(f"{m}.AsyncSandbox") as cls,
            patch(f"{m}.connect_sandbox", connect_mock),
            patch(f"{m}.hold_sandbox", hold),
            patch(f"{m}.release_sandbox", release),
        ):
            cls.get_info = AsyncMock(return_value=MagicMock(metadata=stamp))
            with nullcontext() if connects else pytest.raises(RuntimeError):
                await ExecuteCodeStepBlock().execute_code(
                    api_key="k",
                    code="1",
                    language=ProgrammingLanguage.PYTHON,
                    sandbox_id="sb-mine",
                    pause_when_kept=True,
                    execution_context=context,
                )
        assert order == ["hold", "connect"]
        assert connect_mock.await_args.kwargs["timeout"] > RUN_CODE_TIMEOUT
        assert release.await_args_list == [
            call("sb-mine", box if connects else None, api_key="k")
        ]

    @pytest.mark.parametrize(
        "block_cls, fields",
        [
            (ExecuteCodeBlock, {"code": "print(1)", "dispose_sandbox": False}),
            (InstantiateCodeSandboxBlock, {"setup_code": "print(1)"}),
            (ExecuteCodeStepBlock, {"sandbox_id": "sb-mine", "step_code": "1"}),
        ],
    )
    @pytest.mark.parametrize(
        "credentials, paused",
        [
            (e2b_credentials, True),
            (TEST_CREDENTIALS, False),
            (_EMPTY_KEY_CREDENTIALS, True),
        ],
        ids=["platform-key", "own-key", "empty-key"],
    )
    async def test_each_block_pauses_a_kept_box_only_on_the_platform_key(
        self, block_cls, fields, credentials, paused
    ):
        block = block_cls()
        with patch.object(
            block, "execute_code", AsyncMock(return_value=([], "", "", "", "sb", []))
        ) as execute:
            async for _ in block.run(
                block_cls.Input(
                    credentials={**TEST_CREDENTIALS_INPUT, "id": credentials.id},
                    **fields,
                ),
                credentials=credentials,
                execution_context=_execution_context(),
            ):
                pass
        assert execute.await_args.kwargs["pause_when_kept"] is paused

    @pytest.mark.parametrize(
        "credentials, disposed",
        [
            (e2b_credentials, True),
            (_EMPTY_KEY_CREDENTIALS, True),
            (TEST_CREDENTIALS, False),
        ],
        ids=["platform-key", "empty-key", "own-key"],
    )
    async def test_a_box_this_block_cannot_return_is_disposed_on_the_operators_key(
        self, credentials, disposed
    ):
        """ExecuteCodeBlock returns no sandbox ID, so a box it keeps can never
        be reached again; on the operator's key it must not be left paused for
        good, whatever the user asked for."""
        block = ExecuteCodeBlock()
        with patch.object(
            block, "execute_code", AsyncMock(return_value=([], "", "", "", "sb", []))
        ) as execute:
            async for _ in block.run(
                ExecuteCodeBlock.Input(
                    credentials={**TEST_CREDENTIALS_INPUT, "id": credentials.id},
                    code="print(1)",
                    dispose_sandbox=False,
                ),
                credentials=credentials,
                execution_context=_execution_context(),
            ):
                pass
        assert execute.await_args.kwargs["dispose_sandbox"] is disposed

    async def test_the_block_that_returns_its_box_keeps_it_for_reuse(self):
        """InstantiateCodeSandboxBlock's box is reachable through its
        sandbox_id output, so it is paused for the next step, never disposed."""
        block = InstantiateCodeSandboxBlock()
        with patch.object(
            block, "execute_code", AsyncMock(return_value=([], "", "", "", "sb", []))
        ) as execute:
            async for _ in block.run(
                InstantiateCodeSandboxBlock.Input(
                    credentials={**TEST_CREDENTIALS_INPUT, "id": e2b_credentials.id},
                    setup_code="print(1)",
                ),
                credentials=e2b_credentials,
                execution_context=_execution_context(),
            ):
                pass
        assert execute.await_args.kwargs.get("dispose_sandbox", False) is False
        assert execute.await_args.kwargs["pause_when_kept"] is True
