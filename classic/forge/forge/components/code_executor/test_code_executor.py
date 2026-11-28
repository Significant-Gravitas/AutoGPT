import random
import string
import tempfile
from pathlib import Path

import pytest

from forge.file_storage.base import FileStorage
from forge.utils.exceptions import InvalidArgumentError, OperationNotAllowedError

from .code_executor import (
    CodeExecutorComponent,
    is_docker_available,
    we_are_running_in_a_docker_container,
)


@pytest.fixture
def code_executor_component(storage: FileStorage):
    return CodeExecutorComponent(storage)


@pytest.fixture
def random_code(random_string) -> str:
    return f"print('Hello {random_string}!')"


@pytest.fixture
def python_test_file(storage: FileStorage, random_code: str):
    temp_file = tempfile.NamedTemporaryFile(dir=storage.root, suffix=".py")
    temp_file.write(str.encode(random_code))
    temp_file.flush()

    yield Path(temp_file.name)
    temp_file.close()


@pytest.fixture
def python_test_args_file(storage: FileStorage):
    temp_file = tempfile.NamedTemporaryFile(dir=storage.root, suffix=".py")
    temp_file.write(str.encode("import sys\nprint(sys.argv[1], sys.argv[2])"))
    temp_file.flush()

    yield Path(temp_file.name)
    temp_file.close()


@pytest.fixture
def random_string():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


def test_execute_python_file(
    code_executor_component: CodeExecutorComponent,
    python_test_file: Path,
    random_string: str,
):
    if not (is_docker_available() or we_are_running_in_a_docker_container()):
        pytest.skip("Docker is not available")

    result: str = code_executor_component.execute_python_file(python_test_file)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"


def test_execute_python_file_args(
    code_executor_component: CodeExecutorComponent,
    python_test_args_file: Path,
    random_string: str,
):
    if not (is_docker_available() or we_are_running_in_a_docker_container()):
        pytest.skip("Docker is not available")

    random_args = [random_string] * 2
    random_args_string = " ".join(random_args)
    result = code_executor_component.execute_python_file(
        python_test_args_file, args=random_args
    )
    assert result == f"{random_args_string}\n"


@pytest.mark.asyncio
async def test_execute_python_code(
    code_executor_component: CodeExecutorComponent,
    random_code: str,
    random_string: str,
):
    if not (is_docker_available() or we_are_running_in_a_docker_container()):
        pytest.skip("Docker is not available")

    result: str = await code_executor_component.execute_python_code(random_code)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"


def test_execute_python_file_invalid(code_executor_component: CodeExecutorComponent):
    with pytest.raises(InvalidArgumentError):
        code_executor_component.execute_python_file(Path("not_python.txt"))


def test_execute_python_file_not_found(code_executor_component: CodeExecutorComponent):
    with pytest.raises(
        FileNotFoundError,
        match=r"python: can't open file '([a-zA-Z]:)?[/\\\-\w]*notexist.py': "
        r"\[Errno 2\] No such file or directory",
    ):
        code_executor_component.execute_python_file(Path("notexist.py"))


def test_execute_shell(
    code_executor_component: CodeExecutorComponent, random_string: str
):
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    result = code_executor_component.execute_shell(f"echo 'Hello {random_string}!'")
    assert f"Hello {random_string}!" in result


def test_execute_shell_local_commands_not_allowed(
    code_executor_component: CodeExecutorComponent, random_string: str
):
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell(f"echo 'Hello {random_string}!'")


def test_execute_shell_denylist_should_deny(
    code_executor_component: CodeExecutorComponent, random_string: str
):
    code_executor_component.config.shell_command_control = "denylist"
    code_executor_component.config.shell_denylist = ["echo"]

    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell(f"echo 'Hello {random_string}!'")


def test_execute_shell_denylist_should_allow(
    code_executor_component: CodeExecutorComponent, random_string: str
):
    code_executor_component.config.shell_command_control = "denylist"
    code_executor_component.config.shell_denylist = ["cat"]

    result = code_executor_component.execute_shell(f"echo 'Hello {random_string}!'")
    assert "Hello" in result and random_string in result


def test_execute_shell_allowlist_should_deny(
    code_executor_component: CodeExecutorComponent, random_string: str
):
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["cat"]

    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell(f"echo 'Hello {random_string}!'")


def test_execute_shell_allowlist_should_allow(
    code_executor_component: CodeExecutorComponent, random_string: str
):
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]

    result = code_executor_component.execute_shell(f"echo 'Hello {random_string}!'")
    assert "Hello" in result and random_string in result


@pytest.mark.parametrize(
    "command_line",
    [
        "rm -rf workspace",  # plain name (already covered, baseline)
        "/bin/rm -rf workspace",  # absolute path prefix
        "/usr/bin/rm -rf workspace",  # different absolute path
        "./rm -rf workspace",  # relative path prefix
        "  /bin/rm -rf workspace",  # leading whitespace padding
    ],
)
def test_execute_shell_denylist_blocks_path_prefixed_command(
    code_executor_component: CodeExecutorComponent, command_line: str
):
    """A denied command must stay denied regardless of path-prefix spelling.

    Regression test for the first-token-only denylist bypass: previously the
    denylist was matched against the raw first token only, so "/bin/rm" slipped
    past a denylist of ["rm"]. validate_command must now reject every spelling of
    the denied program, and execute_shell must raise before running anything.
    """
    code_executor_component.config.shell_command_control = "denylist"
    code_executor_component.config.shell_denylist = ["rm"]

    allow_execute, allow_shell = code_executor_component.validate_command(command_line)
    assert allow_execute is False, f"{command_line!r} should be denied"
    assert allow_shell is False

    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell(command_line)


def test_execute_shell_denylist_still_allows_non_denied_command(
    code_executor_component: CodeExecutorComponent, random_string: str
):
    """Basename matching must not over-block unrelated commands."""
    code_executor_component.config.shell_command_control = "denylist"
    code_executor_component.config.shell_denylist = ["rm"]

    # "echo" shares no basename with "rm" and must still run.
    result = code_executor_component.execute_shell(f"echo 'Hello {random_string}!'")
    assert "Hello" in result and random_string in result

    # A path-prefixed allowed command is also permitted by validate_command.
    allow_execute, _ = code_executor_component.validate_command("/bin/echo hi")
    assert allow_execute is True


def test_validate_command_unbalanced_quotes_fails_closed(
    code_executor_component: CodeExecutorComponent,
):
    """An unparsable command line must fail closed instead of raising."""
    code_executor_component.config.shell_command_control = "denylist"
    code_executor_component.config.shell_denylist = ["rm"]

    # Unbalanced quote makes shlex.split raise ValueError; must be denied.
    allow_execute, allow_shell = code_executor_component.validate_command(
        'echo "unterminated'
    )
    assert allow_execute is False
    assert allow_shell is False
