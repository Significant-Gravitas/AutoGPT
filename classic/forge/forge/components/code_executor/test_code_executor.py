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


# SECURITY TESTS: Command Injection Prevention
def test_command_injection_semicolon_blocked(code_executor_component: CodeExecutorComponent):
    """Test that command injection via semicolon is blocked"""
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    
    # This should be blocked due to semicolon
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("echo hello; rm -rf /")


def test_command_injection_pipe_blocked(code_executor_component: CodeExecutorComponent):
    """Test that command injection via pipe is blocked"""
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    
    # This should be blocked due to pipe
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("echo hello | cat /etc/passwd")


def test_command_injection_ampersand_blocked(code_executor_component: CodeExecutorComponent):
    """Test that command injection via ampersand is blocked"""
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    
    # This should be blocked due to ampersand
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("echo hello & rm important_file")


def test_command_injection_command_substitution_blocked(code_executor_component: CodeExecutorComponent):
    """Test that command substitution is blocked"""
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    
    # This should be blocked due to command substitution
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("echo $(whoami)")


def test_command_injection_backticks_blocked(code_executor_component: CodeExecutorComponent):
    """Test that backtick command substitution is blocked"""
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    
    # This should be blocked due to backticks
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("echo `whoami`")


def test_command_injection_redirection_blocked(code_executor_component: CodeExecutorComponent):
    """Test that file redirection is blocked"""
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    
    # This should be blocked due to redirection
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("echo secret > /tmp/stolen_data")


def test_malformed_command_blocked(code_executor_component: CodeExecutorComponent):
    """Test that malformed shell commands are blocked"""
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    
    # This should be blocked due to malformed syntax
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("echo 'unclosed quote")


def test_invalid_shell_command_control_blocked(code_executor_component: CodeExecutorComponent):
    """Test that invalid shell_command_control values default to secure mode"""
    # Manually set an invalid value (bypassing type checking)
    code_executor_component.config.shell_command_control = "invalid_mode"
    
    # This should be blocked due to invalid control mode
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("echo hello")


def test_empty_command_blocked(code_executor_component: CodeExecutorComponent):
    """Test that empty commands are blocked"""
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    
    # This should be blocked due to empty command
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("")


def test_whitespace_only_command_blocked(code_executor_component: CodeExecutorComponent):
    """Test that whitespace-only commands are blocked"""
    code_executor_component.config.shell_command_control = "allowlist"
    code_executor_component.config.shell_allowlist = ["echo"]
    
    # This should be blocked due to whitespace-only command
    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        code_executor_component.execute_shell("   ")


# NOTE: Command injection tests are limited to validation logic due to automated environment limits
# Manual testing recommended for: Full subprocess execution, shell metacharacter handling in live environments
