import inspect
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from backend.blocks import load_all_blocks


def test_block_discovery_does_not_import_tests(tmp_path: Path):
    paths = [
        "provider/action.py",
        "provider/_config.py",
        "provider/action_test.py",
        "provider/test_action.py",
        "provider/conftest.py",
        "provider/tests/helpers.py",
        "test/helpers.py",
        "provider/__init__.py",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    with (
        patch("backend.blocks.__file__", str(tmp_path / "__init__.py")),
        patch("backend.blocks._all_subclasses", return_value=[]),
        patch("backend.blocks.importlib.import_module") as import_module,
    ):
        inspect.unwrap(load_all_blocks)()
    assert {call.args[0] for call in import_module.call_args_list} == {
        ".provider.action",
        ".provider._config",
    }


def test_mem0_sdk_is_loaded_only_when_creating_a_client():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from types import ModuleType
from unittest.mock import Mock, patch
from backend.blocks.mem0 import Mem0Base, TEST_CREDENTIALS

assert "mem0" not in sys.modules
module = ModuleType("mem0")
module.MemoryClient = Mock()
with patch.dict(sys.modules, {"mem0": module}):
    client = Mem0Base._get_client(TEST_CREDENTIALS)
assert client is module.MemoryClient.return_value
module.MemoryClient.assert_called_once_with(
    api_key=TEST_CREDENTIALS.api_key.get_secret_value()
)
""",
        ],
        check=True,
    )
