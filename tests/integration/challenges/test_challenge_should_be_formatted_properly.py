import importlib.util
import inspect
import os
from types import ModuleType
from typing import List

# Path to the challenges folder
CHALLENGES_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../challenges"
)


def get_python_files(directory: str, exclude_file: str) -> List[str]:
    """Recursively get all python files in a directory and subdirectories."""
    python_files: List[str] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (
                file.endswith(".py")
                and file.startswith("test_")
                and file != exclude_file
            ):
                python_files.append(os.path.join(root, file))
    return python_files


def load_module_from_file(test_file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("module.name", test_file)
    assert spec is not None, f"Unable to get spec for module in file {test_file}"
    module = importlib.util.module_from_spec(spec)
    assert (
        spec.loader is not None
    ), f"Unable to get loader for module in file {test_file}"
    spec.loader.exec_module(module)
    return module


def get_test_functions(module: ModuleType) -> List:
    return [
        o
        for o in inspect.getmembers(module)
        if inspect.isfunction(o[1]) and o[0].startswith("test_")
    ]


def assert_single_test_function(functions_list: List, test_file: str) -> None:
    assert len(functions_list) == 1, f"{test_file} should contain only one function"
    assert (
        functions_list[0][0][5:] == os.path.basename(test_file)[5:-3]
    ), f"The function in {test_file} should have the same name as the file without 'test_' prefix"


def test_method_name_and_count() -> None:
    current_file: str = os.path.basename(__file__)
    test_files: List[str] = get_python_files(CHALLENGES_DIR, current_file)
    for test_file in test_files:
        module = load_module_from_file(test_file)
        functions_list = get_test_functions(module)
        assert_single_test_function(functions_list, test_file)
