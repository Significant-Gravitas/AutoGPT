import os
import shutil
from pathlib import Path
import tempfile

from workspace import path_in_workspace, safe_path_join, is_path_within_workspace

def test_path_in_workspace():
    temp_dir = tempfile.mkdtemp()
    workspace_path = Path(temp_dir) / "auto_gpt_workspace"
    os.makedirs(workspace_path)

    # Test relative path
    relative_path = "data/input.txt"
    expected_path = workspace_path / "data/input.txt"
    assert path_in_workspace(relative_path) == expected_path

    # Test Path object
    relative_path = Path("data/input.txt")
    expected_path = workspace_path / "data/input.txt"
    assert path_in_workspace(relative_path) == expected_path

def test_safe_path_join():
    base_path = Path("/home/user/workspace")

    # Test joining relative path
    joined_path = safe_path_join(base_path, "data/input.txt")
    expected_path = base_path / "data/input.txt"
    assert joined_path == expected_path

    # Test joining Path object
    joined_path = safe_path_join(base_path, Path("data/input.txt"))
    expected_path = base_path / "data/input.txt"
    assert joined_path == expected_path

def test_is_path_within_workspace():
    base_path = Path("/home/user/workspace")

    # Test path within workspace
    path = base_path / "data/input.txt"
    assert is_path_within_workspace(base_path, path)

    # Test path outside workspace
    path = Path("/home/user/another_directory/data/input.txt")
    assert not is_path_within_workspace(base_path, path)

# Run tests
test_path_in_workspace()
test_safe_path_join()
test_is_path_within_workspace()
