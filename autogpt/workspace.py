import os
import shutil
import tempfile
from pathlib import Path

# make temporary directory to use as the workspace
temp_dir = tempfile.mkdtemp()
WORKSPACE_PATH = Path(temp_dir) / "auto_gpt_workspace"
os.makedirs(WORKSPACE_PATH)

def path_in_workspace(input_path):
    input_path = Path(input_path)
    if input_path.is_absolute():
        input_path = input_path.relative_to(input_path.anchor)

    expected_path = WORKSPACE_PATH / input_path
    return expected_path

def test_path_in_workspace():
    # test relative path
    relative_path = "data/input.txt"
    expected_path = WORKSPACE_PATH / "data/input.txt"
    assert path_in_workspace(relative_path) == expected_path

    # test absolute path
    absolute_path = Path("/home/user/data/input.txt")
    expected_path = WORKSPACE_PATH / "home/user/data/input.txt"
    assert path_in_workspace(absolute_path) == expected_path

# testing the function
test_path_in_workspace()


