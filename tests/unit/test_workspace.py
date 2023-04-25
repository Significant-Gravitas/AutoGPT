import shutil
import tempfile
from pathlib import Path

# make temporary directory to use as the workspace
temp_dir = tempfile.mkdtemp()
WORKSPACE_PATH = Path(temp_dir) / "auto_gpt_workspace"
os.makedirs(WORKSPACE_PATH)

def test_path_in_workspace():
    # test relative path
    relative_path = "data/input.txt"
    expected_path = WORKSPACE_PATH / "data/input.txt"
    assert path_in_workspace(relative_path) == expected_path

    # test absolute path
    absolute_path = Path("/home/user/data/input.txt")
    expected_path = WORKSPACE_PATH / "home/user/data/input.txt"
    assert path_in_workspace(absolute_path) == expected_path

def test_safe_path_join():
    # test  valid path
    base_path = WORKSPACE_PATH / "data"
    joined_path = safe_path_join(base_path, "input.txt")
    expected_path = WORKSPACE_PATH / "data/input.txt"
    assert joined_path == expected_path

    # test  invalid path
    base_path = WORKSPACE_PATH / "data"
    invalid_path = WORKSPACE_PATH.parent / "data/input.txt"
    try:
        safe_path_join(base_path, invalid_path)
    except ValueError as e:
        assert str(e) == f"Attempted to access path '{invalid_path}' outside of working directory '{WORKSPACE_PATH}'."
    else:
        assert False, "Expected a ValueError to be raised"

# clean the temporary directory
shutil.rmtree(temp_dir)
