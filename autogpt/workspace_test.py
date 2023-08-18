import os

import pytest

# Assuming the classes are defined in a file named workspace.py
from .workspace import LocalWorkspace

# Constants
TEST_BASE_PATH = "/tmp/test_workspace"
TEST_FILE_CONTENT = b"Hello World"


# Setup and Teardown for LocalWorkspace


@pytest.fixture
def setup_local_workspace():
    os.makedirs(TEST_BASE_PATH, exist_ok=True)
    yield
    os.system(f"rm -rf {TEST_BASE_PATH}")  # Cleanup after tests


def test_local_read_write_delete_exists(setup_local_workspace):
    workspace = LocalWorkspace(TEST_BASE_PATH)

    # Write
    workspace.write("test_file.txt", TEST_FILE_CONTENT)

    # Exists
    assert workspace.exists("test_file.txt")

    # Read
    assert workspace.read("test_file.txt") == TEST_FILE_CONTENT

    # Delete
    workspace.delete("test_file.txt")
    assert not workspace.exists("test_file.txt")


def test_local_list(setup_local_workspace):
    workspace = LocalWorkspace(TEST_BASE_PATH)
    workspace.write("test1.txt", TEST_FILE_CONTENT)
    workspace.write("test2.txt", TEST_FILE_CONTENT)

    files = workspace.list(".")
    assert set(files) == set(["test1.txt", "test2.txt"])
