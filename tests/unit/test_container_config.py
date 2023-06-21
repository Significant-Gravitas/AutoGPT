import sys
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path

from autogpt.config.container_config import ContainerConfig

@pytest.fixture
def container_config():
    return ContainerConfig()

def test_prefs_file_exists(container_config):
    assert container_config._prefs_file.exists()

def test_is_docker(container_config):
    if Path('/.dockerenv').exists():
        assert container_config.is_docker() == True
    else:
        assert container_config.is_docker() == False

def test_is_virtual_env(container_config):
    expected_result = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    assert container_config.is_virtual_env() == expected_result

def test_save_prefs(container_config):
    initial_prefs = container_config._prefs
    container_config._prefs = 1
    container_config.save_prefs()
    container_config._prefs = 2
    container_config.save_prefs()

    assert int(container_config._prefs_file.read_text()) == 2

    container_config._prefs_file.write_text(str(initial_prefs))

def test_reset_prefs(container_config):
    initial_prefs = container_config._prefs
    container_config._prefs = 2
    container_config.save_prefs()
    container_config.reset_prefs()

    assert container_config._prefs is None

    container_config._prefs_file.write_text(str(initial_prefs))

# Helper function to create temporary directories for testing
@pytest.fixture
def temp_dirs(tmpdir_factory):
    src_dir = tmpdir_factory.mktemp("src")
    dest_dir = tmpdir_factory.mktemp("docker")
    return src_dir, dest_dir

# Test cases for _copy_file_to_docker method
def test_copy_file_to_docker_src_not_exists(temp_dirs):
    src, dest = temp_dirs
    obj = ContainerConfig()
    non_existent_file = src.join("non_existent.txt")
    assert not obj._copy_file_to_docker(non_existent_file, dest.join("dest.txt"))

def test_copy_file_to_docker_copy_directory(temp_dirs):
    src, dest = temp_dirs
    obj = ContainerConfig()
    test_dir = src.mkdir("test_dir")
    test_dir.join("test_file.txt").write("sample_content")
    assert obj._copy_file_to_docker(test_dir, dest.join("test_dir"))
    assert dest.join("test_dir", "test_file.txt").read() == "sample_content"

def test_copy_file_to_docker_copy_single_file(temp_dirs):
    src, dest = temp_dirs
    obj = ContainerConfig()
    test_file = src.join("test_file.txt")
    test_file.write("sample_content")
    assert obj._copy_file_to_docker(test_file, dest.join("test_file.txt"))
    assert dest.join("test_file.txt").read() == "sample_content"

@patch("autogpt.config.container_config.ContainerConfig._copy_file_to_docker")
def test_copy_files_to_docker_success(copy_mock, temp_dirs):
    obj = ContainerConfig()
    obj._docker_dir = temp_dirs[1]

    copy_mock.return_value = True
    FILES_TO_COPY = {"dest.txt": ["src.txt"]}

    with patch.dict("autogpt.config.container_config.DOCKER_FILES_TO_COPY", FILES_TO_COPY, clear=True):
        obj._copy_files_to_docker(temp_dirs[0])

    print(copy_mock.call_args_list)
    copy_mock.assert_called_once_with(temp_dirs[0].join("src.txt"), temp_dirs[1].join("dest.txt"))

@patch("autogpt.config.container_config.ContainerConfig._copy_file_to_docker")
@patch("requests.get")
def test_copy_files_to_docker_download_success(get_mock, copy_mock, temp_dirs):
    obj = ContainerConfig()
    obj._docker_dir = temp_dirs[1]

    copy_mock.return_value = False
    FILES_TO_COPY = {"dest.txt": ["src.txt"]}
    response = MagicMock(status_code=200, text="downloaded_content")
    get_mock.return_value = response

    with patch.dict("autogpt.config.container_config.DOCKER_FILES_TO_COPY", FILES_TO_COPY, clear=True):
        obj._copy_files_to_docker(temp_dirs[0])

    copy_mock.assert_called_once_with(temp_dirs[0].join("src.txt"), temp_dirs[1].join("dest.txt"))
    get_mock.assert_called_once_with("https://raw.githubusercontent.com/Significant-Gravitas/Auto-GPT/stable/src.txt")
    assert temp_dirs[1].join("dest.txt").read() == "downloaded_content"
