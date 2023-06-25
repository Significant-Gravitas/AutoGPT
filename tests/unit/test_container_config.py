import os
import shutil
import subprocess
import sys
import venv
import pytest
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

from autogpt.config.container_config import ContainerConfig

@pytest.fixture
def container_config() -> ContainerConfig:
    return ContainerConfig(
        image_name="test_image",
        repo="test/repo",
        branch_or_tag="test-branch",
        rebuild_image=False,
        pull_image=False,
        reinstall=False,
        interactive=False,
        allow_virtualenv=False,
        args=[],
    )

def test_running_in_docker(container_config):
    assert container_config.running_in_docker() == Path("/.dockerenv").exists()

def test_running_in_virtual_env(container_config):
    assert (
        container_config.running_in_virtual_env()
        == (hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
    )
    
def test_running_in_docker_true(monkeypatch, container_config):
    monkeypatch.setattr(Path, "exists", lambda x: True)
    assert container_config.running_in_docker() is True

def test_running_in_docker_false(monkeypatch, container_config):
    monkeypatch.setattr(Path, "exists", lambda x: False)
    assert container_config.running_in_docker() is False

def test_save_prefs(container_config):
    container_config.prefs = 2
    container_config.save_prefs()
    assert container_config.prefs_file.read_text() == "2"

def test_reset_prefs(container_config):
    container_config.reset_prefs()
    assert container_config.prefs is None
    assert not container_config.prefs_file.exists()

class MockContainerConfig(ContainerConfig):
    running_in_docker = MagicMock()
    allow_virtualenv = MagicMock()
    check_docker_is_installed = MagicMock()
    show_selections = MagicMock()
    run_reinstall = MagicMock()
    run_rebuild_image = MagicMock()
    run_pull_image = MagicMock()
    run_in_docker = MagicMock()


@pytest.fixture
def mock_container_config():
    return MockContainerConfig(
        image_name="test_image",
        repo="test/repo",
        branch_or_tag="test-branch",
        rebuild_image=False,
        pull_image=False,
        reinstall=False,
        interactive=False,
        allow_virtualenv=False,
        args=[],
    )


def test_run_in_docker_container(mock_container_config):
    mock_container_config.running_in_docker.return_value = True

    mock_container_config.run()

    mock_container_config.running_in_docker.assert_called_once_with()
    mock_container_config.show_selections.assert_not_called()


def test_run_with_virtualenv_allowed(mock_container_config):
    mock_container_config.running_in_docker.return_value = False
    mock_container_config.allow_virtualenv = True
    mock_container_config.run()
    mock_container_config.show_selections.assert_called_once_with()

def test_run_with_docker_not_installed(mock_container_config):
    mock_container_config.running_in_docker.return_value = False
    mock_container_config.allow_virtualenv = False
    mock_container_config.check_docker_is_installed.return_value = False

    with pytest.raises(AssertionError, match="Docker is not installed. Please install Docker and try again."):
        mock_container_config.run()

    mock_container_config.run_reinstall.assert_not_called()
    mock_container_config.run_rebuild_image.assert_not_called()
    mock_container_config.run_pull_image.assert_not_called()
    mock_container_config.run_in_docker.assert_not_called()


def test_install_docker(container_config, monkeypatch):
    # Test for each type of OS
    os_type_windows = "win32"
    os_type_mac = "darwin"
    os_type_linux = "linux"
    monkeypatch.setattr(subprocess, "run", MagicMock())
    monkeypatch.setattr(container_config, "docker_is_installed", MagicMock(return_value=False))
    monkeypatch.setattr(container_config, "_install_docker_windows", MagicMock())
    monkeypatch.setattr(container_config, "_install_docker_mac", MagicMock())
    monkeypatch.setattr(container_config, "_install_docker_linux", MagicMock())
    monkeypatch.setattr("sys.platform", os_type_windows)
    container_config.install_docker()
    container_config._install_docker_windows.assert_called()
    monkeypatch.setattr("sys.platform", os_type_mac)
    container_config.install_docker()
    container_config._install_docker_mac.assert_called()
    monkeypatch.setattr("sys.platform", os_type_linux)
    container_config.install_docker()
    container_config._install_docker_linux.assert_called()
    
def test_copy_config_files(container_config, monkeypatch):
    file_list = ["file1", "file2", "file3"]
    container_config.docker_config_dir =Path("/dummy_dest")
    monkeypatch.setattr(container_config, "copy_file", MagicMock())
    container_config.copy_config_files(Path("/dummy_cwd"), file_list)
    assert container_config.copy_file.call_count == len(file_list)

def test_copy_file(container_config, monkeypatch):
    src = Path("/src_path")
    dest = Path("/dest_path")
    monkeypatch.setattr(Path, "is_dir", lambda x: False)
    monkeypatch.setattr(Path, "exists", lambda x: True)
    monkeypatch.setattr(Path, "is_file", lambda x: True)
    monkeypatch.setattr(shutil, "copy", lambda x, y: dest)
    assert container_config.copy_file(src, dest) is True

def test_get_files_base_url(container_config, monkeypatch):
    base_url = container_config.get_files_base_url()
    assert base_url == f"https://raw.githubusercontent.com/{container_config.repo}/{container_config.branch_or_tag}/"

def test_run_reinstall(container_config, monkeypatch):
    monkeypatch.setattr(container_config, "running_in_docker", MagicMock(return_value=False))
    monkeypatch.setattr(container_config, "init_docker_config", MagicMock())
    docker_config_dir = MagicMock()
    docker_config_dir.exists.return_value = False
    monkeypatch.setattr(container_config, "docker_config_dir", docker_config_dir)
    monkeypatch.setattr(shutil, "copytree", MagicMock())
    config_dir = MagicMock()
    config_dir.unlink = MagicMock()
    monkeypatch.setattr(container_config, "config_dir", config_dir)
    monkeypatch.setattr(subprocess, "run", MagicMock())
    monkeypatch.setattr("os.chdir", MagicMock())
    monkeypatch.setattr(sys, "exit", MagicMock())
    container_config.run_reinstall()
    container_config.init_docker_config.assert_called()

def test_run_pull_image(container_config, monkeypatch):
    monkeypatch.setattr(container_config, "init_docker_config", MagicMock())
    monkeypatch.setattr(subprocess, "run", MagicMock())
    monkeypatch.setattr("os.chdir", MagicMock())
    container_config.run_pull_image()
    container_config.init_docker_config.assert_called()

def test_run_rebuild_image(container_config, monkeypatch):
    monkeypatch.setattr(container_config, "init_docker_config", MagicMock())
    monkeypatch.setattr(subprocess, "run", MagicMock())
    monkeypatch.setattr("os.chdir", MagicMock())
    container_config.run_rebuild_image()
    container_config.init_docker_config.assert_called()

def test_init_docker_config(container_config, monkeypatch):
    docker_config_dir = MagicMock()
    docker_config_dir.exists.return_value = False
    monkeypatch.setattr(container_config, "docker_config_dir", docker_config_dir)
    monkeypatch.setattr(container_config, "copy_config_files", MagicMock())
    container_config.init_docker_config()
    container_config.copy_config_files.assert_called()

def test_run_in_docker(container_config, monkeypatch):
    monkeypatch.setattr(container_config, "running_in_docker", MagicMock(return_value=False))
    monkeypatch.setattr(container_config, "docker_is_installed", MagicMock(return_value=True))
    monkeypatch.setattr(container_config, "init_docker_config", MagicMock())
    monkeypatch.setattr("os.chdir", MagicMock())
    monkeypatch.setattr(sys, "exit", MagicMock())
    monkeypatch.setattr(subprocess, "run", MagicMock())
    container_config.run_in_docker()
    container_config.init_docker_config.assert_called()

def test_run_in_virtual_env(container_config, monkeypatch):
    monkeypatch.setattr(container_config, "running_in_virtual_env", MagicMock(return_value=False))
    assert not container_config.running_in_virtual_env()
    venv_dir = MagicMock()
    venv_dir.exists.return_value = False
    monkeypatch.setattr(container_config, "venv_dir", venv_dir)
    monkeypatch.setattr(venv.EnvBuilder, "create", MagicMock())
    monkeypatch.setattr(subprocess, "run", MagicMock())
    monkeypatch.setattr(sys, "exit", MagicMock())
    container_config.run_in_virtual_env()
    venv.EnvBuilder.create.assert_called()