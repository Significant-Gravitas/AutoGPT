import sys
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
