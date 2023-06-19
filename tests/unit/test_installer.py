import pytest
from unittest.mock import MagicMock, patch

from autogpt.install.installer import Installer

@pytest.fixture
def installer():
    return Installer()

def test_is_installed_positive():
    with patch("shutil.which") as mock_which:
        mock_which.return_value = "autogpt"
        assert Installer.is_installed() is True

def test_is_installed_negative():
    with patch("shutil.which") as mock_which:
        mock_which.return_value = None
        assert Installer.is_installed() is False

def test_init(installer):
    assert installer._prefs_file.exists() == (installer._prefs is not None)

def test_run_not_installed(yes_no_input):
    with patch("autogpt.install.installer.prompt", yes_no_input("2")):
        with patch("autogpt.install.installer.Installer.is_installed", MagicMock(return_value=False)):
            installer = Installer()
            installer.reset_prefs()

            installer.run(interactive=True)
            assert installer._prefs == 2

@pytest.mark.parametrize("prefs_value", [1, 2, 4, None])
def test_run_installed(prefs_value, installer):
    with patch("autogpt.install.installer.Installer.is_installed", return_value=True):
        installer._prefs = prefs_value
        installer.run(interactive=True)


def test_prompt_for_prefs(monkeypatch, installer):
    monkeypatch.setattr("autogpt.install.installer.prompt", lambda _: "1")
    prefs = installer.prompt_for_prefs()
    assert prefs == 1
     
def test_install(installer):
    with patch("subprocess.run") as mock_run:
        installer._prefs = 1
        installer.install()
        mock_run.assert_called_once_with(["pip", "install", "autogpt"])

def test_save_prefs(installer):
    installer._prefs = 2
    installer.save_prefs()
    assert int(installer._prefs_file.read_text()) == 2

def test_reset_prefs(installer):
    installer.reset_prefs()
    assert not installer._prefs_file.exists()
    assert installer._prefs is None

@pytest.fixture
def yes_no_input():
    def _yes_no_input_factory(value):
        return lambda _: value

    return _yes_no_input_factory
