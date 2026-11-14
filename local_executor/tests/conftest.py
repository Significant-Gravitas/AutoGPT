from __future__ import annotations

import pytest

from autogpt_local_executor.computer_use import concealed


@pytest.fixture(autouse=True)
def _clipboard_tools_looked_up_afresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have every test look wl-paste and xclip up for itself: the shim keeps
    their paths once found (concealed._clipboard_tool), and the path one
    test's stand-in for shutil.which gave would stand in for the next's."""
    monkeypatch.setattr(concealed, "_clipboard_tool_paths", {})


@pytest.fixture(autouse=True)
def _away_from_the_real_file_keys(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep every test from the file fallback of whoever runs the tests: its
    audit and recording keys, and their lock, in AUTOGPT_SHIM_FALLBACK_DIR,
    by default ~/.autogpt-local-executor, and the passphrase that reads
    them, which their shell may export. A test that reached them used a real
    key in place of the one it set up, and failed, or wrote to their home.
    The folder is a new one of the test's own, outside its tmp_path, which
    tests list. A test that needs either variable sets it."""
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path_factory.mktemp("fallback")))
    monkeypatch.delenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", raising=False)
