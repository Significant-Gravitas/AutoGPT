from pathlib import Path

import pytest


@pytest.fixture()
def test_workspace(tmp_path: Path) -> Path:
    return tmp_path
