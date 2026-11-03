from collections.abc import Iterator
from pathlib import Path

import pytest

from backend.util import architecture_test as rules


@pytest.fixture
def isolated_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(rules, "BACKEND_ROOT", tmp_path)
    rules._global_clock_patches.cache_clear()
    try:
        yield tmp_path
    finally:
        rules._global_clock_patches.cache_clear()


@pytest.mark.parametrize(
    "source",
    [
        "import dataclasses as dc",
        "import ｄａｔａｃｌａｓｓｅｓ as dc",
        "from dataclasses \\\n import dataclass",
    ],
    ids=["alias", "unicode-identifier", "continued-import"],
)
def test_dataclass_rule_reads_current_source(isolated_backend: Path, source: str):
    path = isolated_backend / "example.py"
    path.write_text("value = 1\n", encoding="utf-8")
    rules.test_backend_uses_pydantic_not_dataclasses()

    path.write_text(source, encoding="utf-8")
    with pytest.raises(AssertionError, match="dataclasses"):
        rules.test_backend_uses_pydantic_not_dataclasses()

    path.write_text("value = 2\n", encoding="utf-8")
    rules.test_backend_uses_pydantic_not_dataclasses()
    path.unlink()
    rules.test_backend_uses_pydantic_not_dataclasses()


@pytest.mark.parametrize("name", ["cached", "ｃａｃｈｅｄ"])
def test_cached_client_rule_reads_unicode_names(isolated_backend: Path, name: str):
    (isolated_backend / "example.py").write_text(
        f"@{name}()\ndef client() -> AsyncClient: pass\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="Process-wide @cached"):
        rules.test_no_process_cached_loop_bound_clients()


@pytest.mark.parametrize(
    "source",
    [
        'from unittest.mock import patch as p\np("time.sleep")',
        'from unittest.mock import ｐａｔｃｈ as p\np("time.sleep")',
        'import time\nsetattr(time, "sleep", lambda: None)',
        'import time\nｓｅｔａｔｔｒ(time, "sleep", lambda: None)',
    ],
    ids=["patch-alias", "unicode-patch", "setattr", "unicode-setattr"],
)
def test_clock_rule_reads_aliased_and_unicode_names(
    isolated_backend: Path, source: str
):
    (isolated_backend / "example.py").write_text(source, encoding="utf-8")
    with pytest.raises(AssertionError, match="time.sleep"):
        rules.test_no_global_clock_or_sleep_patches()
