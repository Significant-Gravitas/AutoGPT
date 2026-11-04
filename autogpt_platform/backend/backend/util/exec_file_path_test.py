"""Containment tests for :func:`backend.util.file.get_exec_file_path`.

Every path handed to ``get_exec_file_path()`` is caller-supplied and must stay
inside ``{temp}/exec_file/{graph_exec_id}``. See GitHub issue #14622.
"""

from pathlib import Path

import pytest

from backend.util import file as file_module
from backend.util.file import clean_exec_files, get_exec_file_path

EXEC_ID = "e2d4f1aa-0000-4000-8000-000000000001"


@pytest.fixture
def exec_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point TEMP_DIR at *tmp_path* and yield the execution's own folder."""
    monkeypatch.setattr(file_module, "TEMP_DIR", tmp_path.resolve())
    base = tmp_path.resolve() / "exec_file" / EXEC_ID
    base.mkdir(parents=True)
    return base


class TestGetExecFilePathRejectsEscapes:
    """Paths that resolve outside the execution folder are refused."""

    def test_rejects_parent_traversal(self, exec_base: Path):
        with pytest.raises(ValueError):
            get_exec_file_path(EXEC_ID, "../../outside.txt")

    def test_rejects_traversal_below_a_subdirectory(self, exec_base: Path):
        with pytest.raises(ValueError):
            get_exec_file_path(EXEC_ID, "nested/../../../outside.txt")

    def test_rejects_absolute_path(self, exec_base: Path):
        with pytest.raises(ValueError):
            get_exec_file_path(EXEC_ID, "/etc/passwd")

    def test_rejects_symlink_pointing_out_of_the_base(
        self, exec_base: Path, tmp_path: Path
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "secret.txt").write_text("outside the sandbox")
        (exec_base / "link").symlink_to(elsewhere, target_is_directory=True)

        with pytest.raises(ValueError):
            get_exec_file_path(EXEC_ID, "link/secret.txt")

    def test_rejects_sibling_folder_sharing_the_base_name_as_prefix(
        self, exec_base: Path
    ):
        """``/tmp/exec_file/<id>_evil`` is not inside ``/tmp/exec_file/<id>``."""
        (exec_base.parent / f"{EXEC_ID}_evil").mkdir()

        with pytest.raises(ValueError):
            get_exec_file_path(EXEC_ID, f"../{EXEC_ID}_evil/loot.txt")

    def test_rejects_exec_id_that_leaves_the_exec_file_root(self, exec_base: Path):
        with pytest.raises(ValueError):
            get_exec_file_path("../..", "loot.txt")

    def test_clean_exec_files_does_not_remove_anything_outside_the_base(
        self, exec_base: Path, tmp_path: Path
    ):
        unrelated = tmp_path / "important"
        unrelated.mkdir()
        (unrelated / "data.txt").write_text("unrelated data")

        with pytest.raises(ValueError):
            clean_exec_files(EXEC_ID, "../..")

        assert (unrelated / "data.txt").is_file()


class TestGetExecFilePathAllowsOrdinaryPaths:
    """Normal relative paths keep resolving to the same place they always did."""

    def test_allows_plain_filename(self, exec_base: Path):
        assert get_exec_file_path(EXEC_ID, "report.csv") == str(
            exec_base / "report.csv"
        )

    def test_allows_dots_inside_the_filename(self, exec_base: Path):
        assert get_exec_file_path(EXEC_ID, "report.2026.csv") == str(
            exec_base / "report.2026.csv"
        )

    def test_allows_nested_subdirectories(self, exec_base: Path):
        assert get_exec_file_path(EXEC_ID, "nested/dir/report.csv") == str(
            exec_base / "nested" / "dir" / "report.csv"
        )

    def test_allows_current_directory_prefix(self, exec_base: Path):
        assert get_exec_file_path(EXEC_ID, "./report.csv") == str(
            exec_base / "report.csv"
        )

    def test_allows_traversal_that_stays_inside_the_base(self, exec_base: Path):
        assert get_exec_file_path(EXEC_ID, "nested/../report.csv") == str(
            exec_base / "report.csv"
        )

    def test_empty_path_returns_the_base_folder(self, exec_base: Path):
        assert get_exec_file_path(EXEC_ID, "") == str(exec_base)

    def test_allows_absolute_path_already_inside_the_base(self, exec_base: Path):
        inside = exec_base / "report.csv"
        assert get_exec_file_path(EXEC_ID, str(inside)) == str(inside)

    def test_clean_exec_files_removes_the_execution_folder(self, exec_base: Path):
        (exec_base / "report.csv").write_text("data")

        clean_exec_files(EXEC_ID)

        assert not exec_base.exists()
