import datetime
import os

from unittest.mock import MagicMock

from gpt_engineer.core.db import DB, DBs, archive


def freeze_at(monkeypatch, time):
    datetime_mock = MagicMock(wraps=datetime.datetime)
    datetime_mock.now.return_value = time
    monkeypatch.setattr(datetime, "datetime", datetime_mock)


def setup_dbs(tmp_path, dir_names):
    directories = [tmp_path / name for name in dir_names]

    # Create DB objects
    dbs = [DB(dir) for dir in directories]

    # Create DBs instance
    return DBs(*dbs)


def test_archive(tmp_path, monkeypatch):
    gpteng_dir = ".gpteng"

    dbs = setup_dbs(
        tmp_path,
        [
            gpteng_dir + "/memory",
            gpteng_dir + "/logs",
            gpteng_dir + "/preprompts",
            gpteng_dir + "/input",
            "",  # workspace is top-level folder
            gpteng_dir + "/archive",
            gpteng_dir + "/project_metadata",
        ],
    )
    freeze_at(monkeypatch, datetime.datetime(2020, 12, 25, 17, 5, 55))
    archive(dbs)
    assert not os.path.exists(tmp_path / gpteng_dir / "memory")
    assert os.path.isdir(tmp_path / gpteng_dir / "archive" / "20201225_170555")

    dbs = setup_dbs(
        tmp_path,
        [
            gpteng_dir + "/memory",
            gpteng_dir + "/logs",
            gpteng_dir + "/preprompts",
            gpteng_dir + "/input",
            "",  # workspace is top-level folder
            gpteng_dir + "/archive",
            gpteng_dir + "/project_metadata",
        ],
    )
    freeze_at(monkeypatch, datetime.datetime(2022, 8, 14, 8, 5, 12))
    archive(dbs)
    assert not os.path.exists(tmp_path / "memory")
    assert os.path.isdir(tmp_path / gpteng_dir / "archive" / "20201225_170555")
    assert os.path.isdir(tmp_path / gpteng_dir / "archive" / "20220814_080512")
