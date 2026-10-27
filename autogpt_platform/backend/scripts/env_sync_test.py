import importlib

import pytest
from dotenv import dotenv_values


def test_sync_preserves_other_values_and_literal_secrets(tmp_path):
    sync = importlib.import_module("scripts.env_sync")
    target = tmp_path / ".env"
    original = "PORT=8123\n# keep this comment\nOTHER='stay'\n"
    target.write_text(original)
    value = "literal$VALUE#with'quotes"
    sync.apply_updates(target, {"MICROSOFT_CLIENT_SECRET": value})
    assert target.read_text().startswith(original)
    assert dotenv_values(target, interpolate=False)["MICROSOFT_CLIENT_SECRET"] == value
    assert target.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.glob(".env.backup-*"))[0].read_text() == original


def test_conflicting_sources_do_not_expose_values():
    sync = importlib.import_module("scripts.env_sync")
    with pytest.raises(ValueError) as error:
        sync.select_values(
            [
                {"MICROSOFT_CLIENT_SECRET": "secret-one"},
                {"MICROSOFT_CLIENT_SECRET": "secret-two"},
            ],
            ["MICROSOFT_CLIENT_SECRET"],
        )
    assert "MICROSOFT_CLIENT_SECRET" in str(error.value)
    assert "secret-one" not in str(error.value)
    assert "secret-two" not in str(error.value)


def test_empty_source_values_are_not_updates():
    sync = importlib.import_module("scripts.env_sync")
    assert (
        sync.select_values(
            [{"MICROSOFT_CLIENT_SECRET": ""}], ["MICROSOFT_CLIENT_SECRET"]
        )
        == {}
    )
