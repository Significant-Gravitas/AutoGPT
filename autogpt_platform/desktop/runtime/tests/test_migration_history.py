import hashlib

import pytest

from autogpt_desktop import migrations
from autogpt_desktop.layout import Bundle


class Connection:
    def __init__(self, rows, exists=True):
        self.rows, self.exists = rows, exists
        self.statements = []
        self.closed = False

    def cursor(self):
        return self

    def execute(self, statement):
        self.statements.append(statement)

    def fetchone(self):
        return (self.exists,)

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


@pytest.fixture
def history(tmp_path):
    bundle = Bundle(tmp_path)
    path = bundle.backend_dir / "migrations/001_initial/migration.sql"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"CREATE TABLE example (id text);\n")
    return bundle, path


def test_completed_migration_missing_from_older_bundle_refuses(history):
    bundle, path = history
    connection = Connection([("002_newer", "a" * 64)])
    with pytest.raises(migrations.IncompatibleSchemaError, match="002_newer"):
        migrations.require_compatible_history(bundle, lambda: connection)
    assert all(sql.startswith("SELECT ") for sql in connection.statements)
    assert connection.closed


def test_changed_applied_migration_refuses(history):
    bundle, _ = history
    connection = Connection([("001_initial", "b" * 64)])
    with pytest.raises(migrations.IncompatibleSchemaError, match="001_initial"):
        migrations.require_compatible_history(bundle, lambda: connection)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_migration_checksums_accept_git_line_ending_conversion(history, ending):
    bundle, path = history
    checksum = hashlib.sha256(path.read_bytes().replace(b"\n", ending)).hexdigest()
    connection = Connection([("001_initial", checksum)])
    migrations.require_compatible_history(bundle, lambda: connection)


def test_first_boot_has_no_history_to_reject(tmp_path):
    connection = Connection([], exists=False)
    migrations.require_compatible_history(Bundle(tmp_path), lambda: connection)
    assert len(connection.statements) == 1


def test_upgrade_with_additional_migrations_is_allowed(history):
    bundle, path = history
    newer = path.parent.parent / "002_newer/migration.sql"
    newer.parent.mkdir()
    newer.write_text("ALTER TABLE example ADD COLUMN title text;", encoding="utf-8")
    connection = Connection(
        [("001_initial", hashlib.sha256(path.read_bytes()).hexdigest())]
    )
    migrations.require_compatible_history(bundle, lambda: connection)
