"""The first account is the owner: admin role, closed registration, and a
way back in when the password is lost (bootstrap.ensure_owner and friends)."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pytest

from autogpt_desktop import bootstrap, ports, settings, supervisor
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.supervisor import Stack

PLATFORM = Path(__file__).resolve().parents[3]

SCHEMA = {
    bootstrap.IDENTITY_TABLE: set(bootstrap.IDENTITY_COLUMNS) | {"email", "name"},
    bootstrap.ACCOUNT_TABLE: set(bootstrap.ACCOUNT_COLUMNS) | {"id"},
    bootstrap.SESSION_TABLE: set(bootstrap.SESSION_COLUMNS) | {"token"},
}


class FakeDatabase:
    """Answers the handful of statements the owner step sends."""

    def __init__(
        self,
        schema: dict[str, set[str]],
        identities: int = 0,
        owners=(),
        refuses: str | None = None,
    ):
        self.schema = schema
        self.identities = identities
        self.owners = list(owners)
        self.refuses = refuses
        self.statements: list[tuple[str, tuple]] = []
        self.autocommit = False
        self.closed = False
        self.result: list[tuple] = []

    def __call__(self) -> "FakeDatabase":
        return self

    def cursor(self) -> "FakeDatabase":
        return self

    def close(self) -> None:
        self.closed = True

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.statements.append((sql, params))
        if self.refuses and self.refuses in sql:
            raise RuntimeError('invalid input value for enum "Role": "admin"')
        if "information_schema.columns" in sql:
            table, columns = params
            self.result = [(len(set(columns) & self.schema.get(table, set())),)]
        elif "to_regclass" in sql:
            self.result = [(bootstrap.IDENTITY_TABLE in self.schema,)]
        elif sql.startswith("SELECT count(*) FROM platform"):
            self.result = [(self.identities,)]
        elif "RETURNING" in sql:
            self.result = [(owner,) for owner in self.owners]

    def fetchone(self) -> tuple:
        return self.result[0]

    def fetchall(self) -> list[tuple]:
        return self.result

    def sent(self, fragment: str) -> list[tuple[str, tuple]]:
        return [statement for statement in self.statements if fragment in statement[0]]


def test_registration_is_open_until_an_owner_exists_then_closed():
    assert settings.registration_gate(None, 0) == "true"
    assert settings.registration_gate(None, 1) == "false"
    assert settings.registration_gate(None, 7) == "false"


def test_an_explicit_setting_wins_over_the_account_count():
    assert settings.registration_gate("true", 3) == "true"
    assert settings.registration_gate("false", 1) == "false"
    # The frontend and the database must agree, so anything that is not a
    # plain "true" closes both.
    assert settings.registration_gate("TRUE", 3) == "true"
    assert settings.registration_gate("yes", 1) == "false"


def test_registration_cannot_be_closed_before_there_is_an_owner(caplog):
    """Closed with nobody to sign in, the install could never be used, and
    the earlier settings template told people to add exactly this line."""
    with caplog.at_level(logging.INFO, logger="autogpt_desktop"):
        assert settings.registration_gate("false", 0) == "true"
    assert (
        "AUTH_ALLOW_NEW_ACCOUNTS=false takes effect once the owner account exists"
        in caplog.text
    )
    assert settings.registration_gate("yes", 0) == "true"


def test_accounts_that_cannot_be_counted_leave_an_explicit_setting_alone():
    """Upstream moved the table: nothing says there is no owner."""
    assert settings.registration_gate("false", None) == "false"
    assert settings.registration_gate("true", None) == "true"
    assert settings.registration_gate(None, None) == "true"


def test_the_database_refuses_new_accounts_unless_the_user_reopened_registration():
    assert settings.closes_registration(None) is True
    assert settings.closes_registration("false") is True
    assert settings.closes_registration("yes") is True
    assert settings.closes_registration("true") is False
    assert settings.closes_registration(" True ") is False


def test_the_environment_has_no_gate_until_the_accounts_are_counted(tmp_path: Path):
    """backend_environment runs before PostgreSQL is up; a default there
    would be mistaken for the user's own setting."""
    data = DataDir(tmp_path / "data")
    data.prepare()
    bundle = Bundle(tmp_path / "runtime")
    secret = settings.ensure_secrets(bundle, data)
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}

    assert "AUTH_ALLOW_NEW_ACCOUNTS" not in settings.backend_environment(
        bundle, data, port, secret, {}
    )
    chosen = settings.backend_environment(
        bundle, data, port, secret, {"AUTH_ALLOW_NEW_ACCOUNTS": "true"}
    )
    assert chosen["AUTH_ALLOW_NEW_ACCOUNTS"] == "true"
    assert "AUTH_ALLOW_NEW_ACCOUNTS" in settings.FRONTEND_PASSTHROUGH


def test_the_close_flag_is_rendered_from_a_bool_and_nothing_else():
    assert "session_user = 'autogpt_frontend' AND TRUE THEN" in bootstrap.owner_sql(
        True
    )
    assert "session_user = 'autogpt_frontend' AND FALSE THEN" in bootstrap.owner_sql(
        False
    )
    not_bools: list[Any] = [
        "true",
        "FALSE",
        1,
        0,
        None,
        "TRUE THEN NULL; END IF; DROP TABLE x; --",
    ]
    for value in not_bools:
        with pytest.raises(TypeError):
            bootstrap.owner_sql(value)


def test_the_owner_sql_is_one_transaction_and_spares_migrations():
    sql = bootstrap.owner_sql(True).strip()
    assert sql.startswith("BEGIN;") and sql.endswith("COMMIT;")
    assert sql.count("BEGIN;") == sql.count("COMMIT;") == 1
    # Only the frontend's role is refused: migrations insert as postgres.
    refusal = sql[sql.index("ELSIF") : sql.index("RAISE EXCEPTION")]
    assert "session_user = 'autogpt_frontend'" in refusal
    assert f"pg_advisory_xact_lock({bootstrap.OWNER_LOCK_KEY})" in sql
    assert "hashtext" not in sql
    assert "CREATE OR REPLACE TRIGGER autogpt_desktop_owner BEFORE INSERT" in sql
    assert "GRANT" not in sql


def test_ensure_owner_installs_the_trigger_and_counts_accounts():
    database = FakeDatabase(SCHEMA, identities=2)

    assert bootstrap.ensure_owner(database, close_registration=True) == 2

    assert database.autocommit and database.closed
    assert database.sent("CREATE OR REPLACE TRIGGER") == [
        (bootstrap.owner_sql(True), ())
    ]
    assert not database.sent("DROP ")


def test_a_changed_upstream_schema_switches_the_owner_step_off(caplog):
    """A trigger that names a column upstream removed would fail every
    sign-up; a failed boot step would stop the app starting."""
    drifted = {**SCHEMA, bootstrap.IDENTITY_TABLE: {"id", "createdAt", "updatedAt"}}
    database = FakeDatabase(drifted, identities=1)

    with caplog.at_level(logging.WARNING, logger="autogpt_desktop"):
        assert bootstrap.ensure_owner(database, close_registration=True) == 1

    assert not database.sent("CREATE OR REPLACE")
    assert database.sent("DROP ") == [(bootstrap.DROP_OWNER_TRIGGER, ())]
    assert "Update autogpt_desktop/bootstrap.py" in caplog.text


def test_a_moved_identity_table_still_loses_its_trigger():
    """A rename carries the trigger along, and its function still reads the
    old name: it has to be removed without naming the table."""
    database = FakeDatabase({})

    assert bootstrap.ensure_owner(database, close_registration=True) is None

    assert not database.sent("CREATE OR REPLACE")
    assert database.sent("DROP ") == [(bootstrap.DROP_OWNER_TRIGGER, ())]


def test_the_trigger_is_removed_by_function_so_no_table_name_is_needed():
    assert (
        bootstrap.DROP_OWNER_TRIGGER
        == "DROP FUNCTION IF EXISTS platform.autogpt_desktop_owner() CASCADE"
    )
    assert "FUNCTION platform.autogpt_desktop_owner()" in bootstrap.owner_sql(True)
    assert bootstrap.IDENTITY_TABLE not in bootstrap.DROP_OWNER_TRIGGER


def test_owner_sql_the_database_refuses_switches_the_step_off(caplog):
    """The names are all there but a type changed (role became an enum, say).
    The failed script leaves its transaction open and must not stop the app."""
    database = FakeDatabase(SCHEMA, identities=1, refuses="CREATE OR REPLACE TRIGGER")

    with caplog.at_level(logging.WARNING, logger="autogpt_desktop"):
        assert bootstrap.ensure_owner(database, close_registration=True) == 1

    after = [sql for sql, _ in database.statements]
    after = after[after.index(bootstrap.owner_sql(True)) + 1 :]
    assert after[:2] == ["ROLLBACK", bootstrap.DROP_OWNER_TRIGGER]
    assert "invalid input value for enum" in caplog.text
    assert "Update autogpt_desktop/bootstrap.py" in caplog.text
    assert database.closed


def test_the_trigger_is_off_while_migrations_run(stack_for_migrate):
    stack, calls = stack_for_migrate
    secret = {"POSTGRES_PASSWORD": "p", "AUTOGPT_FRONTEND_DB_PASSWORD": "f"}

    stack.migrate({"postgres": 1}, secret, False)

    assert calls[:2] == ["require_compatible_history", "create_schemas"]
    assert calls.index("remove_owner_trigger") < calls.index("apply_migrations")
    assert calls.index("apply_migrations") < calls.index("ensure_owner")


def test_a_trigger_that_cannot_be_removed_does_not_stop_the_app(caplog):
    def connect():
        raise ConnectionError("the database went away")

    with caplog.at_level(logging.WARNING, logger="autogpt_desktop"):
        bootstrap.remove_owner_trigger(connect)
    assert "could not remove the owner trigger" in caplog.text

    database = FakeDatabase(SCHEMA)
    bootstrap.remove_owner_trigger(database)
    assert database.statements == [(bootstrap.DROP_OWNER_TRIGGER, ())]
    assert database.autocommit and database.closed


def prisma_model(schema: str, table: str) -> str:
    """The body of the Prisma model mapped to `table`."""
    for body in re.findall(
        r"^model \w+ \{\n(.*?)^\}", schema, re.DOTALL | re.MULTILINE
    ):
        if f'@@map("{table}")' in body:
            return body
    return ""


@pytest.mark.parametrize(
    "table, columns",
    [
        (bootstrap.IDENTITY_TABLE, bootstrap.IDENTITY_COLUMNS),
        (bootstrap.ACCOUNT_TABLE, bootstrap.ACCOUNT_COLUMNS),
        (bootstrap.SESSION_TABLE, bootstrap.SESSION_COLUMNS),
    ],
)
def test_the_upstream_auth_tables_still_have_the_columns_the_owner_step_uses(
    table: str, columns: tuple[str, ...]
):
    schema = (PLATFORM / "backend" / "schema.prisma").read_text(encoding="utf-8")
    model = prisma_model(schema, table)
    fields = set(re.findall(r"^\s+(\w+)\s+\w", model, re.MULTILINE))
    update = (
        "Update the table and column names at the top of "
        "desktop/runtime/autogpt_desktop/bootstrap.py (and the SQL below them) "
        "to match backend/schema.prisma."
    )
    assert model, f'backend/schema.prisma no longer maps a model to "{table}". {update}'
    missing = set(columns) - fields
    assert not missing, f'"{table}" no longer has {sorted(missing)}. {update}'


def test_the_password_rules_are_the_frontends():
    policy = (
        PLATFORM / "frontend" / "src" / "lib" / "auth" / "password-policy.ts"
    ).read_text(encoding="utf-8")
    constants = dict(re.findall(r"export const (\w+) = (\d+);", policy))
    update = "Update PASSWORD_* in desktop/runtime/autogpt_desktop/bootstrap.py."
    assert constants.get("AUTH_PASSWORD_MIN_LENGTH") == str(
        bootstrap.PASSWORD_MIN_LENGTH
    ), update
    assert constants.get("AUTH_PASSWORD_BCRYPT_COST") == str(
        bootstrap.PASSWORD_BCRYPT_COST
    ), update


def test_the_frontend_still_checks_passwords_with_bcrypt():
    auth = (PLATFORM / "frontend" / "src" / "lib" / "auth" / "auth.ts").read_text(
        encoding="utf-8"
    )
    update = "Update hash_password in desktop/runtime/autogpt_desktop/bootstrap.py."
    assert 'from "bcryptjs"' in auth, update
    assert "hash(password, AUTH_PASSWORD_BCRYPT_COST)" in auth, update
    assert (
        "maxPasswordLength" not in auth
    ), f"PASSWORD_MAX_LENGTH assumes the default. {update}"


@pytest.mark.parametrize(
    "content",
    [
        b"correct horse battery",
        b"correct horse battery\n",
        b"correct horse battery\r\nsecond line is ignored\r\n",
        b"\xef\xbb\xbfcorrect horse battery\r\n",
        "correct horse battery\r\n".encode("utf-16"),
    ],
)
def test_a_password_file_is_read_once_and_deleted(tmp_path: Path, content: bytes):
    path = tmp_path / bootstrap.RESET_PASSWORD_FILE
    path.write_bytes(content)

    assert bootstrap.take_password_reset(path) == "correct horse battery"
    assert not path.exists()
    assert bootstrap.take_password_reset(path) is None


@pytest.mark.parametrize(
    "content, reason",
    [
        (b"too short\n", "at least 12 characters"),
        (b"", "at least 12 characters"),
        (b"\n" + b"correct horse battery", "at least 12 characters"),
        (b"x" * 129, "at most 128 characters"),
        (b"\xff\x00\xfe not text at all", "not readable text"),
    ],
)
def test_an_unusable_password_file_is_rejected_logged_and_still_deleted(
    tmp_path: Path, caplog, content: bytes, reason: str
):
    path = tmp_path / bootstrap.RESET_PASSWORD_FILE
    path.write_bytes(content)

    with caplog.at_level(logging.WARNING, logger="autogpt_desktop"):
        assert bootstrap.take_password_reset(path) is None

    assert not path.exists()
    assert reason in caplog.text
    assert "too short" not in caplog.text  # the password itself is never logged


def test_a_password_file_that_cannot_be_read_is_still_deleted(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / bootstrap.RESET_PASSWORD_FILE
    path.write_bytes(b"correct horse battery")

    def refuse(self: Path) -> bytes:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    assert bootstrap.take_password_reset(path) is None
    assert not path.exists()


def test_a_reset_stores_the_hash_for_the_owner_and_signs_them_out(monkeypatch):
    monkeypatch.setattr(
        bootstrap, "hash_password", lambda password: f"hashed:{password}"
    )
    database = FakeDatabase(SCHEMA, identities=1, owners=["owner-id"])

    assert bootstrap.reset_owner_password(database, "correct horse battery") is True

    ((update, parameters),) = database.sent('UPDATE platform."UserAuthAccount"')
    assert parameters == ("hashed:correct horse battery",)
    assert "\"providerId\" = 'credential'" in update
    assert "role = 'admin' ORDER BY \"createdAt\", id LIMIT 1" in update
    assert database.sent('DELETE FROM platform."UserAuthSession"') == [
        ('DELETE FROM platform."UserAuthSession" WHERE "userId" = %s', ("owner-id",))
    ]
    order = [sql for sql, _ in database.statements if sql in ("BEGIN", "COMMIT")]
    assert order == ["BEGIN", "COMMIT"]
    assert database.closed


def test_a_reset_with_no_owner_account_changes_nothing(monkeypatch, caplog):
    monkeypatch.setattr(bootstrap, "hash_password", lambda password: "hashed")
    database = FakeDatabase(SCHEMA)

    with caplog.at_level(logging.WARNING, logger="autogpt_desktop"):
        assert (
            bootstrap.reset_owner_password(database, "correct horse battery") is False
        )

    assert not database.sent("DELETE FROM")
    assert "no owner account" in caplog.text


def test_a_reset_against_a_changed_schema_is_skipped(monkeypatch):
    monkeypatch.setattr(bootstrap, "hash_password", lambda password: "hashed")
    drifted = {**SCHEMA, bootstrap.ACCOUNT_TABLE: {"userId", "providerId"}}
    database = FakeDatabase(drifted, owners=["owner-id"])

    assert bootstrap.reset_owner_password(database, "correct horse battery") is False
    assert not database.sent("UPDATE") and not database.sent("DELETE")


def test_a_reset_that_fails_does_not_stop_the_app_starting(monkeypatch, caplog):
    monkeypatch.setattr(bootstrap, "hash_password", lambda password: "hashed")

    def connect():
        raise ConnectionError("the database went away")

    with caplog.at_level(logging.ERROR, logger="autogpt_desktop"):
        assert bootstrap.reset_owner_password(connect, "correct horse battery") is False
    assert "the owner password was not reset" in caplog.text
    assert "correct horse battery" not in caplog.text


def test_the_hash_is_what_bcryptjs_verifies():
    bcrypt = pytest.importorskip("bcrypt")
    hashed = bootstrap.hash_password("correct horse battery")
    assert hashed.startswith(f"$2b${bootstrap.PASSWORD_BCRYPT_COST}$")
    assert bcrypt.checkpw(b"correct horse battery", hashed.encode())
    # bcryptjs cuts a long password at 72 bytes; so must the hash it checks.
    long = "é" * 60
    assert bcrypt.checkpw(long.encode()[:72], bootstrap.hash_password(long).encode())


@pytest.fixture
def stack(tmp_path: Path) -> Stack:
    data = DataDir(tmp_path / "data")
    data.prepare()
    return Stack(Bundle(tmp_path / "runtime"), data)


@pytest.fixture
def stack_for_migrate(stack: Stack, monkeypatch) -> tuple[Stack, list[str]]:
    calls: list[str] = []
    monkeypatch.setattr(
        supervisor.migrations,
        "require_compatible_history",
        lambda *args: calls.append("require_compatible_history"),
    )

    def record(name: str, result=None) -> None:
        def step(*args, **kwargs):
            calls.append(name)
            return result

        monkeypatch.setattr(bootstrap, name, step)

    for name in (
        "create_schemas",
        "refuse_interrupted_migration",
        "remove_owner_trigger",
        "apply_migrations",
        "configure_frontend_role",
    ):
        record(name)
    record("ensure_owner", 0)
    monkeypatch.setattr(stack, "database_connector", lambda port, password: None)
    monkeypatch.setattr(supervisor.postgres, "first_run_completed", lambda data: None)
    return stack, calls


@pytest.mark.parametrize(
    "configured, identities, closes, gate",
    [
        (None, 0, True, "true"),
        (None, 1, True, "false"),
        ("true", 1, False, "true"),
        ("false", 0, True, "true"),
        ("false", 1, True, "false"),
        ("false", None, True, "false"),
    ],
)
def test_the_stack_sets_the_gate_after_counting_accounts(
    stack: Stack, monkeypatch, configured, identities, closes, gate
):
    seen = []

    def ensure_owner(connect, close_registration):
        seen.append(close_registration)
        return identities

    monkeypatch.setattr(bootstrap, "ensure_owner", ensure_owner)
    stack.env = {"AUTH_ALLOW_NEW_ACCOUNTS": configured} if configured else {}

    stack.secure_owner(connect=None)

    assert seen == [closes]
    assert stack.env["AUTH_ALLOW_NEW_ACCOUNTS"] == gate


def test_the_stack_applies_a_pending_password_reset_once(stack: Stack, monkeypatch):
    calls = []
    monkeypatch.setattr(
        bootstrap, "ensure_owner", lambda connect, close: calls.append("owner") or 1
    )
    monkeypatch.setattr(
        bootstrap,
        "reset_owner_password",
        lambda connect, password: calls.append(password),
    )
    stack.password_reset = "correct horse battery"

    stack.secure_owner(connect=None)
    stack.secure_owner(connect=None)

    assert calls == ["owner", "correct horse battery", "owner"]
    assert stack.password_reset is None


def test_the_password_file_is_taken_before_anything_that_can_fail(
    stack: Stack, monkeypatch
):
    """A start that fails must not leave the password on disk in the clear."""
    path = stack.data.config / bootstrap.RESET_PASSWORD_FILE
    path.write_text("correct horse battery\n", encoding="utf-8")

    def fail(self: DataDir) -> None:
        raise OSError("the disk is full")

    monkeypatch.setattr(DataDir, "prepare", fail)
    with pytest.raises(supervisor.rabbitmq.IncompatibleDataError):
        stack.start()

    assert not path.exists()
    assert stack.password_reset == "correct horse battery"


def events_printed(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_a_start_that_fails_says_the_password_reset_was_lost(
    stack: Stack, monkeypatch, capsys, caplog
):
    def start() -> str:
        stack.password_reset = "correct horse battery"
        raise supervisor.StartupError("rabbitmq did not start.")

    monkeypatch.setattr(stack, "start", start)
    with caplog.at_level(logging.WARNING, logger="autogpt_desktop"):
        assert stack.run_until_stopped() == 1

    (event,) = events_printed(capsys)
    assert event["fatal"] is True
    assert (
        event["message"] == f"rabbitmq did not start. {supervisor.PASSWORD_RESET_LOST}"
    )
    assert "Reset it again" in caplog.text
    assert "correct horse battery" not in caplog.text + event["message"]
    assert stack.password_reset is None


def test_a_cancelled_start_logs_the_lost_password_reset(
    stack: Stack, monkeypatch, capsys, caplog
):
    def start() -> str:
        stack.password_reset = "correct horse battery"
        stack.stop_requested.set()
        raise supervisor.StartupError("startup was cancelled")

    monkeypatch.setattr(stack, "start", start)
    with caplog.at_level(logging.WARNING, logger="autogpt_desktop"):
        assert stack.run_until_stopped() == 0

    assert events_printed(capsys) == []
    assert supervisor.PASSWORD_RESET_LOST in caplog.text


def test_a_failed_start_without_a_reset_says_nothing_about_one(
    stack: Stack, monkeypatch, capsys
):
    def start() -> str:
        raise supervisor.StartupError("rabbitmq did not start.")

    monkeypatch.setattr(stack, "start", start)
    assert stack.run_until_stopped() == 1
    (event,) = events_printed(capsys)
    assert event["message"] == "rabbitmq did not start."
