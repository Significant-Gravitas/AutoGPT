"""Database bootstrap, ported from single-container/bootstrap.sh.

Order and semantics match the appliance: create the schemas, refuse to run
over an interrupted migration, apply Prisma migrations, then lock the
frontend's role down to the auth tables. The role policy SQL is the
appliance's own, extracted from bootstrap.sh when the bundle is built
(see build/frontend_role_sql.py), so the two distributions grant the same
privileges.

The last step is the desktop's own: the first account created on the machine
is its owner and an admin, and once it exists nobody else can register (see
`ensure_owner`). The appliance leaves that to `autogpt-admin promote`.
"""

from __future__ import annotations

import codecs
import logging
from contextlib import closing
from pathlib import Path

from autogpt_desktop import migrations
from autogpt_desktop.layout import Bundle
from autogpt_desktop.process import run_tool
from autogpt_desktop.settings import FRONTEND_DB_ROLE

logger = logging.getLogger("autogpt_desktop")

MIGRATION_TIMEOUT_SECONDS = 900

# The upstream names the owner step depends on. tests/test_owner.py checks
# them against backend/schema.prisma; at run time a missing one switches the
# step off instead of stopping the app (see `ensure_owner`).
IDENTITY_TABLE = "UserAuthIdentity"
IDENTITY_COLUMNS = ("id", "role", "createdAt", "updatedAt")
ACCOUNT_TABLE = "UserAuthAccount"
ACCOUNT_COLUMNS = ("userId", "providerId", "password")
SESSION_TABLE = "UserAuthSession"
SESSION_COLUMNS = ("userId",)

OWNER_TRIGGER = "autogpt_desktop_owner"
# Serialises two first sign-ups arriving together. Any constant will do, as
# long as nothing else in this database takes the same one.
OWNER_LOCK_KEY = 7203114480952213
# By function, not `DROP TRIGGER ... ON <table>`: the trigger has to go even
# when its table has been renamed or dropped since it was installed.
DROP_OWNER_TRIGGER = f"DROP FUNCTION IF EXISTS platform.{OWNER_TRIGGER}() CASCADE"
OWNER_STEP_OFF = (
    "The desktop's owner account handling is switched off. "
    "Update autogpt_desktop/bootstrap.py to the new schema."
)

# frontend/src/lib/auth/password-policy.ts, which tests/test_owner.py reads.
PASSWORD_MIN_LENGTH = 12
PASSWORD_BCRYPT_COST = 10
# Better Auth's default maxPasswordLength, which the frontend does not change.
PASSWORD_MAX_LENGTH = 128
# bcrypt reads 72 bytes. bcryptjs, which checks the hash, cuts a longer
# password there; the Python library refuses one instead.
BCRYPT_INPUT_BYTES = 72
RESET_PASSWORD_FILE = "reset-password"


class InterruptedMigrationError(RuntimeError):
    pass


def create_schemas(bundle: Bundle, connect) -> None:
    sql = bundle.init_sql.read_text(encoding="utf-8")
    with closing(_autocommit(connect)) as connection:
        connection.cursor().execute(sql)


def refuse_interrupted_migration(connect) -> None:
    with closing(_autocommit(connect)) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT to_regclass('platform._prisma_migrations') IS NOT NULL")
        (exists,) = cursor.fetchone()
        if not exists:
            return
        cursor.execute(
            "SELECT coalesce(string_agg(migration_name, ', ' ORDER BY started_at), '') "
            "FROM platform._prisma_migrations "
            "WHERE finished_at IS NULL AND rolled_back_at IS NULL"
        )
        (unfinished,) = cursor.fetchone()
    if unfinished:
        raise InterruptedMigrationError(
            f"A database update did not finish ({unfinished}). This usually means "
            "AutoGPT was closed while it was starting for the first time. If you "
            "have no data yet, delete the AutoGPT data folder and start again."
        )


def apply_migrations(bundle: Bundle, env: dict[str, str]) -> None:
    migrations.require_engines(bundle)
    command = [
        *bundle.node_command(),
        str(bundle.prisma_cli),
        "migrate",
        "deploy",
        "--schema",
        str(bundle.backend_dir / "schema.prisma"),
    ]
    result = run_tool(
        command,
        cwd=bundle.backend_dir,
        # Offline, with the bundle's own engines: see migrations.py.
        env=migrations.environment(bundle, env),
        capture_output=True,
        timeout=MIGRATION_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).decode(errors="replace")
        raise RuntimeError(f"prisma migrate deploy failed:\n{output[-4000:]}")


def configure_frontend_role(bundle: Bundle, connect, password: str) -> None:
    policy = (bundle.root / "assets" / "frontend-role.sql").read_text(encoding="utf-8")
    with closing(_autocommit(connect)) as connection:
        cursor = connection.cursor()
        cursor.execute(policy)
        # The appliance authenticates this role by Unix-socket peer auth and
        # leaves it passwordless; loopback TCP needs a password instead.
        cursor.execute(f'ALTER ROLE "{FRONTEND_DB_ROLE}" PASSWORD %s', (password,))


def remove_owner_trigger(connect) -> None:
    """Take the owner trigger off, by dropping its function.

    Before migrations: the trigger follows its table through a rename while
    its function goes on reading the old name, and it fires for the migration
    role too, so an upstream migration that renames the table and then inserts
    into it would fail half-way and the app would refuse to start from then
    on. `ensure_owner` puts the trigger back afterwards."""
    try:
        with closing(_autocommit(connect)) as connection:
            connection.cursor().execute(DROP_OWNER_TRIGGER)
    except Exception as exc:
        logger.warning(f"could not remove the owner trigger before migrating: {exc}")


def ensure_owner(connect, close_registration: bool) -> int | None:
    """Make the first account the owner, and return how many accounts exist
    (None when upstream moved the table, so they cannot be counted).

    Runs on every boot, after the migrations and the role policy: a migration
    that recreates the table would take the trigger with it. Upstream changing
    a name or a type this relies on switches the step off; it never stops the
    app."""
    sql = owner_sql(close_registration)
    with closing(_autocommit(connect)) as connection:
        cursor = connection.cursor()
        installed = _has_columns(
            cursor, IDENTITY_TABLE, IDENTITY_COLUMNS
        ) and _install_owner_trigger(cursor, sql)
        if not installed:
            # One left by an earlier boot would fail every sign-up.
            cursor.execute(DROP_OWNER_TRIGGER)
        return _count_identities(cursor)


def _install_owner_trigger(cursor, sql: str) -> bool:
    """False when the database refuses the SQL: the names are still there but
    something else about them changed (a column's type, say)."""
    try:
        cursor.execute(sql)
    except Exception as exc:
        cursor.execute("ROLLBACK")  # the failed script left its transaction open
        reason = (str(exc).splitlines() or [type(exc).__name__])[0]
        logger.warning(f"the owner account step failed: {reason}")
        logger.warning(OWNER_STEP_OFF)
        return False
    return True


def owner_sql(close_registration: bool) -> str:
    """One transaction: promote the oldest account when nobody is an admin
    (an install that predates this step), then install the trigger.

    The trigger makes the first row an admin inside its own INSERT, so the
    owner's first session already carries the role. Better Auth's admin plugin
    writes an explicit role='user', which a column default could not beat.
    Further sign-ups are refused only for the frontend's role: a migration
    that inserts accounts runs as postgres, and raising inside one would leave
    it unfinished and the app unable to start."""
    if not isinstance(close_registration, bool):
        raise TypeError("close_registration must be a bool")
    closed = "TRUE" if close_registration else "FALSE"
    table = f'platform."{IDENTITY_TABLE}"'
    return f"""
BEGIN;
UPDATE {table} SET role = 'admin', "updatedAt" = now()
 WHERE id = (SELECT id FROM {table} ORDER BY "createdAt", id LIMIT 1)
   AND NOT EXISTS (SELECT 1 FROM {table} WHERE role = 'admin');
CREATE OR REPLACE FUNCTION platform.{OWNER_TRIGGER}() RETURNS trigger
 LANGUAGE plpgsql SECURITY INVOKER SET search_path = pg_catalog AS $fn$
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock({OWNER_LOCK_KEY});
  IF NOT EXISTS (SELECT 1 FROM {table}) THEN
    NEW.role := 'admin';
  ELSIF session_user = '{FRONTEND_DB_ROLE}' AND {closed} THEN
    RAISE EXCEPTION 'New account registration is not allowed at this time.';
  END IF;
  RETURN NEW;
END $fn$;
CREATE OR REPLACE TRIGGER {OWNER_TRIGGER} BEFORE INSERT ON {table}
 FOR EACH ROW EXECUTE FUNCTION platform.{OWNER_TRIGGER}();
COMMIT;
"""


def take_password_reset(path: Path) -> str | None:
    """Read the one-shot password file and delete it, whatever it holds: a
    password must not stay on disk in the clear. None when there is no usable
    password in it."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning(f"could not read {path.name}: {exc}")
        return None
    finally:
        _delete(path)
    password = _first_line(raw)
    problem = password_problem(password)
    if problem:
        logger.warning(f"the owner password was not reset: {problem}")
        return None
    return password


def password_problem(password: str | None) -> str | None:
    if password is None:
        return f"{RESET_PASSWORD_FILE} is not readable text"
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"the new password must be at least {PASSWORD_MIN_LENGTH} characters"
    if len(password) > PASSWORD_MAX_LENGTH:
        return f"the new password must be at most {PASSWORD_MAX_LENGTH} characters"
    return None


def reset_owner_password(connect, password: str) -> bool:
    """Give the owner (the oldest admin) a new password and sign them out
    everywhere. Never fatal: a failed reset is logged and the app starts."""
    try:
        changed = _store_owner_password(connect, hash_password(password))
    except Exception:
        logger.exception("the owner password was not reset")
        return False
    if changed:
        logger.info("the owner password was reset")
    else:
        logger.warning(
            "the owner password was not reset: there is no owner account "
            "with a password yet"
        )
    return changed


def hash_password(password: str) -> str:
    """What the frontend's bcryptjs writes and verifies (lib/auth/auth.ts)."""
    import bcrypt

    secret = password.encode("utf-8")[:BCRYPT_INPUT_BYTES]
    return bcrypt.hashpw(secret, bcrypt.gensalt(PASSWORD_BCRYPT_COST)).decode("ascii")


def _store_owner_password(connect, hashed: str) -> bool:
    with closing(_autocommit(connect)) as connection:
        cursor = connection.cursor()
        tables = (
            (IDENTITY_TABLE, IDENTITY_COLUMNS),
            (ACCOUNT_TABLE, ACCOUNT_COLUMNS),
            (SESSION_TABLE, SESSION_COLUMNS),
        )
        if not all(_has_columns(cursor, table, columns) for table, columns in tables):
            return False
        cursor.execute("BEGIN")
        cursor.execute(
            f'UPDATE platform."{ACCOUNT_TABLE}" SET password = %s '
            """WHERE "providerId" = 'credential' AND "userId" = ("""
            f'SELECT id FROM platform."{IDENTITY_TABLE}" '
            """WHERE role = 'admin' ORDER BY "createdAt", id LIMIT 1) """
            'RETURNING "userId"',
            (hashed,),
        )
        owners = cursor.fetchall()
        for (owner,) in owners:
            cursor.execute(
                f'DELETE FROM platform."{SESSION_TABLE}" WHERE "userId" = %s', (owner,)
            )
        cursor.execute("COMMIT")
    return bool(owners)


def _has_columns(cursor, table: str, columns: tuple[str, ...]) -> bool:
    cursor.execute(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_schema = 'platform' AND table_name = %s AND column_name = ANY(%s)",
        (table, list(columns)),
    )
    (found,) = cursor.fetchone()
    if found != len(columns):
        logger.warning(
            f'platform."{table}" no longer has the columns {", ".join(columns)}.'
        )
        logger.warning(OWNER_STEP_OFF)
    return found == len(columns)


def _count_identities(cursor) -> int | None:
    if not _identity_table_exists(cursor):
        return None
    cursor.execute(f'SELECT count(*) FROM platform."{IDENTITY_TABLE}"')
    (count,) = cursor.fetchone()
    return count


def _identity_table_exists(cursor) -> bool:
    cursor.execute(
        "SELECT to_regclass(%s) IS NOT NULL", (f'platform."{IDENTITY_TABLE}"',)
    )
    (exists,) = cursor.fetchone()
    return exists


def _first_line(raw: bytes) -> str | None:
    """The file may be written by hand: Notepad adds a byte-order mark, and
    Windows PowerShell's `>` writes UTF-16."""
    utf16 = raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
    try:
        text = raw.decode("utf-16" if utf16 else "utf-8-sig")
    except UnicodeDecodeError:
        return None
    lines = text.splitlines()
    return lines[0] if lines else ""


def _delete(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.error(f"could not delete {path}; delete it by hand: {exc}")


def _autocommit(connect):
    # Not `with connection:` -- psycopg2 opens a transaction for that block
    # even in autocommit mode, and the scripts here manage their own.
    connection = connect()
    connection.autocommit = True
    return connection
