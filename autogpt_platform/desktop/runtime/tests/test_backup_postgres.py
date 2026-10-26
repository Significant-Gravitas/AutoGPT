import hashlib
import os
import shutil
import socket
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from autogpt_desktop import backup, migrations, postgres
from autogpt_desktop.layout import Bundle


@pytest.mark.skipif(
    not os.environ.get("AUTOGPT_TEST_POSTGRES_RUNTIME"),
    reason="requires an explicitly selected existing PostgreSQL runtime",
)
def test_populated_database_upgrade_guard_and_cold_restore(tmp_path):
    import psycopg2
    from cryptography.fernet import Fernet

    runtime = Bundle(Path(os.environ["AUTOGPT_TEST_POSTGRES_RUNTIME"]).resolve())

    class ProofBundle(Bundle):
        def postgres_bin(self, name):
            return runtime.postgres_bin(name)

    proof = tmp_path / "autogpt-backup-proof"
    proof.mkdir()
    root = proof / "AutoGPT"
    root.mkdir()
    cluster = root / "postgres"
    cluster.mkdir()
    old = ProofBundle(proof / "old-runtime")
    new = ProofBundle(proof / "new-runtime")
    initial = b"CREATE TABLE platform.backup_proof (id int, credential text);\n"
    upgrade = b"ALTER TABLE platform.backup_proof ADD COLUMN later text;\n"
    for bundle in (old, new):
        path = bundle.backend_dir / "migrations/001_initial/migration.sql"
        path.parent.mkdir(parents=True)
        path.write_bytes(initial)
    later = new.backend_dir / "migrations/002_upgrade/migration.sql"
    later.parent.mkdir(parents=True)
    later.write_bytes(upgrade)
    password = "synthetic-backup-proof-only"
    password_file = proof / "test-password"
    password_file.write_text(password, encoding="utf-8")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    assert port not in (8003, 8005)
    env = postgres._tool_env(runtime)

    def tool(name, *args):
        with (proof / "tools.log").open("ab") as output:
            return subprocess.run(
                [str(runtime.postgres_bin(name)), *map(str, args)],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=True,
            )

    def start():
        tool(
            "pg_ctl",
            "-D",
            cluster,
            "-w",
            "-t",
            "30",
            "-l",
            proof / "postgres.log",
            "-o",
            f"-h 127.0.0.1 -p {port} -c shared_buffers=16MB -c max_connections=20",
            "start",
        )

    def stop():
        tool("pg_ctl", "-D", cluster, "-w", "-t", "30", "-m", "fast", "stop")

    def connect():
        return psycopg2.connect(
            host="127.0.0.1",
            port=port,
            user="postgres",
            password=password,
            dbname="postgres",
            connect_timeout=5,
        )

    try:
        tool(
            "initdb",
            "--pgdata",
            cluster,
            "--username=postgres",
            "--auth=scram-sha-256",
            "--pwfile",
            password_file,
            "--encoding=UTF8",
            "--locale=C",
        )
        password_file.unlink()
        (root / "autogpt-install.json").write_text(
            '{"install":"AutoGPT"}', encoding="utf-8"
        )
        (root / "config").mkdir()
        key = Fernet.generate_key()
        (root / "config/encryption-key").write_bytes(key)
        saved_credential = Fernet(key).encrypt(b"synthetic-private-api-key").decode()
        (root / "workspaces").mkdir()
        (root / "workspaces/saved.txt").write_text(
            "saved before upgrade", encoding="utf-8"
        )
        start()
        with closing(connect()) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE SCHEMA platform")
                cursor.execute(
                    "CREATE TABLE platform._prisma_migrations (migration_name text, checksum text, finished_at timestamptz, rolled_back_at timestamptz)"
                )
                cursor.execute(initial.decode())
                cursor.execute(
                    "INSERT INTO platform.backup_proof VALUES (1, %s)",
                    (saved_credential,),
                )
                cursor.execute(
                    "INSERT INTO platform._prisma_migrations VALUES ('001_initial', %s, now(), NULL)",
                    (hashlib.sha256(initial).hexdigest(),),
                )
            connection.commit()
        archive = proof / "before-upgrade.agptbackup"
        with pytest.raises(backup.BackupError, match="running"):
            backup.create_backup(old, root, archive, "AutoGPT")
        migrations.require_compatible_history(old, connect)
        stop()
        backup.create_backup(old, root, archive, "AutoGPT")
        start()
        with closing(connect()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(upgrade.decode())
                cursor.execute(
                    "INSERT INTO platform._prisma_migrations VALUES ('002_upgrade', %s, now(), NULL)",
                    (hashlib.sha256(upgrade).hexdigest(),),
                )
                cursor.execute(
                    "UPDATE platform.backup_proof SET later='created after upgrade'"
                )
            connection.commit()
        migrations.require_compatible_history(new, connect)
        with pytest.raises(migrations.IncompatibleSchemaError, match="002_upgrade"):
            migrations.require_compatible_history(old, connect)
        with pytest.raises(backup.BackupError, match="running"):
            backup.restore_backup(old, root, archive, "AutoGPT")
        stop()
        (root / "workspaces/saved.txt").write_text(
            "work after upgrade", encoding="utf-8"
        )
        previous = backup.restore_backup(old, root, archive, "AutoGPT")
        assert (
            previous
            and (previous / "workspaces/saved.txt").read_text("utf-8")
            == "work after upgrade"
        )
        assert (root / "workspaces/saved.txt").read_text(
            "utf-8"
        ) == "saved before upgrade"
        start()
        migrations.require_compatible_history(old, connect)
        with closing(connect()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT credential FROM platform.backup_proof WHERE id=1"
                )
                ciphertext = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM platform._prisma_migrations")
                assert cursor.fetchone()[0] == 1
                cursor.execute(
                    "SELECT count(*) FROM information_schema.columns WHERE table_schema='platform' AND table_name='backup_proof' AND column_name='later'"
                )
                assert cursor.fetchone()[0] == 0
        assert (
            Fernet((root / "config/encryption-key").read_bytes()).decrypt(
                ciphertext.encode()
            )
            == b"synthetic-private-api-key"
        )
        stop()
    finally:
        if (cluster / "postmaster.pid").exists():
            stop()
        assert proof.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(proof)
    assert not proof.exists()
