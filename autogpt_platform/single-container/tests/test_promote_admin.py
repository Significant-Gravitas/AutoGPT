from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


SINGLE_CONTAINER_DIR = Path(__file__).resolve().parents[1]
PROMOTE_ADMIN_PATH = SINGLE_CONTAINER_DIR / "promote-admin.sh"


@dataclass
class Invocation:
    result: subprocess.CompletedProcess[str]
    arguments: list[str]
    sql: str
    password: str


class PromoteAdminTest(unittest.TestCase):
    def test_rejects_invalid_arguments_and_email(self) -> None:
        cases = [
            ((), "usage: autogpt-admin promote EMAIL"),
            (("promote",), "invalid email address"),
            (("delete", "admin@example.com"), "usage: autogpt-admin promote EMAIL"),
            (("promote", "not-an-email"), "invalid email address"),
        ]
        for arguments, error in cases:
            with self.subTest(arguments=arguments):
                invocation = self._run_script(arguments)
                self.assertNotEqual(invocation.result.returncode, 0)
                self.assertIn(error, invocation.result.stderr)
                self.assertEqual(invocation.arguments, [])

    def test_promotes_exactly_one_user_with_parameterized_email(self) -> None:
        email = "o'reilly@example.com"
        invocation = self._run_script(("promote", email), psql_result="1")

        self.assertEqual(invocation.result.returncode, 0, invocation.result.stderr)
        self.assertIn(f"promoted {email} to administrator", invocation.result.stdout)
        self.assertIn(f"--set=target_email={email}", invocation.arguments)
        self.assertIn("lower(:'target_email')", invocation.sql)
        self.assertNotIn(email, invocation.sql)
        self.assertEqual(invocation.password, "test-postgres-password")

    def test_rejects_non_unique_user(self) -> None:
        email = "duplicate@example.com"
        invocation = self._run_script(("promote", email), psql_result="0")

        self.assertNotEqual(invocation.result.returncode, 0)
        self.assertIn("AND (SELECT count(*) FROM target) = 1", invocation.sql)
        self.assertIn(
            f"no unique Better Auth user found for {email}",
            invocation.result.stderr,
        )

    def _run_script(
        self, arguments: tuple[str, ...], psql_result: str = "1"
    ) -> Invocation:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runtime_dir = root / "runtime"
            runtime_dir.mkdir()
            ready_file = runtime_dir / "ready"
            ready_file.touch()
            runtime_config = root / "runtime.env"
            runtime_config.write_text(
                "POSTGRES_PASSWORD=test-postgres-password\n",
                encoding="utf-8",
            )

            postgres_bindir = root / "postgres" / "bin"
            postgres_bindir.mkdir(parents=True)
            fake_psql = postgres_bindir / "psql"
            fake_psql.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\\n' "$@" >"${FAKE_PSQL_ARGS_FILE}"
cat >"${FAKE_PSQL_SQL_FILE}"
printf '%s' "${PGPASSWORD:-}" >"${FAKE_PSQL_PASSWORD_FILE}"
printf '%s\\n' "${FAKE_PSQL_RESULT}"
""",
                encoding="utf-8",
            )
            fake_psql.chmod(0o755)

            args_file = root / "psql.args"
            sql_file = root / "psql.sql"
            password_file = root / "psql.password"
            environment = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "AUTOGPT_ASSET_DIR": str(SINGLE_CONTAINER_DIR),
                "AUTOGPT_RUNTIME_DIR": str(runtime_dir),
                "AUTOGPT_READY_FILE": str(ready_file),
                "AUTOGPT_RUNTIME_ENV": str(runtime_config),
                "POSTGRES_BINDIR": str(postgres_bindir),
                "FAKE_PSQL_ARGS_FILE": str(args_file),
                "FAKE_PSQL_SQL_FILE": str(sql_file),
                "FAKE_PSQL_PASSWORD_FILE": str(password_file),
                "FAKE_PSQL_RESULT": psql_result,
            }
            result = subprocess.run(
                ["bash", str(PROMOTE_ADMIN_PATH), *arguments],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=environment,
            )

            return Invocation(
                result=result,
                arguments=(
                    args_file.read_text(encoding="utf-8").splitlines()
                    if args_file.exists()
                    else []
                ),
                sql=(sql_file.read_text(encoding="utf-8") if sql_file.exists() else ""),
                password=(
                    password_file.read_text(encoding="utf-8")
                    if password_file.exists()
                    else ""
                ),
            )


if __name__ == "__main__":
    unittest.main()
