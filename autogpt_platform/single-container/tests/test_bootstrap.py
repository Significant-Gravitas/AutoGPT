from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ASSET_DIR / "bootstrap.sh"


class InterruptedMigrationPolicyTest(unittest.TestCase):
    def test_sourcing_bootstrap_does_not_run_main(self) -> None:
        result = self._run('printf "sourced-only\\n"')

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "sourced-only\n")

    def test_executing_bootstrap_runs_main(self) -> None:
        missing_runtime = ASSET_DIR / "tests" / ".missing-bootstrap-runtime.env"
        self.assertFalse(missing_runtime.exists())

        result = subprocess.run(
            ["bash", str(BOOTSTRAP_PATH)],
            check=False,
            capture_output=True,
            encoding="utf-8",
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "AUTOGPT_ASSET_DIR": str(ASSET_DIR),
                "AUTOGPT_RUNTIME_ENV": str(missing_runtime),
            },
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing runtime config: {missing_runtime}", result.stderr)

    def test_checks_for_interrupted_migration_before_deploy(self) -> None:
        result = self._run("declare -f migrate_database")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(
            result.stdout.index("report_interrupted_migration"),
            result.stdout.index("prisma migrate deploy"),
        )

    def test_query_selects_only_unfinished_migrations(self) -> None:
        result = self._run("declare -f report_interrupted_migration")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("finished_at IS NULL AND rolled_back_at IS NULL", result.stdout)

    def test_missing_migrations_table_is_a_clean_boot(self) -> None:
        result = self._run(
            'query_scalar() { printf "f\\n"; }; '
            'report_interrupted_migration; printf "completed\\n"'
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "completed\n")
        self.assertEqual(result.stderr, "")

    def test_no_unfinished_migrations_is_a_clean_boot(self) -> None:
        result = self._run(
            "query_scalar() { "
            'if [[ "$1" == *to_regclass* ]]; then printf "t\\n"; '
            'else printf "\\n"; fi; }; '
            'report_interrupted_migration; printf "completed\\n"'
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "completed\n")
        self.assertEqual(result.stderr, "")

    def test_unfinished_migration_fails_closed(self) -> None:
        result = self._run(
            "query_scalar() { "
            'if [[ "$1" == *to_regclass* ]]; then printf "t\\n"; '
            'else printf "20260825_interrupted\\n"; fi; }; '
            "report_interrupted_migration"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("20260825_interrupted", result.stdout)
        self.assertIn(
            "prisma migrate resolve --rolled-back 20260825_interrupted",
            result.stdout,
        )
        self.assertIn("prisma migrate resolve --applied", result.stdout)
        self.assertIn(
            "refusing to migrate over an interrupted migration", result.stderr
        )

    def _run(self, command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                f'source "$1"; {command}',
                "bash",
                str(BOOTSTRAP_PATH),
            ],
            check=False,
            capture_output=True,
            encoding="utf-8",
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "AUTOGPT_ASSET_DIR": str(ASSET_DIR),
            },
        )


if __name__ == "__main__":
    unittest.main()
