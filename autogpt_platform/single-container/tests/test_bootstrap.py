from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ASSET_DIR / "bootstrap.sh"


class InterruptedMigrationPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")

    def test_checks_for_interrupted_migration_before_deploy(self) -> None:
        migrate_body = self.bootstrap.split("migrate_database() {", 1)[1].split(
            "\n}\n\nquery_scalar()", 1
        )[0]

        self.assertLess(
            migrate_body.index("report_interrupted_migration"),
            migrate_body.index("prisma migrate deploy"),
        )

    def test_query_selects_only_unfinished_migrations(self) -> None:
        report_body = self.bootstrap.split("report_interrupted_migration() {", 1)[
            1
        ].split("\n}\n\nconfigure_frontend_database_role()", 1)[0]

        self.assertIn("finished_at IS NULL AND rolled_back_at IS NULL", report_body)

    def test_unfinished_migration_fails_closed(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                "source <(sed '$d' \"$1\"); "
                "query_scalar() { "
                'if [[ "$1" == *to_regclass* ]]; then printf "t\\n"; '
                'else printf "20260825_interrupted\\n"; fi; }; '
                "report_interrupted_migration",
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

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("20260825_interrupted", result.stdout)
        self.assertIn(
            "prisma migrate resolve --rolled-back 20260825_interrupted",
            result.stdout,
        )
        self.assertIn(
            "prisma migrate resolve --applied     20260825_interrupted",
            result.stdout,
        )
        self.assertIn(
            "refusing to migrate over an interrupted migration", result.stderr
        )


if __name__ == "__main__":
    unittest.main()
