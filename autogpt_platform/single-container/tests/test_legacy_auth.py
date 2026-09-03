from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


COMMON_PATH = Path(__file__).resolve().parents[1] / "common.sh"


class LegacyAuthTest(unittest.TestCase):
    def test_rejects_unsafe_legacy_auth_configurations(self) -> None:
        cases = [
            (
                {"JWT_VERIFY_KEY": "x" * 32},
                "legacy JWT secrets were supplied",
            ),
            (
                {
                    "AUTOGPT_ENABLE_LEGACY_AUTH": "true",
                    "JWT_VERIFY_KEY": "x" * 32,
                    "SUPABASE_JWT_SECRET": "y" * 32,
                },
                "must match",
            ),
            (
                {
                    "AUTOGPT_ENABLE_LEGACY_AUTH": "true",
                    "JWT_VERIFY_KEY": "too-short",
                },
                "at least 32 characters",
            ),
        ]
        for environment, error in cases:
            with self.subTest(environment=environment):
                result = self._run_validation(environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)

    def test_accepts_and_exports_explicit_legacy_secret(self) -> None:
        secret = "x" * 32
        result = self._run_validation(
            {
                "AUTOGPT_ENABLE_LEGACY_AUTH": "true",
                "JWT_VERIFY_KEY": secret,
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.rstrip().endswith(f"{secret}|{secret}"))

    def _run_validation(
        self, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        clean_environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        clean_environment.update(environment)
        return subprocess.run(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                'source "$1"; validate_legacy_auth; '
                'printf "%s|%s\\n" "$JWT_VERIFY_KEY" "$SUPABASE_JWT_SECRET"',
                "bash",
                str(COMMON_PATH),
            ],
            check=False,
            capture_output=True,
            encoding="utf-8",
            env=clean_environment,
        )


if __name__ == "__main__":
    unittest.main()
