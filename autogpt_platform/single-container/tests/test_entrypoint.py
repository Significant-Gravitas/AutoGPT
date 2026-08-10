from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parents[1]
ENTRYPOINT_PATH = ASSET_DIR / "entrypoint.sh"
RUN_SERVICE_PATH = ASSET_DIR / "run-service.sh"
DOCKERFILE_PATH = ASSET_DIR / "Dockerfile"
SUPERVISOR_PATH = ASSET_DIR / "supervisor" / "supervisord.conf"


class AccountRegistrationTest(unittest.TestCase):
    def test_defaults_open_for_loopback_origins(self) -> None:
        for public_url in (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://[::1]:3000",
        ):
            with self.subTest(public_url=public_url):
                result = self._configure(public_url)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "true\n")

    def test_defaults_closed_for_remote_origin(self) -> None:
        for allow_new_accounts in (None, ""):
            with self.subTest(allow_new_accounts=allow_new_accounts):
                result = self._configure(
                    "https://autogpt.example.com",
                    allow_new_accounts=allow_new_accounts,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "false\n")

    def test_remote_signup_requires_explicit_opt_in(self) -> None:
        result = self._configure(
            "https://autogpt.example.com", allow_new_accounts="true"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("explicitly enabled", result.stdout)
        self.assertTrue(result.stdout.endswith("true\n"))

    def test_explicit_false_overrides_loopback_default(self) -> None:
        result = self._configure("http://localhost:3000", allow_new_accounts="false")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "false\n")

    def _configure(
        self, public_url: str, allow_new_accounts: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "AUTOGPT_ASSET_DIR": str(ASSET_DIR),
            "AUTOGPT_PUBLIC_URL": public_url,
        }
        if allow_new_accounts is not None:
            environment["AUTH_ALLOW_NEW_ACCOUNTS"] = allow_new_accounts
        return subprocess.run(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                'source "$1"; configure_account_registration; '
                'printf "%s\\n" "$AUTH_ALLOW_NEW_ACCOUNTS"',
                "bash",
                str(ENTRYPOINT_PATH),
            ],
            check=False,
            capture_output=True,
            encoding="utf-8",
            env=environment,
        )


class NormalizationTest(unittest.TestCase):
    def test_rejects_invalid_integer_values(self) -> None:
        for value, error in (
            ("not-a-number", "must be an integer"),
            ("0", "must be between 1 and 5"),
            ("6", "must be between 1 and 5"),
        ):
            with self.subTest(value=value):
                result = self._run(
                    'DB_CONNECTION_LIMIT="$2"; '
                    "normalize_integer DB_CONNECTION_LIMIT 5 1 5",
                    value,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)

    def test_rejects_invalid_and_unsupported_toggles(self) -> None:
        invalid = self._run(
            'AUTOGPT_ENABLE_CLAMAV="$2"; normalize_toggle AUTOGPT_ENABLE_CLAMAV true',
            "yes",
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("must be true or false", invalid.stderr)

        unsupported = self._run("normalize_toggle UNKNOWN_TOGGLE true", "")
        self.assertNotEqual(unsupported.returncode, 0)
        self.assertIn("unsupported toggle", unsupported.stderr)

    def _run(self, expression: str, value: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                f'source "$1"; {expression}',
                "bash",
                str(ENTRYPOINT_PATH),
                value,
            ],
            check=False,
            capture_output=True,
            encoding="utf-8",
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "AUTOGPT_ASSET_DIR": str(ASSET_DIR),
            },
        )


class ValkeyConfigurationTest(unittest.TestCase):
    def test_password_is_kept_out_of_process_arguments(self) -> None:
        entrypoint = ENTRYPOINT_PATH.read_text(encoding="utf-8")
        service_runner = RUN_SERVICE_PATH.read_text(encoding="utf-8")

        self.assertIn("printf 'requirepass %s", entrypoint)
        self.assertIn("printf 'masterauth %s", entrypoint)
        self.assertIn("chmod 0400", entrypoint)
        self.assertNotIn("--requirepass", service_runner)
        self.assertNotIn("--masterauth", service_runner)


class ProxyIsolationTest(unittest.TestCase):
    def test_nginx_uses_a_dedicated_operating_system_user(self) -> None:
        dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
        supervisor = SUPERVISOR_PATH.read_text(encoding="utf-8")
        nginx_program = supervisor.split("[program:nginx]", 1)[1].split(
            "[program:watchdog]", 1
        )[0]

        self.assertIn("--uid 10006", dockerfile)
        self.assertIn("user=autogpt_proxy", nginx_program)
        self.assertIn("AUTOGPT_HOME=/run/autogpt/nginx/home", nginx_program)
        self.assertNotIn("user=autogpt\n", nginx_program)


if __name__ == "__main__":
    unittest.main()
