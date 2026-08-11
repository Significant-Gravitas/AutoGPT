from __future__ import annotations

import ast
import os
import subprocess
import unittest
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parents[1]
COMMON_PATH = ASSET_DIR / "common.sh"
ENTRYPOINT_PATH = ASSET_DIR / "entrypoint.sh"
HEALTHCHECK_PATH = ASSET_DIR / "healthcheck.sh"
RUN_SERVICE_PATH = ASSET_DIR / "run-service.sh"
DOCKERFILE_PATH = ASSET_DIR / "Dockerfile"
SUPERVISOR_PATH = ASSET_DIR / "supervisor" / "supervisord.conf"
BACKEND_SERVICE_PATH = ASSET_DIR.parent / "backend" / "backend" / "util" / "service.py"


class InternalServiceTopologyTest(unittest.TestCase):
    def test_rpc_health_path_matches_backend(self) -> None:
        module = ast.parse(BACKEND_SERVICE_PATH.read_text(encoding="utf-8"))
        route_paths = {
            ast.literal_eval(node.args[0])
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_api_route"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        result = subprocess.run(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                'source "$1"; printf "%s" "$AUTOGPT_INTERNAL_HEALTH_PATH"',
                "bash",
                str(COMMON_PATH),
            ],
            check=False,
            capture_output=True,
            encoding="utf-8",
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(result.stdout, route_paths)
        self.assertIn(
            "${AUTOGPT_INTERNAL_HEALTH_PATH}",
            HEALTHCHECK_PATH.read_text(encoding="utf-8"),
        )


class AccountRegistrationTest(unittest.TestCase):
    def test_defaults_closed_for_all_origins(self) -> None:
        for public_url in (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://[::1]:3000",
            "https://autogpt.example.com",
        ):
            for allow_new_accounts in (None, ""):
                with self.subTest(
                    public_url=public_url,
                    allow_new_accounts=allow_new_accounts,
                ):
                    result = self._configure(public_url, allow_new_accounts)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("account registration is closed", result.stdout)
                    self.assertIn("autogpt-admin promote", result.stdout)
                    self.assertTrue(result.stdout.endswith("false\n"))

    def test_remote_signup_requires_explicit_opt_in(self) -> None:
        result = self._configure(
            "https://autogpt.example.com", allow_new_accounts="true"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("open account registration is enabled", result.stdout)
        self.assertTrue(result.stdout.endswith("true\n"))

    def test_explicit_false_overrides_loopback_default(self) -> None:
        result = self._configure("http://localhost:3000", allow_new_accounts="false")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("account registration is closed", result.stdout)
        self.assertTrue(result.stdout.endswith("false\n"))

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

    def test_rejects_invalid_toggle(self) -> None:
        invalid = self._run(
            'AUTOGPT_ENABLE_BOT_SERVICES="$2"; normalize_toggle AUTOGPT_ENABLE_BOT_SERVICES false',
            "yes",
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("must be true or false", invalid.stderr)

    def test_normalizes_named_toggle(self) -> None:
        result = self._run(
            'CUSTOM_TOGGLE="$2"; normalize_toggle CUSTOM_TOGGLE false; '
            'printf "%s\\n" "$CUSTOM_TOGGLE"',
            "true",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "true\n")

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
