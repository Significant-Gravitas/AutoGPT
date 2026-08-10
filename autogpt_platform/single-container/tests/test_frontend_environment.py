from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SINGLE_CONTAINER_DIR = Path(__file__).resolve().parents[1]
RUN_FRONTEND_PATH = SINGLE_CONTAINER_DIR / "run-frontend.sh"
SUPERVISOR_PATH = SINGLE_CONTAINER_DIR / "supervisor" / "supervisord.conf"
BOOTSTRAP_PATH = SINGLE_CONTAINER_DIR / "bootstrap.sh"


class FrontendEnvironmentTest(unittest.TestCase):
    def test_preserves_only_frontend_runtime_settings(self) -> None:
        required = {
            "AGPT_SERVER_URL": "http://127.0.0.1:8006/api",
            "AGPT_WS_SERVER_URL": "ws://127.0.0.1:8001/ws",
            "AUTH_ALLOW_NEW_ACCOUNTS": "false",
            "AUTH_DB_SCHEMA": "platform",
            "AUTH_REQUIRE_EMAIL_VERIFICATION": "false",
            "BETTER_AUTH_INTERNAL_URL": "http://127.0.0.1:3001",
            "BETTER_AUTH_SECRET": "better-auth-secret",
            "BETTER_AUTH_URL": "https://autogpt.example.com",
        }
        optional = {
            "AUTH_CALLBACK_URL": "/auth/callback",
            "AUTH_DISCORD_CLIENT_ID": "discord-client",
            "AUTH_DISCORD_CLIENT_SECRET": "discord-secret",
            "AUTH_GITHUB_CLIENT_ID": "github-client",
            "AUTH_GITHUB_CLIENT_SECRET": "github-secret",
            "AUTH_GOOGLE_CLIENT_ID": "google-client",
            "AUTH_GOOGLE_CLIENT_SECRET": "value with spaces",
            "AUTH_SIGNUP_ALLOWLIST": "@example.com,admin@example.net",
            "OPENAI_API_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_KEY": "openai-transcription-fallback",
            "SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS": "30",
            "SUPABASE_JWT_SECRET": "legacy-bridge-secret",
            "TRANSCRIPTION_API_BASE_URL": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcription-secret",
            "TRANSCRIPTION_MODEL": "whisper-1",
        }
        forbidden = {
            "AUTOGPT_INTERNAL_SERVICE_TOKEN": "internal-service-token",
            "DATABASE_URL": "postgresql://postgres:db-password@127.0.0.1/postgres",  # pragma: allowlist secret
            "DB_PASS": "database-password",
            "DIRECT_URL": "postgresql://superuser:secret@127.0.0.1/postgres",
            "ENCRYPTION_KEY": "master-encryption-key",
            "GRAPHITI_FALKORDB_PASSWORD": "falkor-password",
            "JWT_VERIFY_KEY": "legacy-backend-secret",
            "POSTGRES_PASSWORD": "postgres-superuser-password",
            "RABBITMQ_DEFAULT_PASS": "rabbitmq-password",
            "REDIS_PASSWORD": "redis-password",
            "UNSUBSCRIBE_SECRET_KEY": "unsubscribe-secret",
            "VAPID_PRIVATE_KEY": "vapid-private-key",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            ready_file = Path(temporary_directory) / "ready"
            ready_file.touch()
            environment = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "AUTOGPT_ASSET_DIR": str(SINGLE_CONTAINER_DIR),
                "AUTOGPT_READY_FILE": str(ready_file),
                **required,
                **optional,
                **forbidden,
            }
            shell = (
                'source "$1"; build_frontend_environment; '
                '/usr/bin/env -i "${frontend_env[@]}" python3 -c '
                "'import json, os; print(json.dumps(dict(os.environ)))'"
            )
            result = subprocess.run(
                [
                    "bash",
                    "-Eeuo",
                    "pipefail",
                    "-c",
                    shell,
                    "bash",
                    str(RUN_FRONTEND_PATH),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=environment,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        child_environment = json.loads(result.stdout.splitlines()[-1])
        for name, value in {**required, **optional}.items():
            self.assertEqual(child_environment[name], value)
        self.assertEqual(
            child_environment["DATABASE_URL"],
            "postgresql:///postgres?host=%2Frun%2Fpostgresql&user=autogpt_frontend",
        )
        for name in forbidden:
            if name != "DATABASE_URL":
                self.assertNotIn(name, child_environment)
        self.assertNotEqual(
            child_environment["DATABASE_URL"], forbidden["DATABASE_URL"]
        )
        self.assertEqual(child_environment["HOME"], "/data/frontend-home")
        self.assertEqual(child_environment["XDG_CACHE_HOME"], "/data/cache/next")
        self.assertEqual(child_environment["USER"], "autogpt_frontend")
        self.assertEqual(child_environment["LOGNAME"], "autogpt_frontend")
        self.assertEqual(child_environment["PORT"], "3001")
        self.assertEqual(child_environment["HOSTNAME"], "127.0.0.1")
        self.assertEqual(child_environment["NODE_ENV"], "production")

    def test_rejects_missing_required_setting(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "-Eeuo",
                "pipefail",
                "-c",
                'source "$1"; build_frontend_environment',
                "bash",
                str(RUN_FRONTEND_PATH),
            ],
            check=False,
            capture_output=True,
            encoding="utf-8",
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "AUTOGPT_ASSET_DIR": str(SINGLE_CONTAINER_DIR),
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required frontend setting is missing", result.stderr)

    def test_supervisor_uses_scrubbed_frontend_launcher(self) -> None:
        supervisor_config = SUPERVISOR_PATH.read_text(encoding="utf-8")
        next_program = supervisor_config.split("[program:next]", 1)[1].split(
            "[program:nginx]", 1
        )[0]
        self.assertIn(
            "command=/opt/autogpt/single-container/run-frontend.sh", next_program
        )
        self.assertNotIn("run-app.sh", next_program)
        self.assertIn("user=root", next_program)

        launcher = RUN_FRONTEND_PATH.read_text(encoding="utf-8")
        self.assertIn("/usr/bin/env -i", launcher)
        self.assertLess(
            launcher.index("/usr/bin/env -i"), launcher.index("/usr/bin/setpriv")
        )
        self.assertIn("--reuid=autogpt_frontend", launcher)
        self.assertIn("--regid=autogpt_frontend", launcher)
        self.assertIn("--bounding-set=-all", launcher)

    def test_frontend_database_policy_is_explicitly_least_privilege(self) -> None:
        bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")
        policy = bootstrap.split("configure_frontend_database_role() {", 1)[1].split(
            "publish_readiness() {", 1
        )[0]

        for attribute in (
            "NOSUPERUSER",
            "NOCREATEDB",
            "NOCREATEROLE",
            "NOINHERIT",
            "NOREPLICATION",
            "NOBYPASSRLS",
            "PASSWORD NULL",
        ):
            self.assertIn(attribute, policy)
        for table in (
            'platform."UserAuthIdentity"',
            'platform."UserAuthSession"',
            'platform."UserAuthAccount"',
            'platform."UserAuthVerification"',
            'platform."UserAuthJwks"',
        ):
            self.assertEqual(policy.count(table), 1)
        self.assertIn('GRANT SELECT (id, email), UPDATE (email, "updatedAt")', policy)
        self.assertIn(
            "ALTER ROLE autogpt_frontend IN DATABASE postgres RESET ALL", policy
        )
        self.assertIn("REVOKE TEMPORARY ON DATABASE postgres FROM PUBLIC", policy)
        self.assertIn(
            "REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA platform", policy
        )
        self.assertIn("REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC", policy)
        self.assertNotIn("GRANT ALL", policy)
        self.assertNotIn("GRANT CREATE", policy)


if __name__ == "__main__":
    unittest.main()
