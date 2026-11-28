"""Each packaged deployment tells the frontend's server its own address.

The frontend's server calls its own login service: the admin pages' session
check, the server-side token mint and impersonation. Behind a proxy that ends
TLS, the address Next gives a request is https on its own plain-http port, so
a call there fails and an admin is sent Home (#17339). BETTER_AUTH_INTERNAL_URL
names the address the server listens on, and these calls use it when set.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from test_compose_internal_ports import _key_items, _services
from test_entrypoint import _configure

ASSET_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ASSET_DIR.parent / "docker-compose.platform.yml"
RUN_FRONTEND_PATH = ASSET_DIR / "run-frontend.sh"
# The other settings run-frontend.sh requires, as placeholders.
LAUNCHER_SETTINGS = {
    "AGPT_SERVER_URL": "http://127.0.0.1:8006/api",
    "AGPT_WS_SERVER_URL": "ws://127.0.0.1:8001/ws",
    "AUTH_ALLOW_NEW_ACCOUNTS": "false",
    "AUTH_DB_SCHEMA": "platform",
    "AUTH_REQUIRE_EMAIL_VERIFICATION": "false",
    "BETTER_AUTH_SECRET": "placeholder",  # pragma: allowlist secret
    "BETTER_AUTH_URL": "https://autogpt.example.com",
}


class FrontendInternalURLTest(unittest.TestCase):
    def test_compose_names_the_port_the_frontend_serves(self) -> None:
        frontend = _services(COMPOSE_PATH.read_text(encoding="utf-8"))["frontend"]
        internal_url = _environment(frontend).get("BETTER_AUTH_INTERNAL_URL")
        self.assertIsNotNone(
            internal_url,
            "set BETTER_AUTH_INTERNAL_URL: http://localhost:<the frontend's "
            "container port> in the frontend service's environment: mapping",
        )
        internal = urlsplit(internal_url or "")
        container_ports = {
            entry.rsplit(":", 1)[-1] for entry in _key_items(frontend, "ports") or []
        }

        self.assertEqual(internal.scheme, "http")
        self.assertIn(internal.hostname, {"localhost", "127.0.0.1"})
        self.assertIn(str(internal.port), container_ports)

    def test_single_container_names_the_address_next_listens_on(self) -> None:
        configured = _configure(output="${BETTER_AUTH_INTERNAL_URL:-}")
        self.assertEqual(configured.returncode, 0, configured.stderr)
        internal_url = configured.stdout.splitlines()[-1]
        self.assertTrue(
            internal_url,
            "export BETTER_AUTH_INTERNAL_URL=http://<HOSTNAME>:<PORT>, the "
            "address run-frontend.sh gives Next, in configure_environment",
        )
        next_environment = _next_environment(BETTER_AUTH_INTERNAL_URL=internal_url)
        internal = urlsplit(next_environment["BETTER_AUTH_INTERNAL_URL"])

        self.assertEqual(internal.scheme, "http")
        self.assertEqual(internal.hostname, next_environment["HOSTNAME"])
        self.assertEqual(str(internal.port), next_environment["PORT"])

    def test_the_scan_reads_the_environment_mapping(self) -> None:
        """Guards the line scan itself."""
        block = [
            "    environment:",
            "      # a comment",
            "      FIRST: http://localhost:3000",
            '      QUOTED: "65000"',
            "",
            "    networks:",
            "      LATER: not the environment",
        ]
        self.assertEqual(
            _environment(block), {"FIRST": "http://localhost:3000", "QUOTED": "65000"}
        )
        self.assertEqual(_environment(["    environment:", "      - A=b"]), {})


def _environment(block: list[str]) -> dict[str, str]:
    """A service's `environment:` written as a mapping of `NAME: value`."""
    values: dict[str, str] = {}
    inside = False
    for line in block:
        if re.match(r"^    environment:\s*$", line):
            inside = True
            continue
        if not inside or not line.strip() or line.lstrip().startswith("#"):
            continue
        entry = re.match(r"^      ([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$", line)
        if not entry:
            break
        values[entry.group(1)] = entry.group(2).strip("\"'")
    return values


def _next_environment(**settings: str) -> dict[str, str]:
    """The environment run-frontend.sh starts Next with."""
    result = subprocess.run(
        [
            "bash",
            "-Eeuo",
            "pipefail",
            "-c",
            'source "$1"; build_frontend_environment; '
            '/usr/bin/env -i "${frontend_env[@]}" python3 -c '
            "'import json, os; print(json.dumps(dict(os.environ)))'",
            "bash",
            str(RUN_FRONTEND_PATH),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "AUTOGPT_ASSET_DIR": str(ASSET_DIR),
            **LAUNCHER_SETTINGS,
            **settings,
        },
    )
    return json.loads(result.stdout.splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
