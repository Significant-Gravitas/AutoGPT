"""Create a disposable local account, drive the native app, verify persisted rows."""

import argparse
import json
import os
import secrets
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

MOBILE = Path(__file__).resolve().parents[1]
PROMPT = "Hello Otto, confirm this native chat is connected."
EXPECTED = "Your local workspace is connected."


def loopback_origin(value: str) -> str:
    url = urlsplit(value)
    if (
        url.scheme != "http"
        or url.hostname not in {"localhost", "127.0.0.1", "::1"}
        or url.username
        or url.password
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise argparse.ArgumentTypeError("Use an isolated loopback test service")
    return value.rstrip("/")


def request(origin, path, data=None, bearer=None):
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(
        origin + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
        return response.headers, json.loads(raw) if raw else None


def verify_persistence(auth_origin, api_origin, account):
    headers, _ = request(auth_origin, "/api/auth/sign-in/email", account)
    bearer = headers["set-auth-token"]
    try:
        _, body = request(auth_origin, "/api/auth/token", bearer=bearer)
        token = body["token"]
        _, page = request(api_origin, "/api/chat/sessions", bearer=token)
        assert len(page["sessions"]) == 1, "Expected exactly one conversation"
        session_id = page["sessions"][0]["id"]
        _, detail = request(
            api_origin, f"/api/chat/sessions/{session_id}", bearer=token
        )
        messages = detail["messages"]
        users = [m for m in messages if m["role"] == "user"]
        assistants = [m for m in messages if m["role"] == "assistant"]
        assert len(users) == 1 and users[0]["content"] == PROMPT
        assert len(assistants) == 1 and EXPECTED in assistants[0]["content"]
    finally:
        request(auth_origin, "/api/auth/sign-out", {}, bearer)


def redact_password(directory: Path, password: str):
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".xml", ".log", ".txt"}:
            content = path.read_text()
            if password in content:
                path.write_text(content.replace(password, "[REDACTED TEST PASSWORD]"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auth-origin", type=loopback_origin, default="http://127.0.0.1:18800"
    )
    parser.add_argument(
        "--api-origin", type=loopback_origin, default="http://127.0.0.1:18806"
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--maestro", default="maestro")
    args = parser.parse_args()
    os.umask(0o077)
    output = MOBILE / "e2e/artifacts" / f"chat-{uuid4()}"
    output.mkdir(parents=True, mode=0o700)
    account = {
        "email": f"native-{uuid4()}@example.test",
        "password": secrets.token_urlsafe(24),
    }
    headers, _ = request(
        args.auth_origin, "/api/auth/sign-up/email", {"name": "Native Test", **account}
    )
    signup_bearer = headers.get("set-auth-token")
    if signup_bearer:
        request(args.auth_origin, "/api/auth/sign-out", {}, signup_bearer)
    env = {
        **os.environ,
        "MAESTRO_AUTH_ORIGIN": args.auth_origin,
        "MAESTRO_API_ORIGIN": args.api_origin,
        "MAESTRO_EMAIL": account["email"],
        "MAESTRO_PASSWORD": account["password"],
        "MAESTRO_CLI_NO_ANALYTICS": "true",
        "MAESTRO_CLI_ANALYSIS_NOTIFICATION_DISABLED": "true",
    }
    try:
        with (output / "maestro.log").open("w") as log:
            result = subprocess.run(
                [
                    args.maestro,
                    "--device",
                    args.device,
                    "test",
                    "--format",
                    "junit",
                    "--output",
                    str(output / "results.xml"),
                    "--test-output-dir",
                    str(output / "maestro"),
                    "e2e/email-chat.yaml",
                ],
                cwd=MOBILE,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise SystemExit(
                f"Native journey failed; inspect private artifacts: {output}"
            )
        subprocess.run(
            [
                "python3",
                str(MOBILE / "scripts/check-junit.py"),
                str(output / "results.xml"),
            ],
            check=True,
        )
        verify_persistence(args.auth_origin, args.api_origin, account)
        print(
            f"Native login, streamed reply, relaunch and persisted row checks passed: {output}"
        )
    finally:
        redact_password(output, account["password"])


if __name__ == "__main__":
    main()
