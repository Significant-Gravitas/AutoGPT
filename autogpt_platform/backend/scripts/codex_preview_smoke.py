"""Verify a local ChatGPT login against the real backend, over HTTP.

Reads ``~/.codex/auth.json`` (read-only -- nothing is written back) and checks
the account, the model catalog, and one real subscription-backed turn with a
tool call.

This consumes a small amount of the connected account's usage.

Run from ``autogpt_platform/backend``::

    poetry run python -m scripts.codex_preview_smoke
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic import SecretStr

from backend.integrations.codex.chatgpt_auth import ChatGPTTokens, bundle_from_tokens
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.http_client import account_snapshot, fetch_models
from backend.integrations.codex.http_session import CodexHttpSession
from backend.integrations.codex.models import (
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
)

_PROBE_TOOL = CodexDynamicToolSpec(
    name="get_temperature",
    description="Return the current temperature in celsius for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
)


def _default_auth_path() -> Path:
    home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    return Path(home) / ".codex" / "auth.json"


def _load_credentials(auth_path: Path):
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    tokens = payload.get("tokens") or {}
    missing = [
        k for k in ("id_token", "access_token", "refresh_token") if not tokens.get(k)
    ]
    if missing:
        raise SystemExit(f"{auth_path.name} is missing: {', '.join(missing)}")
    return credentials_from_bundle(
        bundle_from_tokens(
            ChatGPTTokens(
                id_token=SecretStr(tokens["id_token"]),
                access_token=SecretStr(tokens["access_token"]),
                refresh_token=SecretStr(tokens["refresh_token"]),
            )
        )
    )


async def _run(auth_path: Path, model: str | None) -> None:
    credentials = _load_credentials(auth_path)

    account = account_snapshot(credentials)
    if not account.connected:
        raise SystemExit("The stored ChatGPT token could not be read")
    print(f"account: {account.email or '(no email)'} on plan {account.plan_type}")

    models = await fetch_models(credentials)
    if not models:
        raise SystemExit("ChatGPT advertised no models for this account")
    default = next((m.model for m in models if m.is_default), models[0].model)
    chosen = model or default
    if chosen not in {m.model for m in models}:
        raise SystemExit(f"Model not available on this account: {chosen}")
    print(f"models: {len(models)} available, default {default}, using {chosen}")

    calls: list[str] = []

    async def handler(call: CodexDynamicToolCall) -> CodexDynamicToolResult:
        calls.append(call.tool)
        return CodexDynamicToolResult(content=json.dumps({"celsius": 12}))

    session = CodexHttpSession(
        credentials, turn_timeout_seconds=180, tool_timeout_seconds=60
    )
    result = await session.invoke(
        CodexInvocationRequest(
            prompt="What is the temperature in Paris? Use the tool, then answer.",
            instructions="Be brief.",
            model=chosen,
        ),
        [_PROBE_TOOL],
        handler,
    )

    if not calls:
        raise SystemExit("The model never called the tool -- tool calling is broken")
    if not result.final_response.strip():
        raise SystemExit("The turn completed with no text")
    print(f"turn: {result.status} in {result.duration_ms} ms, tools called {calls}")
    print(f"reply: {result.final_response.strip()[:120]}")
    if result.usage:
        print(f"usage: {result.usage.total_tokens} tokens")

    limits = session.rate_limits
    if limits and limits.primary:
        print(
            f"quota: {limits.primary.used_percent}% of a "
            f"{limits.primary.window_duration_mins} min window ({limits.plan_type})"
        )

    print("codex-http-preview-ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-path", type=Path, default=_default_auth_path())
    parser.add_argument("--model", default=None, help="Defaults to the account default")
    args = parser.parse_args()

    if not args.auth_path.is_file():
        raise SystemExit(f"No ChatGPT login found at {args.auth_path}")
    try:
        asyncio.run(_run(args.auth_path, args.model))
    except SystemExit:
        raise
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
