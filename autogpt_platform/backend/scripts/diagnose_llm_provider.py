"""Test the environment-selected chat provider without exposing its key."""

import asyncio
import argparse
import json

from pydantic import SecretStr

from backend.data import db
from backend.util.llm.diagnostics import diagnose_chat_provider
from backend.util.llm.runtime_config import LlmRuntimeOverrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="Ignore persisted database settings.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    overrides = LlmRuntimeOverrides(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=SecretStr(args.api_key) if args.api_key else None,
    )
    connected = False
    if not args.env_only:
        try:
            await db.connect()
            connected = True
        except Exception:
            connected = False
    try:
        result = await diagnose_chat_provider(
            overrides=overrides,
            use_persisted=not args.env_only,
        )
    finally:
        if connected:
            await db.disconnect()
    print(json.dumps(result.model_dump(exclude_none=True), indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
