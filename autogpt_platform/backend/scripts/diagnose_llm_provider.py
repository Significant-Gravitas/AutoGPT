"""Test the environment-selected chat provider without exposing its key."""

import asyncio
import json

from backend.util.llm.diagnostics import diagnose_chat_provider


async def main() -> int:
    result = await diagnose_chat_provider()
    print(json.dumps(result.model_dump(exclude_none=True), indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
