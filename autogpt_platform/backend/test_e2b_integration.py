"""End-to-end integration test for the E2B sandbox lifecycle.

Tests the direct-to-sandbox architecture:
  1. get_or_create_sandbox  — create / reconnect via Redis
  2. bash_exec on E2B        — write files via sandbox.commands.run()
  3. E2B files API           — read/write files via sandbox.files
  4. Cross-turn persistence  — reconnect to the same sandbox and verify files

Run via docker-compose:
  docker compose --profile e2b-test -f docker-compose.platform.yml run --rm e2b_integration_test

Or directly (requires E2B_API_KEY and REDIS_HOST env vars):
  python test_e2b_integration.py
"""

import asyncio
import os
import sys
import uuid


async def main(e2b_api_key: str) -> None:
    from backend.copilot.tools.e2b_sandbox import get_or_create_sandbox

    # Use a UUID-based session_id to avoid Redis key collisions across reruns
    session_id = f"test-e2b-{uuid.uuid4().hex[:12]}"
    sbx = None
    print(f"session_id={session_id}")

    try:
        # ------------------------------------------------------------------
        # Turn 1: create sandbox, write files via bash + files API
        # ------------------------------------------------------------------
        print("\n[Turn 1] Creating sandbox...")
        sbx = await get_or_create_sandbox(
            session_id, api_key=e2b_api_key, template="base", timeout=300
        )
        print(f"  sandbox_id={sbx.sandbox_id}")

        print("[Turn 1] bash_exec — write file on E2B...")
        result = await sbx.commands.run(
            "echo 'hello from turn 1' > /home/user/turn1.txt && "
            "mkdir -p /home/user/subdir && echo 'nested' > /home/user/subdir/nested.txt"
        )
        assert result.exit_code == 0, f"bash failed: {result.stderr}"

        print("[Turn 1] files API — write file directly...")
        await sbx.files.write("/home/user/api_written.txt", "written via files API\n")

        # Verify bash-written file via files API
        content = await sbx.files.read("/home/user/turn1.txt", format="text")
        assert content.strip() == "hello from turn 1", f"Unexpected: {content!r}"
        print("  PASS: bash-written file readable via files API")

        # Verify API-written file via bash
        check = await sbx.commands.run("cat /home/user/api_written.txt")
        assert (
            check.stdout.strip() == "written via files API"
        ), f"Unexpected: {check.stdout!r}"
        print("  PASS: API-written file visible via bash")

        # ------------------------------------------------------------------
        # Turn 2: reconnect (same sandbox_id via Redis), verify persistence
        # ------------------------------------------------------------------
        print("\n[Turn 2] Reconnecting to sandbox via Redis...")
        sbx2 = await get_or_create_sandbox(
            session_id, api_key=e2b_api_key, template="base", timeout=300
        )
        assert (
            sbx2.sandbox_id == sbx.sandbox_id
        ), f"Expected same sandbox, got {sbx2.sandbox_id} != {sbx.sandbox_id}"
        print(f"  reconnected to same sandbox_id={sbx2.sandbox_id}")

        # Verify all files persist across turns
        for path, expected in [
            ("/home/user/turn1.txt", "hello from turn 1"),
            ("/home/user/subdir/nested.txt", "nested"),
            ("/home/user/api_written.txt", "written via files API"),
        ]:
            content = await sbx2.files.read(path, format="text")
            assert (
                content.strip() == expected
            ), f"{path}: expected {expected!r}, got {content!r}"
        print("  PASS: all files from turn 1 persist in turn 2")

        print("\n=== ALL TESTS PASSED ===")

    finally:
        # ------------------------------------------------------------------
        # Cleanup — always runs even if an assertion fails
        # ------------------------------------------------------------------
        print("\n[Cleanup] Killing sandbox...")
        if sbx is not None:
            try:
                await sbx.kill()
            except Exception as kill_err:
                print(f"  warning: sandbox kill failed: {kill_err}", file=sys.stderr)
        print("  done")


if __name__ == "__main__":
    _api_key = os.environ.get("E2B_API_KEY", "")
    if not _api_key:
        print("SKIP: E2B_API_KEY not set", file=sys.stderr)
        sys.exit(0)
    asyncio.run(main(_api_key))
