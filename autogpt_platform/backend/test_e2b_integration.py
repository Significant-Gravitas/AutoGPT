"""End-to-end integration test for the E2B sandbox lifecycle.

Tests the full turn lifecycle:
  1. get_or_create_sandbox  — create / reconnect via Redis
  2. sync_from_sandbox       — download E2B /home/user → local dir
  3. bash_exec on E2B        — write a file via sandbox.commands.run()
  4. sync_to_sandbox         — upload local dir → E2B /home/user
  5. Cross-turn persistence  — reconnect to the same sandbox and verify files

Run via docker-compose:
  docker compose -f docker-compose.platform.yml run --rm e2b_integration_test

Or directly (requires E2B_API_KEY and REDIS_HOST env vars):
  python test_e2b_integration.py
"""

import asyncio
import os
import sys
import tempfile

E2B_API_KEY = os.environ.get("E2B_API_KEY", "")
if not E2B_API_KEY:
    print("SKIP: E2B_API_KEY not set", file=sys.stderr)
    sys.exit(0)


async def main() -> None:
    from backend.copilot.tools.e2b_sandbox import (
        get_or_create_sandbox,
        sync_from_sandbox,
        sync_to_sandbox,
    )

    session_id = f"test-{os.getpid()}"
    local_dir = tempfile.mkdtemp(prefix="e2b-test-")
    print(f"session_id={session_id}  local_dir={local_dir}")

    # ------------------------------------------------------------------
    # Turn 1: create sandbox, write a file via bash, sync back down
    # ------------------------------------------------------------------
    print("\n[Turn 1] Creating sandbox...")
    sbx = await get_or_create_sandbox(
        session_id, api_key=E2B_API_KEY, template="base", timeout=300
    )
    print(f"  sandbox_id={sbx.sandbox_id}")

    print("[Turn 1] sync_from_sandbox (should be empty)...")
    await sync_from_sandbox(sbx, local_dir)
    files_after_sync = [f for f in os.listdir(local_dir) if not f.startswith(".")]
    print(f"  local files after sync_from: {files_after_sync}")
    assert files_after_sync == [], f"Expected empty, got {files_after_sync}"

    print("[Turn 1] bash_exec — write file on E2B...")
    result = await sbx.commands.run(
        "echo 'hello from turn 1' > /home/user/turn1.txt && "
        "mkdir -p /home/user/subdir && echo 'nested' > /home/user/subdir/nested.txt"
    )
    assert result.exit_code == 0, f"bash failed: {result.stderr}"

    print("[Turn 1] Write a file locally (simulates SDK Write tool)...")
    with open(os.path.join(local_dir, "sdk_written.txt"), "w") as f:
        f.write("written by SDK tool in turn 1\n")

    print("[Turn 1] sync_to_sandbox — upload local → E2B...")
    await sync_to_sandbox(sbx, local_dir)

    # Verify SDK-written file is now on E2B
    check = await sbx.commands.run("cat /home/user/sdk_written.txt")
    assert (
        check.stdout.strip() == "written by SDK tool in turn 1"
    ), f"Unexpected: {check.stdout!r}"
    print("  PASS: SDK-written file visible on E2B")

    # ------------------------------------------------------------------
    # Turn 2: reconnect (same sandbox_id via Redis), sync down, verify
    # ------------------------------------------------------------------
    print("\n[Turn 2] Reconnecting to sandbox via Redis...")
    local_dir2 = tempfile.mkdtemp(prefix="e2b-test-turn2-")
    sbx2 = await get_or_create_sandbox(
        session_id, api_key=E2B_API_KEY, template="base", timeout=300
    )
    assert (
        sbx2.sandbox_id == sbx.sandbox_id
    ), f"Expected same sandbox, got {sbx2.sandbox_id} != {sbx.sandbox_id}"
    print(f"  reconnected to same sandbox_id={sbx2.sandbox_id}")

    print("[Turn 2] sync_from_sandbox — download sandbox → local_dir2...")
    await sync_from_sandbox(sbx2, local_dir2)

    def read(path: str) -> str:
        with open(path) as f:
            return f.read().strip()

    assert (
        read(os.path.join(local_dir2, "turn1.txt")) == "hello from turn 1"
    ), "turn1.txt not synced correctly"
    assert (
        read(os.path.join(local_dir2, "subdir", "nested.txt")) == "nested"
    ), "subdir/nested.txt not synced"
    assert (
        read(os.path.join(local_dir2, "sdk_written.txt"))
        == "written by SDK tool in turn 1"
    ), "sdk_written.txt not synced"
    print("  PASS: all files from turn 1 visible in turn 2 local dir")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("\n[Cleanup] Killing sandbox...")
    await sbx.kill()
    import shutil

    shutil.rmtree(local_dir, ignore_errors=True)
    shutil.rmtree(local_dir2, ignore_errors=True)
    print("  done")

    print("\n=== ALL TESTS PASSED ===")


asyncio.run(main())
