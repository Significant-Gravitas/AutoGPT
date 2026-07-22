"""Real-API smoke test for the E2B desktop sandbox blocks.

Run manually (costs a few cents of E2B credits):
    poetry run python scripts/desktop_smoke_test.py

Reads E2B_API_KEY from backend/.env. Creates a desktop, verifies the live
stream URL responds, exercises input + files + suspend/resume + the
workspace volume, then destroys the sandbox. Prints timings.
"""

import asyncio
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from backend.blocks.desktop._api import WORKSPACE_PATH, DesktopSession

VOLUME_NAME = "autogpt-smoke-test"
TEST_FILE = f"{WORKSPACE_PATH}/persist-test.txt"


def timed(label: str, started: float, timings: dict[str, float]) -> None:
    timings[label] = round(time.monotonic() - started, 2)
    print(f"  {label}: {timings[label]}s")


async def main() -> None:
    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.environ["E2B_API_KEY"]
    timings: dict[str, float] = {}

    print("1) create desktop sandbox (with volume attempt)")
    t = time.monotonic()
    session, persistence = await DesktopSession.create(
        api_key=api_key,
        timeout_seconds=300,
        width=1280,
        height=720,
        volume_name=VOLUME_NAME,
    )
    timed("cold_start_ready", t, timings)
    print(f"  sandbox_id={session.sandbox_id}")
    print(f"  persistence={persistence.model_dump()}")

    try:
        print("2) start stream")
        t = time.monotonic()
        stream = await session.start_stream()
        timed("stream_start", t, timings)
        print(f"  stream_url={stream.url}")
        async with httpx.AsyncClient() as client:
            resp = await client.get(stream.url, timeout=15)
        print(f"  stream HTTP status: {resp.status_code}")

        print("3) input actions")
        t = time.monotonic()
        await session.click(button=1, x=640, y=360)
        timed("click_round_trip", t, timings)
        t = time.monotonic()
        shot = await session.screenshot_base64()
        timed("screenshot_round_trip", t, timings)
        print(f"  screenshot bytes(b64): {len(shot)}")

        print("4) workspace file write")
        await session.sandbox.files.write(TEST_FILE, "persistence check")

        print("5) suspend")
        t = time.monotonic()
        await session.pause()
        timed("suspend", t, timings)

        print("6) resume via connect")
        t = time.monotonic()
        resumed = await DesktopSession.connect(session.sandbox_id, api_key)
        timed("resume", t, timings)
        content = await resumed.sandbox.files.read(TEST_FILE)
        print(f"  file after resume: {content!r}")
        session = resumed

        if persistence.volume_mounted:
            print("7) cross-sandbox volume check")
            other, other_persistence = await DesktopSession.create(
                api_key=api_key,
                timeout_seconds=120,
                width=1024,
                height=768,
                volume_name=VOLUME_NAME,
            )
            try:
                cross = await other.sandbox.files.read(TEST_FILE)
                print(f"  volume file in second sandbox: {cross!r}")
            finally:
                await other.kill()
        else:
            print(f"7) volume not mounted, skipping ({persistence.warning})")
    finally:
        print("8) destroy")
        await session.kill()

    print("\nTimings:", timings)


if __name__ == "__main__":
    asyncio.run(main())
