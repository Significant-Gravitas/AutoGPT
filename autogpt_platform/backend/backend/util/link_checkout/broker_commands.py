"""The agent's browser commands, narrowed to what the browser tools send.

The agent drives the same browser a payment later happens in, so it gets
navigation and form input but no script evaluation, cookie or storage export,
network capture, console or raw CDP: nothing that could plant a listener for,
or read back, a card the worker fills. A sealed browser runs nothing at all.
"""

import base64
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from backend.util.link_checkout.broker_protocol import (
    BrowserCommand,
    BrowserOutput,
    session_key,
)
from backend.util.link_checkout.runtime import browser_command, browser_operation


async def execute_browser(request: BrowserCommand) -> BrowserOutput:
    args = request.args
    validate_command(args)
    async with browser_operation(session_key(request)) as directory:
        if args[0] == "screenshot":
            return await screenshot(directory, "--annotate" in args)
        code, output, error = await browser_command(directory, *args)
        return BrowserOutput(code=code, output=output, error=error)


def validate_command(args: list[str]) -> None:
    if not args or any(len(arg) > 16000 or "\x00" in arg for arg in args):
        raise ValueError("Unsupported browser command")
    verb = args[0]
    if verb == "open" and len(args) == 2:
        parsed = urlsplit(args[1])
        if (
            parsed.scheme == "https"
            and parsed.hostname
            and parsed.port in {None, 443}
            and not parsed.username
            and not parsed.password
        ):
            return
    if args in (
        ["get", "url"],
        ["get", "title"],
        ["snapshot", "-i"],
        ["snapshot", "-i", "--json"],
        ["close"],
    ):
        return
    if verb == "snapshot" and args[1:] in (["-i", "-c"], ["-i", "-c", "--json"]):
        return
    if verb == "screenshot" and args[1:] in ([], ["--annotate"]):
        return
    if verb in {"back", "forward", "reload"} and len(args) == 1:
        return
    if (
        verb == "wait"
        and len(args) == 3
        and args[1] == "--load"
        and args[2] in {"networkidle", "load", "domcontentloaded"}
    ):
        return
    if (
        verb == "scroll"
        and len(args) == 2
        and args[1] in {"up", "down", "left", "right"}
    ):
        return
    if (
        verb in {"click", "dblclick", "hover", "check", "uncheck", "wait"}
        and len(args) == 2
        and safe_argument(args[1])
    ):
        return
    if (
        verb == "press"
        and len(args) == 2
        and args[1]
        in {
            "Enter",
            "Tab",
            "Shift+Tab",
            "Escape",
            "Space",
            "ArrowUp",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
            "Backspace",
            "Delete",
            "Home",
            "End",
        }
    ):
        return
    if (
        verb in {"fill", "type", "select"}
        and len(args) == 3
        and all(safe_argument(arg) for arg in args[1:])
    ):
        return
    raise ValueError("Unsupported browser command")


def safe_argument(value: str) -> bool:
    return bool(value) and not value.startswith("-")


async def screenshot(directory: Path, annotate: bool) -> BrowserOutput:
    path = directory / f"capture-{uuid.uuid4().hex}.png"
    try:
        args = ["screenshot", str(path), *(["--annotate"] if annotate else [])]
        code, _, _ = await browser_command(directory, *args)
        if code or path.stat().st_size > 8_000_000:
            raise RuntimeError("Capture unavailable")
        return BrowserOutput(code=0, image=base64.b64encode(path.read_bytes()).decode())
    finally:
        path.unlink(missing_ok=True)
