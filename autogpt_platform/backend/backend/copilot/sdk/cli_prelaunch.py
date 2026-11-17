"""Start a turn's CLI process while the turn still builds its first message.

The CLI is a large native binary and takes most of a second to start. Its
options are known before the turn has written its query (the user context,
the budget block, attachments), so the turn spawns the process as soon as the
options are final and lets it boot during that work.

Only the OS process starts early. The attempt that claims it connects a
``ClaudeSDKClient`` over it as before, so the SDK's reader and its tool and
hook handlers are still created after the attempt's per-turn resets and see
the same context they always did. The process starts from the environment
every launch gets (``launch_options``), worked out as it spawns.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from backend.copilot.sdk.launch_env import launch_options

logger = logging.getLogger(__name__)

# Stops still running after their caller was cancelled (``_stop``), held until
# they finish: the loop keeps only weak references to tasks.
_stopping: set[asyncio.Task[None]] = set()


class CLIPrelaunch:
    """A CLI process spawned for one set of options, used by at most one launch."""

    def __init__(
        self, options: ClaudeAgentOptions, transport: SubprocessCLITransport
    ) -> None:
        self._options = options
        self._transport = transport
        self._spent = False

    @classmethod
    async def start(
        cls, options: ClaudeAgentOptions, log_prefix: str
    ) -> "CLIPrelaunch | None":
        """Spawn the CLI for ``options``; ``None`` if it can't be spawned
        (the attempt then spawns its own, as it would have anyway)."""
        transport = SubprocessCLITransport(
            prompt=_no_prompt(), options=launch_options(options)
        )
        connected = False
        try:
            await transport.connect()
            connected = True
        except Exception:
            logger.warning(
                "%s Couldn't start the CLI ahead of the query",
                log_prefix,
                exc_info=True,
            )
            return None
        finally:
            # Also when the turn is cancelled mid-spawn: nothing holds the
            # process yet, so nothing else would stop it.
            if not connected:
                await _stop(transport)
        return cls(options, transport)

    def claim(self, options: ClaudeAgentOptions) -> SubprocessCLITransport | None:
        """The running process, once, and only for the options it was spawned
        with: a retry rebuilds its options and spawns its own."""
        if self._spent or options is not self._options:
            return None
        self._spent = True
        return self._transport

    async def close(self) -> None:
        """Stop the process if no launch claimed it."""
        if self._spent:
            return
        self._spent = True
        await _stop(self._transport)


async def _stop(transport: SubprocessCLITransport) -> None:
    """Close ``transport`` in a task of its own and wait for it.

    Both callers usually run while their turn is being cancelled, and the SDK's
    ``close()`` defers only anyio cancellation: an asyncio cancellation reaching
    it mid-close skips the terminate/kill that ends the process. The shield
    lets the caller's cancellation go on while the close finishes.
    """
    stop = asyncio.create_task(transport.close())
    _stopping.add(stop)
    stop.add_done_callback(_forget_stop)
    with suppress(Exception):
        await asyncio.shield(stop)


def _forget_stop(stop: "asyncio.Task[None]") -> None:
    _stopping.discard(stop)
    if not stop.cancelled():
        stop.exception()  # retrieved: a failed close was already suppressed


async def _no_prompt() -> AsyncIterator[dict[str, Any]]:
    """What ``ClaudeSDKClient`` gives its own transport: the client writes each
    message itself, so the transport is never fed a prompt stream."""
    return
    yield {}
