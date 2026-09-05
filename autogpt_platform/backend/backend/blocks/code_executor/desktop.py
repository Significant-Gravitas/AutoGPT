"""Desktop provisioning runs off-loop; cancellation schedules cleanup of late results."""

import asyncio
import logging
from asyncio import to_thread
from typing import TYPE_CHECKING

from e2b import AsyncSandbox, CommandExitException, TimeoutException

if TYPE_CHECKING:
    from backend.util.desktop_sdk import DesktopSandbox

logger = logging.getLogger(__name__)
_cleanup_tasks: set[asyncio.Task[None]] = set()
INTERPRETER_PORT = 49999
INTERPRETER_RETRY_TIMEOUT = 10
INTERPRETER_REQUEST_TIMEOUT = 10
INTERPRETER_COMMAND_TIMEOUT = (
    INTERPRETER_RETRY_TIMEOUT + INTERPRETER_REQUEST_TIMEOUT + 5
)
CLEANUP_TIMEOUT = 10
TEMPLATE_DOCS_URL = "https://docs.agpt.co/integrations/block-integrations/misc/#instantiate-code-sandbox"


async def create_desktop_sandbox(
    api_key: str, template_id: str, timeout: int
) -> tuple[str, str]:
    template_id = template_id.strip()
    if not template_id or template_id.casefold() in {"base", "desktop"}:
        raise ValueError(
            "Live view requires a template_id with both desktop dependencies and "
            f"the E2B code interpreter. See {TEMPLATE_DOCS_URL}"
        )
    task = asyncio.create_task(
        to_thread(_create_desktop_sandbox, api_key, template_id, timeout)
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(_cleanup_cancelled_creation(task, api_key))
        _cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(_cleanup_tasks.discard)
        raise


async def _cleanup_cancelled_creation(
    task: asyncio.Task[tuple[str, str]], api_key: str
) -> None:
    try:
        sandbox_id, _ = await task
    except Exception:
        logger.warning("Desktop provisioning failed after cancellation", exc_info=True)
        return
    await kill_desktop_sandbox(api_key, sandbox_id)


async def kill_desktop_sandbox(api_key: str, sandbox_id: str) -> None:
    task = asyncio.create_task(_kill_desktop_sandbox(api_key, sandbox_id))
    _cleanup_tasks.add(task)
    task.add_done_callback(_cleanup_tasks.discard)
    await asyncio.shield(task)


async def _kill_desktop_sandbox(api_key: str, sandbox_id: str) -> None:
    try:
        await asyncio.wait_for(
            AsyncSandbox.kill(
                sandbox_id=sandbox_id, api_key=api_key, request_timeout=CLEANUP_TIMEOUT
            ),
            timeout=CLEANUP_TIMEOUT,
        )
    except Exception:
        logger.warning(
            "Could not clean up desktop sandbox %s", sandbox_id, exc_info=True
        )


def _create_desktop_sandbox(
    api_key: str, template_id: str, timeout: int
) -> tuple[str, str]:
    from backend.util.desktop_sdk import DesktopSandbox

    sandbox = DesktopSandbox.create(
        api_key=api_key, template=template_id, timeout=timeout, request_timeout=10
    )
    try:
        _check_code_interpreter(sandbox)
        sandbox.stream.start(require_auth=True)
        live_url = sandbox.stream.get_url(
            auth_key=sandbox.stream.get_auth_key(), view_only=False
        )
        return sandbox.sandbox_id, live_url
    except BaseException:
        try:
            sandbox.kill()
        except Exception:
            logger.warning(
                "Could not clean up desktop sandbox %s",
                sandbox.sandbox_id,
                exc_info=True,
            )
        raise


def _check_code_interpreter(sandbox: "DesktopSandbox") -> None:
    try:
        sandbox.commands.run(
            f"curl --fail --silent --retry 10 --retry-connrefused --retry-delay 1 "
            f"--retry-max-time {INTERPRETER_RETRY_TIMEOUT} --max-time {INTERPRETER_REQUEST_TIMEOUT} "
            f"http://localhost:{INTERPRETER_PORT}/health >/dev/null",
            timeout=INTERPRETER_COMMAND_TIMEOUT,
        )
    except (CommandExitException, TimeoutException) as exc:
        raise ValueError(
            "The desktop template must also provide a running E2B code interpreter "
            f"on port {INTERPRETER_PORT}. See {TEMPLATE_DOCS_URL}"
        ) from exc
