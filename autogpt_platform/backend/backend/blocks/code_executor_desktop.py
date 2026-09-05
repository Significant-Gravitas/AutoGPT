import asyncio
import logging

from e2b import AsyncSandbox, CommandExitException, TimeoutException
from e2b_desktop import Sandbox as DesktopSandbox

logger = logging.getLogger(__name__)


async def create_desktop_sandbox(
    api_key: str, template_id: str, timeout: int
) -> tuple[str, str]:
    template_id = template_id.strip()
    if not template_id or template_id in {"base", "desktop"}:
        raise ValueError(
            "Live view requires a template_id with both desktop dependencies and "
            "the E2B code interpreter. Build scripts/e2b_desktop/build.py first."
        )
    task = asyncio.create_task(
        asyncio.to_thread(_create_desktop_sandbox, api_key, template_id, timeout)
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            sandbox_id, _ = await task
        except Exception:
            pass
        else:
            await kill_desktop_sandbox(api_key, sandbox_id)
        raise


async def kill_desktop_sandbox(api_key: str, sandbox_id: str) -> None:
    try:
        await AsyncSandbox.kill(sandbox_id=sandbox_id, api_key=api_key)
    except Exception:
        logger.warning("Could not clean up desktop sandbox %s", sandbox_id)


def _create_desktop_sandbox(
    api_key: str, template_id: str, timeout: int
) -> tuple[str, str]:
    sandbox = DesktopSandbox.create(
        api_key=api_key, template=template_id, timeout=timeout
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
            logger.warning("Could not clean up desktop sandbox %s", sandbox.sandbox_id)
        raise


def _check_code_interpreter(sandbox: DesktopSandbox) -> None:
    try:
        sandbox.commands.run(
            "curl --fail --silent --max-time 10 http://localhost:49999/health >/dev/null",
            timeout=15,
        )
    except (CommandExitException, TimeoutException) as exc:
        raise ValueError(
            "The desktop template must also provide a running E2B code interpreter "
            "on port 49999. Build scripts/e2b_desktop/build.py or fix the custom "
            "template's interpreter startup command."
        ) from exc
