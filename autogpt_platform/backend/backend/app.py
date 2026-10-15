import logging
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from backend.copilot.chat_env import ChatEnvironmentError, check_chat_environment

load_dotenv()

if TYPE_CHECKING:
    from backend.util.process import AppProcess

logger = logging.getLogger(__name__)


def run_processes(*processes: "AppProcess", **kwargs):
    """
    Execute all processes in the app. The last process is run in the foreground.
    Includes enhanced error handling and process lifecycle management.

    The chat settings are checked first: a service never starts on a retired
    setting or a route missing what it needs.
    """
    try:
        for warning in check_chat_environment():
            logger.warning(warning)
    except ChatEnvironmentError as exc:
        raise SystemExit(str(exc)) from None
    try:
        # Run all processes except the last one in the background.
        for process in processes[:-1]:
            process.start(background=True, **kwargs)

        # Run the last process in the foreground.
        processes[-1].start(background=False, **kwargs)
    finally:
        for process in reversed(processes):
            try:
                process.stop()
            except Exception as e:
                logger.exception(f"[{process.service_name}] unable to stop: {e}")


def main(**kwargs):
    """
    Run all the processes required for the AutoGPT-server (REST and WebSocket APIs).
    """

    from backend.api.rest_api import AgentServer
    from backend.api.ws_api import WebsocketServer
    from backend.copilot.bot.app import CoPilotChatBridge
    from backend.copilot.executor.manager import CoPilotExecutor
    from backend.copilot.swap_service import SwapCredentialService
    from backend.data.db_manager import DatabaseManager
    from backend.executor import ExecutionManager, Scheduler
    from backend.executor.batch_executor import BatchExecutor
    from backend.notifications.notifications import NotificationManager
    from backend.platform_linking.manager import PlatformLinkingManager

    run_processes(
        DatabaseManager().set_log_level("warning"),
        SwapCredentialService().set_log_level("warning"),
        Scheduler(),
        # BatchExecutor polls submitted LLM batches (Anthropic today) and
        # dispatches results to caller-registered namespace handlers.
        # The dream pass uses it when an anthropic key is configured;
        # falls back to sync_baseline otherwise. Own subprocess so a
        # slow Anthropic batch poll never starves the Scheduler's cron
        # tick.
        BatchExecutor(),
        NotificationManager(),
        PlatformLinkingManager(),
        WebsocketServer(),
        AgentServer(),
        ExecutionManager(),
        CoPilotChatBridge(),
        CoPilotExecutor(),
        **kwargs,
    )


if __name__ == "__main__":
    main()
