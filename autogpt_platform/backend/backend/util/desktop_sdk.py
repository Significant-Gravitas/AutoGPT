"""Bound readiness retries in the pinned Desktop SDK, including failed probes."""

import logging
import time
from typing import Callable

from e2b import CommandExitException, CommandResult, TimeoutException
from e2b_desktop import Sandbox

logger = logging.getLogger(__name__)


class DesktopSandbox(Sandbox):
    def _wait_and_verify(
        self,
        cmd: str,
        on_result: Callable[[CommandResult], bool],
        timeout: int = 10,
        interval: float = 0.5,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                if on_result(self.commands.run(cmd, timeout=remaining)):
                    return True
            except (CommandExitException, TimeoutException):
                logger.debug("Desktop readiness probe has not succeeded")
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
        try:
            self.kill()
        except Exception:
            logger.warning(
                "Could not clean up desktop after startup timeout", exc_info=True
            )
        raise TimeoutException(
            "Desktop services did not become ready before the deadline"
        )
