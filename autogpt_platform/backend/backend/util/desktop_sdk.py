"""Build desktop templates and bound readiness retries in the pinned Desktop SDK."""

import argparse
import logging
import time
from collections.abc import Callable

from dotenv import load_dotenv
from e2b import (
    CommandExitException,
    CommandResult,
    Template,
    TimeoutException,
    default_build_logger,
    wait_for_url,
)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="autogpt-code-desktop")
    parser.add_argument("--cpu-count", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=8192)
    args = parser.parse_args()
    load_dotenv()
    Template.build(
        desktop_template(),
        alias=args.alias,
        cpu_count=args.cpu_count,
        memory_mb=args.memory_mb,
        on_build_logs=default_build_logger(),
    )


def desktop_template():
    return (
        Template()
        .from_template("code-interpreter-v1")
        .set_user("root")
        .set_envs({"DEBIAN_FRONTEND": "noninteractive", "DISPLAY": ":0"})
        .apt_install(
            [
                "xfce4",
                "xfce4-terminal",
                "xvfb",
                "x11-utils",
                "x11vnc",
                "net-tools",
                "dbus-x11",
                "xdotool",
                "scrot",
                "firefox-esr",
                "mousepad",
            ]
        )
        .run_cmd(
            "git clone https://github.com/e2b-dev/noVNC.git /opt/noVNC "
            "&& git -C /opt/noVNC checkout --detach 461b7f1ccb20755037d8995612e5fb08ed16f9e4"
        )
        .run_cmd(
            "git clone https://github.com/novnc/websockify.git /opt/noVNC/utils/websockify "
            "&& git -C /opt/noVNC/utils/websockify checkout --detach 99f83ca08390dc876b1b3580c210abea5b9f4edd"
        )
        .set_user("user")
        .set_workdir("/home/user")
        .set_start_cmd(
            "sudo systemctl start jupyter",
            wait_for_url("http://localhost:49999/health"),
        )
    )


if __name__ == "__main__":
    main()
