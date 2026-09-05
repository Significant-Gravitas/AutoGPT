"""Build a combined desktop and code-interpreter template for E2B code blocks."""

import argparse

from dotenv import load_dotenv
from e2b import Template, default_build_logger, wait_for_url


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
