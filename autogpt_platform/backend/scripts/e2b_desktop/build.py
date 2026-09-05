import argparse

from dotenv import load_dotenv
from e2b import Template, default_build_logger, wait_for_url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="autogpt-code-desktop")
    args = parser.parse_args()
    load_dotenv()
    Template.build(
        desktop_template(),
        alias=args.alias,
        cpu_count=8,
        memory_mb=8192,
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
        .git_clone(
            "https://github.com/e2b-dev/noVNC.git",
            "/opt/noVNC",
            branch="e2b-desktop",
        )
        .git_clone(
            "https://github.com/novnc/websockify.git",
            "/opt/noVNC/utils/websockify",
            branch="v0.12.0",
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
