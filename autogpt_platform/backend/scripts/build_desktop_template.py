"""Build the platform's desktop sandbox template on an E2B team.

E2B fixes a sandbox's vCPU / RAM at template build time, and its public
``desktop`` template is 8 vCPU / 8 GiB — roughly $0.53 an hour running.  This
re-snapshots that template at a size we choose and tags the build so the
image itself carries its provenance (``e2b template list`` shows the tags).

Run once per E2B team, then point ``CHAT_E2B_DESKTOP_TEMPLATE`` at the alias:

    poetry run build-desktop-template                 # agpt-desktop-1x1, 1 vCPU / 1 GiB
    poetry run build-desktop-template --cpu 2 --mem 2048 --alias agpt-desktop-2x2

Reads ``E2B_API_KEY`` from the environment or ``backend/.env``.  A plain alias
resolves to the ``default`` tag, so the build is tagged ``default`` as well as
its size, the git revision and the caller.
"""

import argparse
import getpass
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from e2b import Template, default_build_logger

DEFAULT_ALIAS = "agpt-desktop-1x1"
SOURCE_TEMPLATE = "desktop"


def main() -> None:
    args = _parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    tags = ["default", f"{args.cpu}x{args.mem // 1024}", f"from:{SOURCE_TEMPLATE}"]
    tags += [f"git:{_git_revision()}", f"by:{getpass.getuser()}"]
    info = Template.build(
        Template().from_template(SOURCE_TEMPLATE),
        args.alias,
        tags=tags,
        cpu_count=args.cpu,
        memory_mb=args.mem,
        on_build_logs=default_build_logger(),
    )
    print(f"Built {info.alias} ({info.template_id}) tags={info.tags}")
    print(f"Set CHAT_E2B_DESKTOP_TEMPLATE={info.alias} to use it.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the platform's desktop sandbox template on an E2B team."
    )
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--cpu", type=int, default=1, help="vCPU count")
    parser.add_argument("--mem", type=int, default=1024, help="RAM in MiB")
    return parser.parse_args()


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


if __name__ == "__main__":
    main()
