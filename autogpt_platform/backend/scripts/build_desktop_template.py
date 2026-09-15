"""Build the platform's sandbox image on an E2B team ahead of first use.

The backend builds ``agpt-desktop-1x2`` itself the first time a team needs
it (see ``backend.util.e2b_template``).  Run this to do it up front, or to
build an experimental size under another alias:

    poetry run build-desktop-template
    poetry run build-desktop-template --alias agpt-desktop-2x2 --cpu 2 --mem 2048

Reads ``E2B_API_KEY`` from the environment or ``backend/.env``; the template
lands on whichever team that key belongs to.
"""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from backend.util.e2b_template import DESKTOP_IMAGE, TemplateSpec, build_template


def main() -> None:
    args = _parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        raise SystemExit("E2B_API_KEY must be set in the environment or backend/.env")
    spec = TemplateSpec(
        alias=args.alias,
        source=DESKTOP_IMAGE.source,
        cpu_count=args.cpu,
        memory_mb=args.mem,
    )
    info = asyncio.run(build_template(spec, api_key))
    print(f"Built {info.name} ({info.template_id}) tags={info.tags}")
    if spec != DESKTOP_IMAGE:
        print(f"Set CHAT_E2B_SANDBOX_TEMPLATE={info.name} to use it.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the platform's sandbox image on an E2B team."
    )
    parser.add_argument("--alias", default=DESKTOP_IMAGE.alias)
    parser.add_argument("--cpu", type=int, default=DESKTOP_IMAGE.cpu_count)
    parser.add_argument("--mem", type=int, default=DESKTOP_IMAGE.memory_mb)
    return parser.parse_args()


if __name__ == "__main__":
    main()
