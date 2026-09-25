"""Run preview/apply or preview-rollback/rollback against one environment."""

import argparse
import asyncio
import json
from pathlib import Path

from backend.api.features.store.catalog_release_load import load_release
from backend.api.features.store.catalog_release_model import Adoption, Preview
from backend.api.features.store.catalog_release_service import (
    apply_release,
    preview_release,
    preview_rollback,
    rollback_release,
)
from backend.data import db as database


async def run(args: argparse.Namespace) -> None:
    adoption = Adoption.model_validate_json(args.adoption.read_bytes())
    approved = (
        Preview.model_validate_json(args.approved.read_bytes())
        if args.approved
        else None
    )
    release = load_release(args.catalogue, args.revision) if args.catalogue else None
    await database.connect()
    try:
        if args.command == "preview" and release:
            output = (await preview_release(release, adoption)).model_dump_json(
                indent=2
            )
        elif args.command == "apply" and release and approved:
            output = json.dumps(
                {"release_id": await apply_release(release, adoption, approved)}
            )
        elif args.command == "preview-rollback":
            output = (
                await preview_rollback(args.release_id, adoption)
            ).model_dump_json(indent=2)
        elif args.command == "rollback" and approved:
            output = json.dumps(
                {
                    "release_id": await rollback_release(
                        args.release_id, adoption, approved
                    )
                }
            )
        else:
            raise ValueError("invalid catalogue operation")
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
    finally:
        await database.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "apply", "preview-rollback", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("--adoption", type=Path, required=True)
        command.add_argument("--output", type=Path)
        command.set_defaults(catalogue=None, approved=None)
        if name in {"preview", "apply"}:
            command.add_argument("--catalogue", type=Path, required=True)
            command.add_argument("--revision", required=True)
        else:
            command.add_argument("--release-id", required=True)
        if name in {"apply", "rollback"}:
            command.add_argument("--approved", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
