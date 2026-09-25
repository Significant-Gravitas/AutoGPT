"""Publish the skills catalog into this environment's marketplace.

    poetry run publish-skills-catalog [--dry-run] [--path DIR | --ref REF]

Runs right after ``prisma migrate deploy`` on every deploy, so an environment
serves the catalog's ``main`` as of that deploy. Idempotent: a second run
against the same commit changes nothing. Rollback is a run against an older
commit (``--ref <sha>``).

The catalog is downloaded from GitHub unless ``--path`` or
``SKILLS_CATALOG_PATH`` names a local checkout. See
:mod:`backend.api.features.store.skill_catalog_release` for the environment
variables and :mod:`backend.api.features.store.skill_catalog` for what a
publish does and does not touch.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from backend.api.features.store.skill_catalog import (
    CatalogConflictError,
    publish_catalog,
)
from backend.api.features.store.skill_catalog_release import (
    CatalogError,
    CatalogSource,
    fetch_catalog,
    load_release,
    local_revision,
    resolve_catalog,
)
from backend.data import db as database

logger = logging.getLogger(__name__)

_DESCRIPTION = "Publish the skills catalog into this environment's marketplace."


def main() -> None:
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        "--path", help="publish from this local checkout (also SKILLS_CATALOG_PATH)"
    )
    parser.add_argument(
        "--repo", help="GitHub repo, owner/name (also SKILLS_CATALOG_REPO)"
    )
    parser.add_argument("--ref", help="branch, tag or commit (also SKILLS_CATALOG_REF)")
    parser.add_argument(
        "--dry-run", action="store_true", help="report the plan without writing"
    )
    parser.add_argument(
        "--skip-experts",
        action="store_true",
        help="publish skills only; leave the expert roster templates alone",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_run(args)))


async def _run(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="skills-catalog-") as tmp:
        try:
            source = _source(args, Path(tmp))
            loaded = load_release(source.root)
        except CatalogError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        logger.info(
            f"Loaded {loaded.release_key} from {source.repository}@{source.revision[:12]}: "
            f"{len(loaded.packages)} packages, {len(loaded.experts)} experts"
        )
        await database.connect()
        try:
            summary = await publish_catalog(
                loaded,
                repository=source.repository,
                revision=source.revision,
                dry_run=args.dry_run,
                seed_experts=not args.skip_experts,
            )
        except CatalogConflictError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        finally:
            await database.disconnect()
    print(json.dumps(summary.model_dump(), indent=2))
    return 0


def _source(args: argparse.Namespace, into: Path) -> CatalogSource:
    if args.path:
        root = Path(args.path)
        return CatalogSource(
            repository=f"file:{root}", revision=local_revision(root), root=root
        )
    if args.repo or args.ref:
        repo = args.repo or os.environ.get("SKILLS_CATALOG_REPO") or ""
        ref = args.ref or os.environ.get("SKILLS_CATALOG_REF") or ""
        return fetch_catalog(
            into,
            repo=repo or "Significant-Gravitas/skills-catalog",
            ref=ref or "main",
        )
    return resolve_catalog(into)


if __name__ == "__main__":
    main()
