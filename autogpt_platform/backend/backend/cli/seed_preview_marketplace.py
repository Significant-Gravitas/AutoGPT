"""Seed a small synthetic marketplace into this environment's database.

    poetry run seed-preview-marketplace

Preview environments run this after their migrations, so the store opens with
synthetic creators, approved listings, reviews and skills instead of empty
lists. Idempotent; see :mod:`backend.api.features.store.preview_seed` for what
a run creates and what it leaves alone. Refuses to run with APP_ENV=prod.

Exit status: 0 when everything is seeded, 1 when some listing or skill could
not be, 2 when refused.
"""

import argparse
import asyncio
import json
import logging
import sys

from backend.api.features.store.preview_seed import seed_marketplace
from backend.data import db as database
from backend.util.settings import AppEnvironment, Settings

_DESCRIPTION = "Seed a small synthetic marketplace into this environment."


def main() -> None:
    argparse.ArgumentParser(description=_DESCRIPTION).parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_run()))


async def _run() -> int:
    if Settings().config.app_env == AppEnvironment.PRODUCTION:
        print("error: refusing to seed synthetic data into production", file=sys.stderr)
        return 2
    await database.connect()
    try:
        summary = await seed_marketplace()
    finally:
        await database.disconnect()
    print(json.dumps(summary.model_dump(), indent=2))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    main()
