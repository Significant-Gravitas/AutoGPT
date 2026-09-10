import logging

import click


@click.group()
def store():
    """
    Group for marketplace maintenance commands
    """
    logging.disable(logging.INFO)


@store.command(name="backfill-categories")
@click.option("--apply", is_flag=True, help="Write the results. Without it, dry run.")
@click.option(
    "--limit",
    type=click.IntRange(min=0),
    default=None,
    help="Classify at most N listings.",
)
@click.option(
    "--concurrency", type=int, default=5, help="Classifier calls in flight at once."
)
def backfill_categories(apply: bool, limit: int | None, concurrency: int):
    """Give every store listing a canonical category.

    Folds legacy free-text values onto the canonical set by alias where the
    mapping is unambiguous, and asks the classifier for the rest. Idempotent:
    a listing already on a canonical category is never touched, so a re-run
    only picks up what the previous one could not classify.
    """
    import asyncio

    asyncio.run(_run_backfill(apply=apply, limit=limit, concurrency=concurrency))


async def _run_backfill(*, apply: bool, limit: int | None, concurrency: int) -> None:
    import asyncio

    import prisma.models

    from backend.api.features.store.categories import normalize_categories
    from backend.api.features.store.category_classifier import classify_category
    from backend.data.db import connect, disconnect

    await connect()
    try:
        versions = await prisma.models.StoreListingVersion.prisma().find_many(
            where={"isDeleted": False}, order={"createdAt": "asc"}
        )
        folds = {
            v.id: [c.value for c in normalize_categories(v.categories)]
            for v in versions
        }
        pending = [v for v in versions if not folds[v.id]]
        if limit is not None:
            pending = pending[:limit]

        mode = "" if apply else "  (dry run — nothing will be written)"
        print(
            f"{len(versions)} listing versions; {len(pending)} need the classifier, "
            f"{sum(1 for v in versions if folds[v.id] and folds[v.id] != v.categories)}"
            f" fold from legacy values{mode}"
        )

        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def classify(version: prisma.models.StoreListingVersion):
            async with semaphore:
                return version, await classify_category(
                    version.name, version.subHeading, version.description
                )

        classified = 0
        for version, category in await asyncio.gather(*map(classify, pending)):
            if category is None:
                print(f"  skip   {version.name!r}: no category")
                continue
            classified += 1
            print(f"  {category.value:<12} {version.name!r}")
            if apply:
                await prisma.models.StoreListingVersion.prisma().update(
                    where={"id": version.id}, data={"categories": [category.value]}
                )

        folded = 0
        for version in versions:
            canonical = folds[version.id]
            if not canonical or canonical == version.categories:
                continue
            folded += 1
            print(f"  fold   {version.name!r}: {version.categories} -> {canonical}")
            if apply:
                await prisma.models.StoreListingVersion.prisma().update(
                    where={"id": version.id}, data={"categories": canonical}
                )

        print(
            f"\n{classified} classified, {folded} folded, "
            f"{len(pending) - classified} still uncategorised"
        )
        if not apply:
            print("Dry run — re-run with --apply to write.")
    finally:
        await disconnect()
