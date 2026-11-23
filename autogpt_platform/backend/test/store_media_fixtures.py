"""Marketplace images the stack serves itself, for the official listings.

The listings in ``agents/StoreAgent_rows.csv`` and the official creator's
avatar point at production media on GCS. A stack seeded with them makes the
frontend's image optimizer fetch from GCS whenever a Marketplace page shows
them. ``load_store_agents --local-media`` saves the placeholder images in
``fixtures/store_media`` to the stack's own media storage instead, where an
upload goes on a stack without a bucket, and points the listings and the
avatar at them. The e2e stack seeds this way, so its pages load no image from
outside the stack.
"""

from pathlib import Path

from backend.api.features.store import local_media

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "store_media"
LISTING_IMAGES = ("listing-1.png", "listing-2.png", "listing-3.png")
AVATAR_IMAGE = "creator-avatar.png"


async def store_listing_images(owner_id: str, slug: str, count: int) -> list[str]:
    """A local URL for each of a listing's `count` images, each its own file,
    so a listing's gallery shows as many images as the CSV gives it. Named
    after the listing, not at random like an upload, so loading again saves
    the same files under the same URLs."""
    return [
        await _store(
            owner_id,
            f"{slug}-{index + 1}.png",
            LISTING_IMAGES[index % len(LISTING_IMAGES)],
        )
        for index in range(count)
    ]


async def store_avatar(owner_id: str) -> str:
    return await _store(owner_id, AVATAR_IMAGE, AVATAR_IMAGE)


async def _store(owner_id: str, filename: str, fixture: str) -> str:
    content = (FIXTURES_DIR / fixture).read_bytes()
    return await local_media.store_media(owner_id, "images", filename, content)
