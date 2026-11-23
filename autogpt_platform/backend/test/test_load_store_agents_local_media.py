"""`load_store_agents --local-media`, how the e2e stack loads the official
listings: every image they and their creator show is one the stack serves
itself, so no Marketplace page fetches production media from GCS."""

from test import load_store_agents, store_media_fixtures
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.store import local_media, routes
from backend.util.settings import Settings

PLATFORM_BASE_URL = "http://localhost:8006"
MEDIA_PREFIX = (
    f"{PLATFORM_BASE_URL}/api/store/media/{load_store_agents.AUTOGPT_USER_ID}/images/"
)


@pytest.fixture
def media_settings(monkeypatch, tmp_path):
    settings = Settings()
    settings.config.workspace_storage_dir = str(tmp_path / "workspaces")
    settings.config.platform_base_url = PLATFORM_BASE_URL
    monkeypatch.setattr(local_media, "Settings", lambda: settings)
    return settings


@pytest.fixture
def client():
    app = fastapi.FastAPI()
    app.include_router(routes.router, prefix="/api/store")
    with fastapi.testclient.TestClient(app) as test_client:
        yield test_client


def _served(client: fastapi.testclient.TestClient, url: str) -> bytes:
    assert url.startswith(MEDIA_PREFIX), url
    response = client.get(urlsplit(url).path)
    assert response.status_code == 200, url
    assert response.headers["content-type"] == "image/png"
    return response.content


async def test_every_listing_image_is_served_by_the_stack(media_settings, client):
    metadata = await load_store_agents.load_csv_metadata()
    csv_counts = {meta["slug"]: len(meta["agent_image"]) for meta in metadata.values()}
    # The CSV's own images, which the e2e stack must not fetch.
    assert sum(csv_counts.values()) > 0

    await load_store_agents.use_local_images(metadata)

    fixtures = {
        (store_media_fixtures.FIXTURES_DIR / name).read_bytes()
        for name in store_media_fixtures.LISTING_IMAGES
    }
    for meta in metadata.values():
        urls = meta["agent_image"]
        # As many images as the CSV gives the listing, each its own file.
        assert len(urls) == csv_counts[meta["slug"]]
        assert len(set(urls)) == len(urls)
        for url in urls:
            assert _served(client, url) in fixtures


async def test_the_creator_avatar_is_served_by_the_stack(media_settings, client):
    fixture = store_media_fixtures.FIXTURES_DIR / store_media_fixtures.AVATAR_IMAGE

    url = await store_media_fixtures.store_avatar(load_store_agents.AUTOGPT_USER_ID)

    assert _served(client, url) == fixture.read_bytes()


def _earlier_load() -> AsyncMock:
    """A database a load without --local-media already filled: the official
    profile and listings exist, with the CSV's production image URLs."""
    db = AsyncMock()
    db.user.find_unique.return_value = SimpleNamespace(
        id=load_store_agents.AUTOGPT_USER_ID
    )
    db.profile.find_first.return_value = SimpleNamespace(
        id="profile-1", avatarUrl=load_store_agents.AUTOGPT_AVATAR_URL
    )
    db.storelisting.find_unique.return_value = SimpleNamespace(id="listing-1")
    return db


async def _a_listing_with_local_images() -> dict:
    metadata = await load_store_agents.load_csv_metadata()
    await load_store_agents.use_local_images(metadata)
    return next(meta for meta in metadata.values() if meta["agent_image"])


async def test_loading_again_with_local_media_repoints_an_earlier_load(
    media_settings, client
):
    db = _earlier_load()
    listing = await _a_listing_with_local_images()
    avatar = await store_media_fixtures.store_avatar(load_store_agents.AUTOGPT_USER_ID)

    await load_store_agents.create_user_and_profile(db, avatar, local_media=True)
    await load_store_agents.create_store_listing(
        db, "graph-1", 1, listing, local_media=True
    )

    db.profile.update.assert_awaited_once_with(
        where={"id": "profile-1"}, data={"avatarUrl": avatar}
    )
    db.storelistingversion.update_many.assert_awaited_once_with(
        where={"id": listing["store_listing_version_id"]},
        data={"imageUrls": listing["agent_image"]},
    )
    for url in [avatar, *listing["agent_image"]]:
        _served(client, url)
    # Nothing is created twice.
    db.profile.create.assert_not_awaited()
    db.storelisting.create.assert_not_awaited()
    db.storelistingversion.create.assert_not_awaited()


async def test_loading_again_without_local_media_leaves_an_earlier_load_alone(
    media_settings,
):
    db = _earlier_load()
    listing = (await load_store_agents.load_csv_metadata()).popitem()[1]

    await load_store_agents.create_user_and_profile(db)
    await load_store_agents.create_store_listing(db, "graph-1", 1, listing)

    db.profile.update.assert_not_awaited()
    db.storelistingversion.update_many.assert_not_awaited()
    db.profile.create.assert_not_awaited()
    db.storelisting.create.assert_not_awaited()
