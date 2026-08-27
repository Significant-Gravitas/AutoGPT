from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.api.features.orgs import routes


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["../avatar.png", "..\\avatar.png"])
async def test_get_org_avatar_rejects_path_segments(filename):
    with pytest.raises(HTTPException) as error:
        await routes.get_org_avatar("org-1", filename)

    assert error.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("org_id", ["../org-1", "..\\org-1"])
async def test_get_org_avatar_rejects_org_path_segments(org_id):
    with pytest.raises(HTTPException) as error:
        await routes.get_org_avatar(org_id, "avatar.png")

    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_get_org_avatar_serves_only_the_scoped_file(tmp_path, monkeypatch):
    avatar = tmp_path / "orgs" / "org-1" / "images" / "avatar.png"
    avatar.parent.mkdir(parents=True)
    avatar.write_bytes(b"avatar")
    monkeypatch.setattr(
        routes.store_media, "get_local_media_root", lambda: str(tmp_path)
    )

    response = await routes.get_org_avatar("org-1", "avatar.png")

    assert b"".join([chunk async for chunk in response.body_iterator]) == b"avatar"
    assert response.headers["content-length"] == "6"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.media_type == "image/png"


@pytest.mark.asyncio
async def test_get_org_avatar_rejects_symlink_outside_media_root(
    tmp_path, tmp_path_factory, monkeypatch
):
    outside = tmp_path_factory.mktemp("outside-media")
    avatar = outside / "images" / "avatar.png"
    avatar.parent.mkdir(parents=True)
    avatar.write_bytes(b"outside")
    orgs = tmp_path / "orgs"
    orgs.mkdir()
    try:
        (orgs / "org-1").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are not available")
    monkeypatch.setattr(
        routes.store_media, "get_local_media_root", lambda: str(tmp_path)
    )

    with pytest.raises(HTTPException) as error:
        await routes.get_org_avatar("org-1", "avatar.png")

    assert error.value.status_code == 404


def test_avatar_access_dependency_lives_through_streaming(tmp_path, monkeypatch):
    avatar = tmp_path / "orgs" / "org-1" / "images" / "avatar.png"
    avatar.parent.mkdir(parents=True)
    avatar.write_bytes(b"avatar")
    events: list[str] = []

    @asynccontextmanager
    async def allowed(*_args, **_kwargs):
        events.append("authorized")
        try:
            yield True
        finally:
            events.append("released")

    def stream(file):
        events.append("streamed")
        try:
            yield file.read()
        finally:
            file.close()

    monkeypatch.setattr(routes, "live_org_permission_barrier", allowed)
    monkeypatch.setattr(routes, "_stream_open_file", stream)
    monkeypatch.setattr(
        routes.store_media, "get_local_media_root", lambda: str(tmp_path)
    )
    app = FastAPI()
    app.include_router(routes.router, prefix="/orgs")
    app.dependency_overrides[routes.get_user_id] = lambda: "user-1"

    response = TestClient(app).get("/orgs/org-1/avatar/avatar.png")

    assert response.status_code == 200
    assert response.content == b"avatar"
    assert events == ["authorized", "streamed", "released"]


def test_denied_avatar_access_streams_no_bytes(tmp_path, monkeypatch):
    streamed = False

    @asynccontextmanager
    async def denied(*_args, **_kwargs):
        yield False

    def stream(_file):
        nonlocal streamed
        streamed = True
        yield b"forbidden"

    monkeypatch.setattr(routes, "live_org_permission_barrier", denied)
    monkeypatch.setattr(routes, "_stream_open_file", stream)
    monkeypatch.setattr(
        routes.store_media, "get_local_media_root", lambda: str(tmp_path)
    )
    app = FastAPI()
    app.include_router(routes.router, prefix="/orgs")
    app.dependency_overrides[routes.get_user_id] = lambda: "user-1"

    response = TestClient(app).get("/orgs/org-1/avatar/avatar.png")

    assert response.status_code == 403
    assert streamed is False
