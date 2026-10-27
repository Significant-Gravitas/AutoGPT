"""Files go by the account's own prefixes in every configured bucket, or its
own folders on a self-hosted stack, except the media a kept listing still
shows. A dry run only lists, an object gone since the listing is fine, and an
ID that is not UUID-like never becomes a prefix."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from backend.data.account_erasure import storage
from backend.data.account_erasure.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
WORKSPACE_ID = "a1b2c3d4-0000-4000-8000-000000000001"
APP_ID = "b1b2c3d4-0000-4000-8000-000000000002"
SUBJECT = Subject(user_id=USER_ID, workspace_id=WORKSPACE_ID, oauth_app_ids=[APP_ID])
# The real lookup; the autouse fixture below replaces it for the sweep tests.
KEPT_MEDIA = storage.kept_media


def _settings(private: str = "", public: str = "", legacy: str = "", local: str = ""):
    return SimpleNamespace(
        config=SimpleNamespace(
            resolved_private_user_data_bucket=private or legacy,
            resolved_public_site_media_bucket=public or legacy,
            media_gcs_bucket_name=legacy,
            workspace_storage_dir=local,
        )
    )


class FakeBucket:
    """A GCS client over one dict of bucket -> object names, paged by two."""

    def __init__(self, objects: dict[str, list[str]]):
        self.objects = objects
        self.deleted: list[tuple[str, str]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_objects(self, bucket: str, params: dict) -> dict:
        names = [
            n for n in self.objects.get(bucket, []) if n.startswith(params["prefix"])
        ]
        start = int(params.get("pageToken", 0))
        page = {"items": [{"name": n} for n in names[start : start + 2]]}
        if start + 2 < len(names):
            page["nextPageToken"] = str(start + 2)
        return page

    async def delete(self, bucket: str, name: str) -> None:
        self.deleted.append((bucket, name))


@pytest.fixture(autouse=True)
def kept(monkeypatch):
    """No media kept unless a test says so."""
    kept_media = AsyncMock(return_value=set())
    monkeypatch.setattr(storage, "kept_media", kept_media)
    return kept_media


@pytest.fixture
def gcs(monkeypatch):
    bucket = FakeBucket(
        {
            "private": [
                f"workspaces/{WORKSPACE_ID}/f1/a.txt",
                f"workspaces/{WORKSPACE_ID}/f2/b.txt",
                f"workspaces/cli-sessions/{USER_ID}/s1.jsonl",
                f"uploads/users/{USER_ID}/u1/voice.webm",
                f"users/{USER_ID}/images/1.png",
                "workspaces/someone-else/f/x.txt",
                "users/9d4e2b7a-1111-4c2a-9a51-0c6f2d8e1a77/images/2.png",
            ],
            "public": [
                f"users/{USER_ID}/images/1.png",
                f"oauth-apps/{APP_ID}/logo/l.png",
            ],
        }
    )
    monkeypatch.setattr(storage, "Settings", lambda: _settings("private", "public"))
    monkeypatch.setattr(storage.async_storage, "Storage", lambda: bucket)
    return bucket


@pytest.mark.asyncio
async def test_a_dry_run_lists_the_accounts_objects_and_deletes_nothing(gcs):
    report = await storage.delete_files(SUBJECT, dry_run=True)

    assert report.status == StepStatus.WOULD_DO
    assert report.counts == {"private_bucket_objects": 5, "public_bucket_objects": 2}
    assert gcs.deleted == []


@pytest.mark.asyncio
async def test_only_the_accounts_objects_are_deleted(gcs):
    await storage.delete_files(SUBJECT, dry_run=False)

    names = {name for _, name in gcs.deleted}
    assert len(gcs.deleted) == 7
    assert "workspaces/someone-else/f/x.txt" not in names
    assert not any("9d4e2b7a" in name for name in names)


@pytest.mark.asyncio
async def test_an_object_gone_since_the_listing_is_fine(gcs):
    async def delete(bucket, name):
        raise aiohttp.ClientResponseError(MagicMock(), (), status=404)

    gcs.delete = delete
    report = await storage.delete_files(SUBJECT, dry_run=False)

    assert report.status == StepStatus.DONE


@pytest.mark.asyncio
async def test_any_other_delete_error_fails_the_step(gcs):
    gcs.delete = AsyncMock(
        side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=503)
    )

    with pytest.raises(aiohttp.ClientResponseError):
        await storage.delete_files(SUBJECT, dry_run=False)


@pytest.mark.asyncio
async def test_a_legacy_single_bucket_is_swept_once(monkeypatch, gcs):
    gcs.objects = {"legacy": gcs.objects["private"]}
    monkeypatch.setattr(storage, "Settings", lambda: _settings(legacy="legacy"))

    report = await storage.delete_files(SUBJECT, dry_run=True)

    assert report.counts == {"private_bucket_objects": 5}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subject",
    [
        Subject(user_id=""),
        Subject(user_id="../../etc"),
        Subject(user_id=USER_ID, workspace_id="a/b"),
        Subject(user_id=USER_ID, oauth_app_ids=[""]),
    ],
)
async def test_an_unsafe_id_is_refused_before_any_listing(gcs, subject):
    with pytest.raises(ValueError):
        await storage.delete_files(subject, dry_run=True)


def test_transcripts_are_only_swept_for_an_id_they_are_stored_under():
    plain = storage._prefixes(Subject(user_id="user_with_underscores"))
    assert not any("cli-sessions" in prefix for prefix in plain)
    assert f"workspaces/cli-sessions/{USER_ID}/" in storage._prefixes(SUBJECT)


@pytest.mark.asyncio
async def test_a_self_hosted_stack_removes_the_accounts_folders(monkeypatch, tmp_path):
    workspaces = tmp_path / "workspaces"
    media = tmp_path / "store-media"
    mine = [
        workspaces / WORKSPACE_ID / "f1" / "a.txt",
        workspaces / "cli-sessions" / USER_ID / "s1.jsonl",
        media / "users" / USER_ID / "images" / "1.png",
    ]
    theirs = workspaces / "other-workspace" / "f" / "x.txt"
    for path in [*mine, theirs]:
        path.parent.mkdir(parents=True)
        path.write_text("x")
    monkeypatch.setattr(storage, "Settings", lambda: _settings(local=str(workspaces)))
    monkeypatch.setattr(storage, "media_root", lambda: media)

    dry = await storage.delete_files(SUBJECT, dry_run=True)
    assert dry.counts == {"local_files": 3}
    assert all(path.exists() for path in mine)

    await storage.delete_files(SUBJECT, dry_run=False)
    assert not any(path.exists() for path in mine)
    assert theirs.exists()

    again = await storage.delete_files(SUBJECT, dry_run=False)
    assert again.counts == {"local_files": 0}


@pytest.mark.asyncio
async def test_media_a_kept_listing_shows_is_not_deleted(gcs, kept):
    kept.return_value = {f"users/{USER_ID}/images/1.png"}

    dry = await storage.delete_files(SUBJECT, dry_run=True)
    await storage.delete_files(SUBJECT, dry_run=False)

    assert dry.counts == {
        "private_bucket_objects": 4,
        "public_bucket_objects": 1,
        "kept_media": 1,
    }
    assert f"users/{USER_ID}/images/1.png" not in {name for _, name in gcs.deleted}
    assert len(gcs.deleted) == 5


@pytest.mark.asyncio
async def test_a_self_hosted_stack_keeps_media_a_kept_listing_shows(
    monkeypatch, tmp_path, kept
):
    media = tmp_path / "store-media"
    shown = media / "users" / USER_ID / "images" / "shown.png"
    unused = media / "users" / USER_ID / "images" / "unused.png"
    for path in (shown, unused):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    kept.return_value = {f"users/{USER_ID}/images/shown.png"}
    monkeypatch.setattr(
        storage, "Settings", lambda: _settings(local=str(tmp_path / "workspaces"))
    )
    monkeypatch.setattr(storage, "media_root", lambda: media)

    report = await storage.delete_files(SUBJECT, dry_run=False)

    assert shown.exists() and not unused.exists()
    assert report.counts == {"local_files": 1, "kept_media": 1}


@pytest.mark.parametrize(
    "url",
    [
        f"https://storage.googleapis.com/public/users/{USER_ID}/images/a.png",
        f"/api/store/submissions/media/{USER_ID}/images/a.png",
        f"https://platform.example/api/store/media/{USER_ID}/images/a.png",
        "https://platform.example/_next/image?url=https%3A%2F%2Fstorage."
        f"googleapis.com%2Fpublic%2Fusers%2F{USER_ID}%2Fimages%2Fa.png&w=640",
    ],
)
def test_every_form_of_a_media_url_maps_to_its_object(url):
    assert storage.media_path(url, USER_ID) == f"users/{USER_ID}/images/a.png"


def test_another_accounts_media_url_maps_to_nothing():
    other = "https://storage.googleapis.com/public/users/someone/images/a.png"
    assert storage.media_path(other, USER_ID) is None


@pytest.mark.asyncio
async def test_kept_media_reads_listings_that_stay_and_other_libraries(monkeypatch):
    image = f"https://storage.googleapis.com/b/users/{USER_ID}/images/a.png"
    query = AsyncMock(return_value=[{"url": image}, {"url": "not media"}])
    monkeypatch.setattr(storage, "query_raw_with_schema", query)

    handed_over = SUBJECT.model_copy(update={"kept_listing_ids": ["l1"]})
    assert await KEPT_MEDIA(handed_over) == {f"users/{USER_ID}/images/a.png"}
    sql, *args = query.await_args.args
    assert 'l."id" = ANY($2::text[])' in sql
    assert args == [USER_ID, ["l1"]]

    # No empty array parameter when nothing was handed over.
    await KEPT_MEDIA(SUBJECT)
    sql, *args = query.await_args.args
    assert "$2" not in sql and "__HANDED_OVER__" not in sql
    assert args == [USER_ID]
