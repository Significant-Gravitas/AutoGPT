"""Files go by the account's own prefixes in every configured bucket, or its
own folders on a self-hosted stack. A dry run only lists, an object gone
since the listing is fine, and an ID that is not UUID-like never becomes a
prefix."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from backend.data.account_deletion import storage
from backend.data.account_deletion.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
WORKSPACE_ID = "a1b2c3d4-0000-4000-8000-000000000001"
APP_ID = "b1b2c3d4-0000-4000-8000-000000000002"
SUBJECT = Subject(user_id=USER_ID, workspace_id=WORKSPACE_ID, oauth_app_ids=[APP_ID])


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
