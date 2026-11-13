import re
from datetime import datetime, timedelta, timezone

import aiohttp
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

import scripts.delete_orphan_bucket_folders as tool

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)
OLD = (NOW - timedelta(days=90)).isoformat().replace("+00:00", "Z")
FRESH = (NOW - timedelta(days=1)).isoformat().replace("+00:00", "Z")
BUCKET = "private-bucket"
LIVE = "11111111-aaaa-4aaa-8aaa-111111111111"
GONE = "22222222-bbbb-4bbb-8bbb-222222222222"
SHOWN = "33333333-cccc-4ccc-8ccc-333333333333"
RECENT = "44444444-dddd-4ddd-8ddd-444444444444"
BACK = "55555555-eeee-4eee-8eee-555555555555"
LATE_SHOWN = "66666666-ffff-4fff-8fff-666666666666"
OTHER = "77777777-aaaa-4aaa-8aaa-777777777777"


def _gcs_error(status: int, name: str) -> aiohttp.ClientResponseError:
    url = URL(f"https://storage.test/storage/v1/b/{BUCKET}/o/{name}")
    info = aiohttp.RequestInfo(url, "DELETE", CIMultiDictProxy(CIMultiDict()), url)
    return aiohttp.ClientResponseError(info, (), status=status, message=str(url))


class FakeBucket:
    def __init__(self, objects: dict[str, tuple[int, str]]):
        self.objects = dict(objects)
        self.deleted: list[str] = []
        self.fail: set[str] = set()

    async def list_objects(
        self, bucket: str, *, params: dict[str, str] | None = None
    ) -> dict:
        assert bucket == BUCKET and params
        prefix = params["prefix"]
        names = sorted(n for n in self.objects if n.startswith(prefix))
        if params.get("delimiter") == "/":
            folders = {prefix + n[len(prefix) :].split("/")[0] + "/" for n in names}
            return {"prefixes": sorted(f for f in folders if f.count("/") > 0)}
        return {
            "items": [
                {
                    "name": n,
                    "size": str(self.objects[n][0]),
                    "updated": self.objects[n][1],
                }
                for n in names
            ]
        }

    async def delete(self, bucket: str, object_name: str) -> None:
        if object_name in self.fail:
            raise _gcs_error(503, object_name)
        if object_name not in self.objects:
            raise _gcs_error(404, object_name)
        del self.objects[object_name]
        self.deleted.append(object_name)


def _rows(*live: str, appear_on_recheck: frozenset[str] = frozenset()):
    calls = 0

    async def existing(table: str, ids: list[str]) -> set[str]:
        nonlocal calls
        calls += 1
        found = set(live) | (set(appear_on_recheck) if calls > 1 else set())
        return found & set(ids)

    return existing


def _shown(first: set[str], on_recheck: set[str] | None = None):
    calls = 0

    async def shown_media_owners() -> set[str]:
        nonlocal calls
        calls += 1
        return first if calls == 1 or on_recheck is None else first | on_recheck

    return shown_media_owners


async def _sweep(bucket, prefix, table, *, apply):
    return await tool.sweep_prefix(bucket, BUCKET, prefix, table, apply=apply, now=NOW)


async def test_dry_run_counts_orphans_and_deletes_nothing_nor_other_prefixes(
    monkeypatch,
):
    bucket = FakeBucket(
        {
            f"workspaces/{LIVE}/f/a.txt": (10, OLD),
            f"workspaces/{GONE}/f/a.txt": (1_000_000, OLD),
            f"workspaces/{GONE}/g/b.txt": (500_000, OLD),
            f"workspaces/cli-sessions/{GONE}/s.jsonl": (5, OLD),
            f"workspaces/chat-transcripts/{LIVE}/s.jsonl": (5, OLD),
        }
    )
    monkeypatch.setattr(tool, "existing", _rows(LIVE))

    report = await _sweep(bucket, "workspaces/", "UserWorkspace", apply=False)

    assert (report.folders, report.orphans, report.deleted) == (2, 1, 1)
    assert (report.files, report.size) == (2, 1_500_000)
    assert bucket.deleted == []


async def test_apply_deletes_only_orphans_nobody_still_needs(monkeypatch):
    bucket = FakeBucket(
        {
            f"users/{LIVE}/images/a.png": (1, OLD),
            f"users/{GONE}/images/a.png": (1, OLD),
            f"users/{GONE}/videos/b.mp4": (1, OLD),
            f"users/{SHOWN}/images/c.png": (1, OLD),
            f"users/{RECENT}/images/old.png": (1, OLD),
            f"users/{RECENT}/images/new.png": (1, FRESH),
            f"users/{BACK}/images/d.png": (1, OLD),
            f"users/{LATE_SHOWN}/images/e.png": (1, OLD),
            "users/not-an-id/x.png": (1, OLD),
        }
    )
    monkeypatch.setattr(
        tool, "existing", _rows(LIVE, appear_on_recheck=frozenset({BACK}))
    )
    monkeypatch.setattr(tool, "shown_media_owners", _shown({SHOWN}, {LATE_SHOWN}))

    report = await _sweep(bucket, "users/", "User", apply=True)

    assert sorted(bucket.deleted) == [
        f"users/{GONE}/images/a.png",
        f"users/{GONE}/videos/b.mp4",
    ]
    assert (report.orphans, report.deleted, report.files) == (5, 1, 2)
    assert (report.kept_shown, report.kept_recent, report.kept_back) == (2, 1, 1)
    assert not report.failed


async def test_a_folder_with_a_failed_delete_is_not_counted_as_deleted(monkeypatch):
    bucket = FakeBucket(
        {
            f"uploads/users/{GONE}/1/a.bin": (1, OLD),
            f"uploads/users/{GONE}/2/b.bin": (1, OLD),
            f"uploads/users/{OTHER}/1/c.bin": (1, OLD),
        }
    )
    bucket.fail.add(f"uploads/users/{GONE}/1/a.bin")
    monkeypatch.setattr(tool, "existing", _rows())

    report = await _sweep(bucket, "uploads/users/", "User", apply=True)

    assert report.failed == {"ClientResponseError 503": 1}
    assert (report.deleted, report.files) == (1, 1)
    line = report.line("uploads/users/", "User", True)
    assert "1 files failed to delete" in line and GONE not in line
    assert "maybe partly deleted" in line
    assert not await tool.delete_objects(bucket, BUCKET, ["already-gone"])


async def test_a_file_written_after_the_first_listing_keeps_its_folder(monkeypatch):
    bucket = FakeBucket({f"workspaces/{GONE}/f/a.txt": (1, OLD)})
    calls = 0

    async def existing(table: str, ids: list[str]) -> set[str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            bucket.objects[f"workspaces/{GONE}/f/new.txt"] = (1, FRESH)
        return set()

    monkeypatch.setattr(tool, "existing", existing)

    report = await _sweep(bucket, "workspaces/", "UserWorkspace", apply=True)

    assert (report.deleted, report.kept_recent) == (0, 1)
    assert bucket.deleted == []


async def test_a_folder_emptied_elsewhere_is_not_counted_as_deleted(monkeypatch):
    bucket = FakeBucket(
        {
            f"workspaces/{GONE}/f/a.txt": (1, OLD),
            f"workspaces/{OTHER}/f/b.txt": (1, OLD),
        }
    )
    calls = 0

    async def existing(table: str, ids: list[str]) -> set[str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            del bucket.objects[f"workspaces/{GONE}/f/a.txt"]
        return set()

    monkeypatch.setattr(tool, "existing", existing)

    report = await _sweep(bucket, "workspaces/", "UserWorkspace", apply=True)

    assert (report.orphans, report.deleted, report.emptied) == (2, 1, 1)
    assert bucket.deleted == [f"workspaces/{OTHER}/f/b.txt"]
    line = report.line("workspaces/", "UserWorkspace", True)
    assert "deleted 1 (1 files" in line
    assert "1 emptied by something else before the delete" in line


async def test_report_line_names_no_folder(monkeypatch):
    bucket = FakeBucket({f"oauth-apps/{GONE}/logo.png": (2_500_000, OLD)})
    monkeypatch.setattr(tool, "existing", _rows())

    report = await _sweep(bucket, "oauth-apps/", "OAuthApplication", apply=False)
    line = report.line("oauth-apps/", "OAuthApplication", apply=False)

    assert GONE not in line and "logo" not in line
    assert "1 orphans; would delete 1 (1 files, 2.5 MB)" in line


def test_media_owner_matches_every_stored_url_form():
    owner = re.compile(tool.MEDIA_OWNER)
    for url in [
        f"https://storage.googleapis.com/b/users/{SHOWN}/images/a.png",
        f"/api/store/media/{SHOWN}/videos/b.mp4",
        f"/_next/image?url=%2Fapi%2Fstore%2Fmedia%2F{SHOWN}%2Fimages%2Fc.png&w=64",
    ]:
        match = owner.search(url)
        assert match and match[1] == SHOWN, url


async def test_a_failing_prefix_prints_no_folder_and_fails_the_run(monkeypatch, capsys):
    class Storage:
        async def __aenter__(self):
            return FakeBucket({})

        async def __aexit__(self, *exc):
            return None

    async def sweep(client, bucket, prefix, table, **kwargs):
        if prefix == "users/":
            raise _gcs_error(403, f"users/{GONE}/")
        return tool.PrefixReport()

    async def nothing():
        return None

    monkeypatch.setattr(tool, "buckets", lambda: {"private": BUCKET})
    monkeypatch.setattr(tool.db, "connect", nothing)
    monkeypatch.setattr(tool.db, "disconnect", nothing)
    monkeypatch.setattr(tool.gcs, "Storage", Storage)
    monkeypatch.setattr(tool, "sweep_prefix", sweep)

    assert await tool.run(apply=True) == 1
    out = capsys.readouterr().out
    assert "users/: stopped by ClientResponseError 403" in out
    assert GONE not in out
    assert "oauth-apps/<OAuthApplication id>/: 0 folders" in out


async def test_apply_refuses_a_bucket_this_database_does_not_own(monkeypatch, capsys):
    bucket = FakeBucket(
        {f"workspaces/{GONE}/f/a.txt": (1, OLD), f"users/{OTHER}/a.png": (1, OLD)}
    )

    class Storage:
        async def __aenter__(self):
            return bucket

        async def __aexit__(self, *exc):
            return None

    async def nothing():
        return None

    monkeypatch.setattr(tool, "buckets", lambda: {"private": BUCKET})
    monkeypatch.setattr(tool.db, "connect", nothing)
    monkeypatch.setattr(tool.db, "disconnect", nothing)
    monkeypatch.setattr(tool.gcs, "Storage", Storage)
    monkeypatch.setattr(tool, "existing", _rows())

    assert await tool.run(apply=True) == 1
    assert "100% of this bucket's folders have no row" in capsys.readouterr().out
    assert bucket.deleted == []


async def test_chat_transcripts_are_left_alone_unless_asked(monkeypatch, capsys):
    transcript = f"workspaces/chat-transcripts/{GONE}/s.jsonl"
    bucket = FakeBucket(
        {
            f"workspaces/{LIVE}/f/a.txt": (1, OLD),
            f"users/{LIVE}/images/a.png": (1, OLD),
            transcript: (1, OLD),
        }
    )

    class Storage:
        async def __aenter__(self):
            return bucket

        async def __aexit__(self, *exc):
            return None

    async def nothing():
        return None

    monkeypatch.setattr(tool, "buckets", lambda: {"private": BUCKET})
    monkeypatch.setattr(tool.db, "connect", nothing)
    monkeypatch.setattr(tool.db, "disconnect", nothing)
    monkeypatch.setattr(tool.gcs, "Storage", Storage)
    monkeypatch.setattr(tool, "existing", _rows(LIVE))

    assert await tool.run(apply=True) == 0
    out = capsys.readouterr().out
    assert "workspaces/chat-transcripts/: skipped" in out
    assert "--include-chat-transcripts" in out
    assert bucket.deleted == []

    assert await tool.run(apply=True, include_chat_transcripts=True) == 0
    out = capsys.readouterr().out
    assert "workspaces/chat-transcripts/<User id>/: 1 folders, 1 orphans" in out
    assert "skipped" not in out
    assert bucket.deleted == [transcript]


async def test_the_ownership_check_covers_chat_transcripts_when_included(
    monkeypatch, capsys
):
    bucket = FakeBucket({f"workspaces/chat-transcripts/{GONE}/s.jsonl": (1, OLD)})

    class Storage:
        async def __aenter__(self):
            return bucket

        async def __aexit__(self, *exc):
            return None

    async def nothing():
        return None

    monkeypatch.setattr(tool, "buckets", lambda: {"private": BUCKET})
    monkeypatch.setattr(tool.db, "connect", nothing)
    monkeypatch.setattr(tool.db, "disconnect", nothing)
    monkeypatch.setattr(tool.gcs, "Storage", Storage)
    monkeypatch.setattr(tool, "existing", _rows())

    assert await tool.run(apply=True, include_chat_transcripts=True) == 1
    assert "100% of this bucket's folders have no row" in capsys.readouterr().out
    assert bucket.deleted == []
