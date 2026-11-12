from datetime import datetime, timedelta, timezone

import aiohttp

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
            raise aiohttp.ClientResponseError(None, (), status=503)  # type: ignore[arg-type]
        if object_name not in self.objects:
            raise aiohttp.ClientResponseError(None, (), status=404)  # type: ignore[arg-type]
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


async def _sweep(bucket, prefix, table, *, apply, shown: set[str] | None = None):
    return await tool.sweep_prefix(
        bucket, BUCKET, prefix, table, shown=shown or set(), apply=apply, now=NOW
    )


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
            "users/not-an-id/x.png": (1, OLD),
        }
    )
    monkeypatch.setattr(
        tool, "existing", _rows(LIVE, appear_on_recheck=frozenset({BACK}))
    )

    report = await _sweep(bucket, "users/", "User", apply=True, shown={SHOWN})

    assert sorted(bucket.deleted) == [
        f"users/{GONE}/images/a.png",
        f"users/{GONE}/videos/b.mp4",
    ]
    assert (report.orphans, report.deleted, report.files) == (4, 1, 2)
    assert (report.kept_shown, report.kept_recent, report.kept_back) == (1, 1, 1)
    assert report.failed == 0


async def test_a_failed_delete_is_counted_and_a_missing_object_is_not(monkeypatch):
    bucket = FakeBucket(
        {
            f"uploads/users/{GONE}/1/a.bin": (1, OLD),
            f"uploads/users/{GONE}/2/b.bin": (1, OLD),
        }
    )
    bucket.fail.add(f"uploads/users/{GONE}/1/a.bin")
    monkeypatch.setattr(tool, "existing", _rows())

    report = await _sweep(bucket, "uploads/users/", "User", apply=True)

    assert report.failed == 1
    assert await tool.delete_objects(bucket, BUCKET, ["already-gone"]) == 0


async def test_report_line_names_no_folder(monkeypatch):
    bucket = FakeBucket({f"oauth-apps/{GONE}/logo.png": (2_500_000, OLD)})
    monkeypatch.setattr(tool, "existing", _rows())

    report = await _sweep(bucket, "oauth-apps/", "OAuthApplication", apply=False)
    line = report.line("oauth-apps/", "OAuthApplication", apply=False)

    assert GONE not in line and "logo" not in line
    assert "1 orphans; would delete 1 (1 files, 2.5 MB)" in line


def test_media_owner_matches_every_stored_url_form():
    import re

    owner = re.compile(tool.MEDIA_OWNER)
    for url in [
        f"https://storage.googleapis.com/b/users/{SHOWN}/images/a.png",
        f"/api/store/media/{SHOWN}/videos/b.mp4",
        f"/_next/image?url=%2Fapi%2Fstore%2Fmedia%2F{SHOWN}%2Fimages%2Fc.png&w=64",
    ]:
        match = owner.search(url)
        assert match and match[1] == SHOWN, url
