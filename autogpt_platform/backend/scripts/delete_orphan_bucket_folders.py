"""One-off: delete the bucket folders of accounts that no longer exist.

Accounts deleted before the erasure routine (`backend/data/account_erasure/`)
left their files behind. This finds every folder under an account's prefixes
(the ones `account_erasure/storage.py` sweeps, and the old transcript folder)
whose user, workspace or OAuth app row is gone, in every configured bucket,
and deletes it. Only folders named by a UUID count. It keeps a
`users/{id}/` folder whose media a listing or library entry still shows, and
any folder with a file from the last few days, which may belong to a row
created after the check.

The default is a read-only dry run that prints what would go. ``--apply``
deletes. Output is counts only, never an ID or a file name.

Runs from a checkout on any backend image, inside a database manager pod
(it has the database and the bucket settings); see
`docs/engineering/account-deletion.md`::

    kubectl exec -i deploy/autogpt-database-manager -- \\
        poetry run python - < scripts/delete_orphan_bucket_folders.py
    kubectl exec -i deploy/autogpt-database-manager -- \\
        poetry run python - --apply < scripts/delete_orphan_bucket_folders.py
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from gcloud.aio import storage as gcs

from backend.data import db
from backend.util.gcs_utils import is_not_found_error
from backend.util.settings import Settings

# (folder prefix, the table the folder name is an ID of)
PREFIXES = [
    ("workspaces/", "UserWorkspace"),
    ("workspaces/cli-sessions/", "User"),
    # Otto's transcripts until April 2026 (#12804); nothing reads them now.
    ("workspaces/chat-transcripts/", "User"),
    ("uploads/users/", "User"),
    ("users/", "User"),
    ("oauth-apps/", "OAuthApplication"),
]
# Every ID these folders are named by is a UUID. Anything else (such as
# `workspaces/cli-sessions/`, a prefix of its own and not a workspace) is left
# alone.
ID = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")
RECENT = timedelta(days=7)
CHUNK = 5000
CONCURRENT_DELETES = 16
# A user's media in any form a listing stores: a bucket URL, the media routes,
# or either URL-encoded inside Next.js's image optimizer.
_SEP = "(?:/|%2[fF])"
MEDIA_OWNER = (
    rf"(?:users|media){_SEP}([0-9A-Za-z-]{{8,64}}){_SEP}(?:images|videos){_SEP}"
)
SHOWN_MEDIA_OWNERS = """
    SELECT DISTINCT (regexp_match(url, $1))[1] AS owner FROM (
        SELECT unnest(v."imageUrls" || ARRAY[v."videoUrl", v."agentOutputDemoUrl"])
               AS url
          FROM {schema_prefix}"StoreListingVersion" v
        UNION ALL
        SELECT la."imageUrl" FROM {schema_prefix}"LibraryAgent" la
    ) urls
    WHERE url ~ $1
"""


class Client(Protocol):
    """The two `gcloud.aio.storage.Storage` calls this uses."""

    async def list_objects(
        self, bucket: str, *, params: dict[str, str] | None = None
    ) -> dict[str, Any]: ...

    async def delete(self, bucket: str, object_name: str) -> Any: ...


@dataclass
class PrefixReport:
    folders: int = 0
    orphans: int = 0
    kept_shown: int = 0
    kept_recent: int = 0
    kept_back: int = 0
    deleted: int = 0
    files: int = 0
    size: int = 0
    failed: int = 0

    def line(self, prefix: str, table: str, apply: bool) -> str:
        verb = "deleted" if apply else "would delete"
        line = (
            f"{prefix}<{table} id>/: {self.folders} folders, {self.orphans} orphans;"
            f" {verb} {self.deleted} ({self.files} files, {self.size / 1e6:.1f} MB)"
        )
        kept = [
            f"{self.kept_shown} shown by a listing or library entry",
            f"{self.kept_recent} with a file from the last {RECENT.days} days",
            f"{self.kept_back} whose row exists again",
        ]
        line += "; kept: " + ", ".join(kept)
        if self.failed:
            line += f"; {self.failed} files failed to delete"
        return line


@dataclass
class Folder:
    name: str
    objects: list[dict] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(int(o.get("size", 0)) for o in self.objects)

    def is_recent(self, now: datetime) -> bool:
        return any(_parse(o.get("updated", "")) > now - RECENT for o in self.objects)


def _parse(stamp: str) -> datetime:
    if not stamp:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


async def list_folders(client: Client, bucket: str, prefix: str) -> set[str]:
    names: set[str] = set()
    token = None
    while True:
        params = {
            "prefix": prefix,
            "delimiter": "/",
            "fields": "prefixes,nextPageToken",
        }
        if token:
            params["pageToken"] = token
        page = await client.list_objects(bucket, params=params)
        names |= {p[len(prefix) :].rstrip("/") for p in page.get("prefixes", [])}
        token = page.get("nextPageToken")
        if not token:
            return {n for n in names if ID.fullmatch(n)}


async def list_objects(client: Client, bucket: str, prefix: str) -> list[dict]:
    objects: list[dict] = []
    token = None
    while True:
        params = {"prefix": prefix, "fields": "items(name,size,updated),nextPageToken"}
        if token:
            params["pageToken"] = token
        page = await client.list_objects(bucket, params=params)
        objects += [o for o in page.get("items", []) if o.get("name")]
        token = page.get("nextPageToken")
        if not token:
            return objects


async def existing(table: str, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for i in range(0, len(ids), CHUNK):
        rows = await db.query_raw_with_schema(
            f'SELECT "id" FROM {{schema_prefix}}"{table}" WHERE "id" = ANY($1::text[])',
            ids[i : i + CHUNK],
        )
        found |= {r["id"] for r in rows}
    return found


async def shown_media_owners() -> set[str]:
    rows = await db.query_raw_with_schema(SHOWN_MEDIA_OWNERS, MEDIA_OWNER)
    return {r["owner"] for r in rows if r["owner"]}


async def delete_objects(client: Client, bucket: str, names: list[str]) -> int:
    """Delete the objects; return how many failed. One gone already is fine."""
    sem = asyncio.Semaphore(CONCURRENT_DELETES)
    failed = 0

    async def delete(name: str) -> None:
        nonlocal failed
        async with sem:
            try:
                await client.delete(bucket, name)
            except Exception as e:
                if not is_not_found_error(e):
                    failed += 1

    await asyncio.gather(*(delete(n) for n in names))
    return failed


async def sweep_prefix(
    client: Client,
    bucket: str,
    prefix: str,
    table: str,
    *,
    shown: set[str],
    apply: bool,
    now: datetime,
) -> PrefixReport:
    report = PrefixReport()
    folders = sorted(await list_folders(client, bucket, prefix))
    report.folders = len(folders)
    if not folders:
        return report
    orphans = sorted(set(folders) - await existing(table, folders))
    report.orphans = len(orphans)

    doomed: list[Folder] = []
    for name in orphans:
        if prefix == "users/" and name in shown:
            report.kept_shown += 1
            continue
        folder = Folder(name, await list_objects(client, bucket, f"{prefix}{name}/"))
        if folder.is_recent(now):
            report.kept_recent += 1
            continue
        doomed.append(folder)

    # A row created since the first check takes its folder back.
    back = await existing(table, [f.name for f in doomed]) if doomed else set()
    report.kept_back = len(back)
    doomed = [f for f in doomed if f.name not in back]

    report.deleted = len(doomed)
    report.files = sum(len(f.objects) for f in doomed)
    report.size = sum(f.size for f in doomed)
    if apply and doomed:
        names = [o["name"] for f in doomed for o in f.objects]
        report.failed = await delete_objects(client, bucket, names)
    return report


def buckets() -> dict[str, str]:
    config = Settings().config
    out: dict[str, str] = {}
    for label, name in [
        ("private", config.resolved_private_user_data_bucket),
        ("public", config.resolved_public_site_media_bucket),
        ("legacy", config.media_gcs_bucket_name),
    ]:
        if name and name not in out.values():
            out[label] = name
    return out


async def run(apply: bool) -> int:
    configured = buckets()
    if not configured:
        print("No bucket is configured on this pod; nothing to do.")
        return 1
    print(
        "APPLY: deleting orphan folders."
        if apply
        else "DRY RUN: nothing is deleted. Run again with --apply to delete."
    )
    now = datetime.now(timezone.utc)
    failed = 0
    await db.connect()
    try:
        shown = await shown_media_owners()
        async with gcs.Storage() as client:
            for label, bucket in configured.items():
                print(f"\n== {label} bucket {bucket}")
                for prefix, table in PREFIXES:
                    report = await sweep_prefix(
                        client, bucket, prefix, table, shown=shown, apply=apply, now=now
                    )
                    if report.folders:
                        print(report.line(prefix, table, apply))
                    failed += report.failed
    finally:
        await db.disconnect()
    if failed:
        print(f"\n{failed} files failed to delete; run --apply again.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete the bucket folders of accounts that no longer exist."
    )
    parser.add_argument(
        "--apply", action="store_true", help="delete (default: dry run)"
    )
    args = parser.parse_args(argv)
    return asyncio.run(run(args.apply))


if __name__ == "__main__":
    sys.exit(main())
