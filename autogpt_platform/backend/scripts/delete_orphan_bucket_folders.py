"""One-off: delete the bucket folders of accounts that no longer exist.

Accounts deleted before the erasure routine (`backend/data/account_erasure/`)
left their files behind. This finds every folder under an account's prefixes
(the ones `account_erasure/storage.py` sweeps) whose user, workspace or OAuth
app row is gone, in every configured bucket, and deletes it. The old
transcript folder, `workspaces/chat-transcripts/`, is left alone unless
``--include-chat-transcripts``. Only folders named by a UUID count. It keeps a
`users/{id}/` folder whose media a listing or library entry still shows, and
any folder with a file from the last few days, which may belong to a row
created after the check.

The default is a read-only dry run that prints what would go. ``--apply``
deletes. Output is counts only, never an ID or a file name.

Runs from a checkout on any backend image, inside the database manager pod
using the least memory (it has the database and the bucket settings); see
`docs/engineering/account-deletion.md`::

    kubectl top pod | grep database-manager
    kubectl exec -i <that pod> -- \\
        poetry run python -u - < scripts/delete_orphan_bucket_folders.py
    kubectl exec -i <that pod> -- \\
        poetry run python -u - --apply < scripts/delete_orphan_bucket_folders.py
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import aiohttp
from gcloud.aio import storage as gcs
from pydantic import BaseModel

from backend.data import db
from backend.util.settings import Settings

# (folder prefix, the table the folder name is an ID of)
PREFIXES = [
    ("workspaces/", "UserWorkspace"),
    ("workspaces/cli-sessions/", "User"),
    ("uploads/users/", "User"),
    ("users/", "User"),
    ("oauth-apps/", "OAuthApplication"),
]
# Otto's transcripts until April 2026 (#12804); nothing reads them now. Only
# with --include-chat-transcripts.
CHAT_TRANSCRIPTS = ("workspaces/chat-transcripts/", "User")
# Every ID these folders are named by is a UUID. Anything else (such as
# `workspaces/cli-sessions/`, a prefix of its own and not a workspace) is left
# alone.
ID = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")
RECENT = timedelta(days=7)
CHUNK = 5000
CONCURRENT_DELETES = 16
# A bucket where nearly every folder has no row is not this database's bucket
# (a preview's pod reads dev's bucket with a database of its own): no apply.
MAX_ORPHAN_SHARE = 0.9
# Media an account uploaded can be shown by listings, library entries and the
# avatars of orgs, org profiles, experts and profiles. A user's media in any
# form these store: a bucket URL, the media routes,
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
        UNION ALL
        SELECT "avatarUrl" FROM {schema_prefix}"Organization"
        UNION ALL
        SELECT "avatarUrl" FROM {schema_prefix}"OrganizationProfile"
        UNION ALL
        SELECT "avatarUrl" FROM {schema_prefix}"Expert"
        UNION ALL
        SELECT "avatarUrl" FROM {schema_prefix}"Profile"
    ) urls
    WHERE url ~ $1
"""


class Client(Protocol):
    """The two `gcloud.aio.storage.Storage` calls this uses."""

    async def list_objects(
        self, bucket: str, *, params: dict[str, str] | None = None
    ) -> dict[str, Any]: ...

    async def delete(self, bucket: str, object_name: str) -> Any: ...


class PrefixReport(BaseModel):
    folders: int = 0
    orphans: int = 0
    kept_shown: int = 0
    kept_recent: int = 0
    kept_back: int = 0
    # Orphans whose files were all gone by the last listing (removed elsewhere).
    emptied: int = 0
    deleted: int = 0
    files: int = 0
    size: int = 0
    # Files that failed to delete, by error type and status.
    failed: dict[str, int] = {}

    def line(self, prefix: str, table: str, apply: bool) -> str:
        verb = "deleted" if apply else "would delete"
        line = (
            f"{prefix}<{table} id>/: {self.folders} folders, {self.orphans} orphans;"
            f" {verb} {self.deleted} ({self.files} files, {self.size / 1e6:.1f} MB)"
        )
        kept = [
            f"{self.kept_shown} shown by a listing, library entry or avatar",
            f"{self.kept_recent} with a file from the last {RECENT.days} days",
            f"{self.kept_back} whose row exists again",
        ]
        line += "; kept: " + ", ".join(kept)
        if self.emptied:
            line += f"; {self.emptied} emptied by something else before the delete"
        if self.failed:
            line += (
                f"; {sum(self.failed.values())} files failed to delete"
                f" ({dict(self.failed)}); their folders stay, maybe partly"
                " deleted, so run it again"
            )
        return line


class Folder(BaseModel):
    name: str
    objects: list[dict] = []

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


def error_kind(error: Exception, status: int | None = None) -> str:
    """An error's type and HTTP status, never its message: a storage error's
    message carries the request URL, which names the folder."""
    return f"{type(error).__name__} {status}" if status else type(error).__name__


async def delete_objects(
    client: Client, bucket: str, names: list[str]
) -> dict[str, int]:
    """Delete the objects; count the failures by kind. One gone already is fine."""
    failed: dict[str, int] = {}

    def count(kind: str) -> None:
        failed[kind] = failed.get(kind, 0) + 1

    async def delete(name: str) -> None:
        try:
            await client.delete(bucket, name)
        except aiohttp.ClientResponseError as e:
            if e.status != 404:
                count(error_kind(e, e.status))
        except Exception as e:
            count(error_kind(e))

    for i in range(0, len(names), CONCURRENT_DELETES):
        await asyncio.gather(*(delete(n) for n in names[i : i + CONCURRENT_DELETES]))
    return failed


async def sweep_prefix(
    client: Client,
    bucket: str,
    prefix: str,
    table: str,
    *,
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
    has_media = prefix == "users/"
    shown = await shown_media_owners() if has_media and orphans else set()

    doomed: list[Folder] = []
    for name in orphans:
        if name in shown:
            report.kept_shown += 1
            continue
        objects = await list_objects(client, bucket, f"{prefix}{name}/")
        folder = Folder(name=name, objects=objects)
        if folder.is_recent(now):
            report.kept_recent += 1
            continue
        doomed.append(folder)
    if not doomed:
        return report

    # Just before deleting, check again: a row created since the first check
    # takes its folder back, and so does a listing that now shows its media or
    # a file written since the first listing.
    back = await existing(table, [f.name for f in doomed])
    shown = await shown_media_owners() if has_media else set()
    for folder in doomed:
        if folder.name in back:
            report.kept_back += 1
            continue
        if folder.name in shown:
            report.kept_shown += 1
            continue
        objects = await list_objects(client, bucket, f"{prefix}{folder.name}/")
        folder = Folder(name=folder.name, objects=objects)
        if folder.is_recent(now):
            report.kept_recent += 1
            continue
        if not folder.objects:
            report.emptied += 1
            continue
        names = [o["name"] for o in folder.objects]
        failed = await delete_objects(client, bucket, names) if apply else {}
        if failed:
            for kind, count in failed.items():
                report.failed[kind] = report.failed.get(kind, 0) + count
            continue
        report.deleted += 1
        report.files += len(folder.objects)
        report.size += folder.size
    return report


async def orphan_share(
    client: Client, bucket: str, prefixes: list[tuple[str, str]]
) -> float:
    """The share of the bucket's account folders that have no row here."""
    folders = orphans = 0
    for prefix, table in prefixes:
        names = sorted(await list_folders(client, bucket, prefix))
        if names:
            folders += len(names)
            orphans += len(set(names) - await existing(table, names))
    return orphans / folders if folders else 0.0


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


async def run(apply: bool, include_chat_transcripts: bool = False) -> int:
    configured = buckets()
    if not configured:
        print("No bucket is configured on this pod; nothing to do.")
        return 1
    print(
        (
            "APPLY: deleting orphan folders."
            if apply
            else "DRY RUN: nothing is deleted. Run again with --apply to delete."
        ),
        flush=True,
    )
    prefixes = PREFIXES + ([CHAT_TRANSCRIPTS] if include_chat_transcripts else [])
    now = datetime.now(timezone.utc)
    failed = errors = 0
    await db.connect()
    try:
        async with gcs.Storage() as client:
            for label, bucket in configured.items():
                print(f"\n== {label} bucket {bucket}", flush=True)
                try:
                    share = await orphan_share(client, bucket, prefixes)
                except aiohttp.ClientResponseError as e:
                    print(f"stopped by {error_kind(e, e.status)}", flush=True)
                    errors += 1
                    continue
                except Exception as e:
                    print(f"stopped by {error_kind(e)}", flush=True)
                    errors += 1
                    continue
                if share > MAX_ORPHAN_SHARE:
                    print(
                        f"{share:.0%} of this bucket's folders have no row in this"
                        " database, so it is not this database's bucket (a preview?)."
                        " Nothing is deleted here.",
                        flush=True,
                    )
                    errors += 1
                    continue
                for prefix, table in prefixes:
                    try:
                        report = await sweep_prefix(
                            client, bucket, prefix, table, apply=apply, now=now
                        )
                    except aiohttp.ClientResponseError as e:
                        # Never the traceback: its request URL names a folder.
                        kind = error_kind(e, e.status)
                        print(f"{prefix}: stopped by {kind}", flush=True)
                        errors += 1
                        continue
                    except Exception as e:
                        print(f"{prefix}: stopped by {error_kind(e)}", flush=True)
                        errors += 1
                        continue
                    print(report.line(prefix, table, apply), flush=True)
                    failed += sum(report.failed.values())
                if not include_chat_transcripts:
                    print(
                        f"{CHAT_TRANSCRIPTS[0]}: skipped; add"
                        " --include-chat-transcripts to include it",
                        flush=True,
                    )
    finally:
        await db.disconnect()
    if failed or errors:
        print(f"\n{failed} files failed and {errors} steps stopped; see above.")
    return 1 if failed or errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete the bucket folders of accounts that no longer exist."
    )
    parser.add_argument(
        "--apply", action="store_true", help="delete (default: dry run)"
    )
    parser.add_argument(
        "--include-chat-transcripts",
        action="store_true",
        help=f"also sweep {CHAT_TRANSCRIPTS[0]} (default: left alone)",
    )
    args = parser.parse_args(argv)
    return asyncio.run(run(args.apply, args.include_chat_transcripts))


if __name__ == "__main__":
    sys.exit(main())
