"""The account's files: object storage on a cloud stack, the data directory on
a self-hosted one.

Every prefix below holds exactly one account's objects, keyed by its user ID,
its `UserWorkspace` ID (one per user, never shared) or an OAuth app it owns.
Marketplace media an account uploaded lives under its own `users/{id}/`, so
the media a listing that stays still shows is kept: listings handed to the
"deleted user" account (`shared_agents.py`), org listings it uploaded for,
and the images other accounts' library entries copied from them.
"""

import asyncio
import logging
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

from gcloud.aio import storage as async_storage

from backend.api.features.store.local_media import media_root
from backend.data.account_erasure.models import SAFE_ID, StepReport, StepStatus, Subject
from backend.data.db import query_raw_with_schema
from backend.util.data import get_data_path
from backend.util.gcs_utils import is_not_found_error
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

# `copilot/transcript.py` strips everything but hex and hyphens from the user
# ID in transcript paths, so only such an ID maps onto its own folder there.
_TRANSCRIPT_ID = re.compile(r"[0-9a-fA-F-]{8,36}")
_CONCURRENT_DELETES = 16
# URLs of the account's media that a listing staying in the marketplace, or
# another account's library entry, still shows. $1 = the account (an ID
# `SAFE_ID` allows, so no LIKE wildcard); `_HANDED_OVER` adds $2, the
# listings handed over, when there are any.
_KEPT_MEDIA_URLS = """
    SELECT DISTINCT url FROM (
        SELECT unnest(v."imageUrls" || ARRAY[v."videoUrl", v."agentOutputDemoUrl"])
               AS url
          FROM {schema_prefix}"StoreListingVersion" v
          JOIN {schema_prefix}"StoreListing" l ON l."id" = v."storeListingId"
         WHERE l."owningUserId" <> $1 __HANDED_OVER__
        UNION ALL
        SELECT la."imageUrl" FROM {schema_prefix}"LibraryAgent" la
         WHERE la."userId" <> $1
    ) urls
    WHERE url LIKE '%' || $1 || '%'
"""
_HANDED_OVER = 'OR l."id" = ANY($2::text[])'


async def delete_files(subject: Subject, *, dry_run: bool) -> StepReport:
    _require_safe_ids(subject)
    kept = await kept_media(subject)
    buckets = _buckets()
    if buckets:
        counts = {
            bucket_label: await _sweep_bucket(bucket, _prefixes(subject), kept, dry_run)
            for bucket_label, bucket in buckets.items()
        }
    else:
        kept_files = {media_root() / path for path in kept}
        counts = {
            "local_files": _sweep_local(_local_dirs(subject), kept_files, dry_run)
        }
    if kept:
        counts["kept_media"] = len(kept)
    return StepReport(
        name="files",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts=counts,
    )


def _require_safe_ids(subject: Subject) -> None:
    ids = [subject.user_id, *subject.oauth_app_ids]
    if subject.workspace_id:
        ids.append(subject.workspace_id)
    unsafe = [i for i in ids if not SAFE_ID.fullmatch(i)]
    if unsafe:
        raise ValueError(f"{len(unsafe)} IDs are not safe as storage prefixes")


async def kept_media(subject: Subject) -> set[str]:
    """Object paths (`users/{id}/{images|videos}/{file}`) of the account's
    media that something staying still shows."""
    handed_over = subject.kept_listing_ids
    rows = await query_raw_with_schema(
        _KEPT_MEDIA_URLS.replace(
            "__HANDED_OVER__", _HANDED_OVER if handed_over else ""
        ),
        subject.user_id,
        *([handed_over] if handed_over else []),
    )
    paths = (media_path(row["url"], subject.user_id) for row in rows)
    return {path for path in paths if path}


def media_path(url: str, user_id: str) -> str | None:
    """The object a media URL of this account names, in any form a listing
    stores: a bucket URL, the private media route, the self-hosted media
    route, or any of them wrapped by Next.js's image optimizer."""
    match = re.search(
        rf"(?:^|/)(?:users|api/store/(?:submissions/)?media)/{re.escape(user_id)}"
        r"/(images|videos)/([^/?#&]+)",
        unquote(url),
    )
    if not match:
        return None
    return f"users/{user_id}/{match[1]}/{match[2]}"


def _buckets() -> dict[str, str]:
    """Every configured bucket, by a label for the report. In a split setup
    the legacy bucket is the private one; the public one gets copies."""
    config = Settings().config
    labelled = {
        "private_bucket_objects": config.resolved_private_user_data_bucket,
        "public_bucket_objects": config.resolved_public_site_media_bucket,
        "legacy_bucket_objects": config.media_gcs_bucket_name,
    }
    unique: dict[str, str] = {}
    for label, bucket in labelled.items():
        if bucket and bucket not in unique.values():
            unique[label] = bucket
    return unique


def _prefixes(subject: Subject) -> list[str]:
    uid = subject.user_id
    prefixes = [f"uploads/users/{uid}/", f"users/{uid}/"]
    if subject.workspace_id:
        prefixes.append(f"workspaces/{subject.workspace_id}/")
    if _TRANSCRIPT_ID.fullmatch(uid):
        prefixes.append(f"workspaces/cli-sessions/{uid}/")
    prefixes += [f"oauth-apps/{app_id}/" for app_id in subject.oauth_app_ids]
    return prefixes


async def _sweep_bucket(
    bucket: str, prefixes: list[str], kept: set[str], dry_run: bool
) -> int:
    async with async_storage.Storage() as client:
        names = [
            name
            for prefix in prefixes
            for name in await _list(client, bucket, prefix)
            if name not in kept
        ]
        if not dry_run:
            await _delete_all(client, bucket, names)
    return len(names)


async def _list(client: async_storage.Storage, bucket: str, prefix: str) -> list[str]:
    names: list[str] = []
    page_token: str | None = None
    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token
        page = await client.list_objects(bucket, params=params)
        names += [item["name"] for item in page.get("items", []) if item.get("name")]
        page_token = page.get("nextPageToken")
        if not page_token:
            return names


async def _delete_all(
    client: async_storage.Storage, bucket: str, names: list[str]
) -> None:
    semaphore = asyncio.Semaphore(_CONCURRENT_DELETES)

    async def delete(name: str) -> None:
        async with semaphore:
            try:
                await client.delete(bucket, name)
            except Exception as e:
                # Gone since the listing: that is the state we want.
                if not is_not_found_error(e):
                    raise

    await asyncio.gather(*(delete(name) for name in names))


def _local_dirs(subject: Subject) -> list[Path]:
    """The same data on a self-hosted stack, which keeps it on disk."""
    configured = Settings().config.workspace_storage_dir
    workspaces = Path(configured) if configured else get_data_path() / "workspaces"
    dirs = [media_root() / "users" / subject.user_id]
    if subject.workspace_id:
        dirs.append(workspaces / subject.workspace_id)
    if _TRANSCRIPT_ID.fullmatch(subject.user_id):
        dirs.append(workspaces / "cli-sessions" / subject.user_id)
    return dirs


def _sweep_local(dirs: list[Path], kept: set[Path], dry_run: bool) -> int:
    count = 0
    for directory in dirs:
        if not directory.is_dir():
            continue
        files = [path for path in directory.rglob("*") if path.is_file()]
        doomed = [path for path in files if path not in kept]
        count += len(doomed)
        if dry_run:
            continue
        if len(doomed) < len(files):
            for path in doomed:
                path.unlink(missing_ok=True)
            continue
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
    return count
