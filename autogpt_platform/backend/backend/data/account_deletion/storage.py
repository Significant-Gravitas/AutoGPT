"""The account's files: object storage on a cloud stack, the data directory on
a self-hosted one.

Every prefix below holds exactly one account's objects, keyed by its user ID,
its `UserWorkspace` ID (one per user, never shared) or an OAuth app it owns.
Marketplace media an account uploaded lives under its own `users/{id}/` even
when an org listing shows it, so that goes too.
"""

import asyncio
import logging
import re
import shutil
from pathlib import Path

from gcloud.aio import storage as async_storage

from backend.api.features.store.local_media import media_root
from backend.data.account_deletion.models import StepReport, StepStatus, Subject
from backend.util.data import get_data_path
from backend.util.gcs_utils import is_not_found_error
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

# User, workspace and app IDs are UUIDs. Anything else is refused rather than
# turned into a prefix, so a malformed ID can never widen a sweep.
_SAFE_ID = re.compile(r"[0-9A-Za-z-]{8,64}")
# `copilot/transcript.py` strips everything but hex and hyphens from the user
# ID in transcript paths, so only such an ID maps onto its own folder there.
_TRANSCRIPT_ID = re.compile(r"[0-9a-fA-F-]{8,36}")
_CONCURRENT_DELETES = 16


async def delete_files(subject: Subject, *, dry_run: bool) -> StepReport:
    _require_safe_ids(subject)
    buckets = _buckets()
    if buckets:
        counts = {
            bucket_label: await _sweep_bucket(bucket, _prefixes(subject), dry_run)
            for bucket_label, bucket in buckets.items()
        }
    else:
        counts = {"local_files": _sweep_local(_local_dirs(subject), dry_run)}
    return StepReport(
        name="files",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts=counts,
    )


def _require_safe_ids(subject: Subject) -> None:
    ids = [subject.user_id, *subject.oauth_app_ids]
    if subject.workspace_id:
        ids.append(subject.workspace_id)
    unsafe = [i for i in ids if not _SAFE_ID.fullmatch(i)]
    if unsafe:
        raise ValueError(f"{len(unsafe)} IDs are not safe as storage prefixes")


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


async def _sweep_bucket(bucket: str, prefixes: list[str], dry_run: bool) -> int:
    async with async_storage.Storage() as client:
        names = [
            name for prefix in prefixes for name in await _list(client, bucket, prefix)
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


def _sweep_local(dirs: list[Path], dry_run: bool) -> int:
    count = 0
    for directory in dirs:
        if not directory.is_dir():
            continue
        count += sum(1 for path in directory.rglob("*") if path.is_file())
        if dry_run:
            continue
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
    return count
