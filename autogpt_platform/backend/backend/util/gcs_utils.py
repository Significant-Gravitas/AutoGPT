"""
Shared GCS utilities for workspace and cloud storage backends.

This module provides common functionality for working with Google Cloud Storage,
including path parsing, client management, and signed URL generation.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import aiohttp
from gcloud.aio import storage as async_gcs_storage
from google.cloud import storage as gcs_storage

logger = logging.getLogger(__name__)


def parse_gcs_path(path: str) -> tuple[str, str]:
    """
    Parse a GCS path in the format 'gcs://bucket/blob' to (bucket, blob).

    Args:
        path: GCS path string (e.g., "gcs://my-bucket/path/to/file")

    Returns:
        Tuple of (bucket_name, blob_name)

    Raises:
        ValueError: If the path format is invalid
    """
    if not path.startswith("gcs://"):
        raise ValueError(f"Invalid GCS path: {path}")

    path_without_prefix = path[6:]  # Remove "gcs://"
    parts = path_without_prefix.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS path format: {path}")

    return parts[0], parts[1]


async def download_with_fresh_session(bucket: str, blob: str) -> bytes:
    """
    Download file content using a fresh session.

    This approach avoids event loop issues that can occur when reusing
    sessions across different async contexts (e.g., in executors).

    Args:
        bucket: GCS bucket name
        blob: Blob path within the bucket

    Returns:
        File content as bytes

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=10, force_close=True)
    )
    client: async_gcs_storage.Storage | None = None
    try:
        client = async_gcs_storage.Storage(session=session)
        content = await client.download(bucket, blob)
        return content
    except Exception as e:
        if "404" in str(e) or "Not Found" in str(e):
            raise FileNotFoundError(f"File not found: gcs://{bucket}/{blob}")
        raise
    finally:
        if client:
            try:
                await client.close()
            except Exception:
                pass  # Best-effort cleanup
        await session.close()


async def download_range(bucket: str, blob: str, max_bytes: int) -> bytes:
    """
    Download only the first ``max_bytes`` of a GCS object using a Range request.

    Falls back to a full download when the installed ``gcloud-aio`` build does
    not accept a ``headers`` kwarg; the result is sliced to ``max_bytes`` either
    way so callers always get at most ``max_bytes``.

    Args:
        bucket: GCS bucket name
        blob: Blob path within the bucket
        max_bytes: Maximum number of leading bytes to fetch

    Returns:
        Up to ``max_bytes`` of file content as bytes

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=10, force_close=True)
    )
    client: async_gcs_storage.Storage | None = None
    try:
        client = async_gcs_storage.Storage(session=session)
        try:
            content = await client.download(
                bucket, blob, headers={"Range": f"bytes=0-{max(0, max_bytes - 1)}"}
            )
        except TypeError:
            # Older gcloud-aio without a headers kwarg: full download, then slice.
            content = await client.download(bucket, blob)
        return content[:max_bytes]
    except Exception as e:
        if "404" in str(e) or "Not Found" in str(e):
            raise FileNotFoundError(f"File not found: gcs://{bucket}/{blob}")
        raise
    finally:
        if client:
            try:
                await client.close()
            except Exception:
                pass  # Best-effort cleanup
        await session.close()


async def generate_signed_url(
    sync_client: gcs_storage.Client,
    bucket_name: str,
    blob_name: str,
    expires_in: int,
) -> str:
    """
    Generate a signed URL for temporary access to a GCS file.

    Uses asyncio.to_thread() to run the sync operation without blocking.

    Args:
        sync_client: Sync GCS client with service account credentials
        bucket_name: GCS bucket name
        blob_name: Blob path within the bucket
        expires_in: URL expiration time in seconds

    Returns:
        Signed URL string
    """
    bucket = sync_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return await asyncio.to_thread(
        blob.generate_signed_url,
        version="v4",
        expiration=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        method="GET",
    )
