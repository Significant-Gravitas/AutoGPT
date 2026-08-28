import logging
import mimetypes
import os
import uuid
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import aiofiles
import fastapi
from gcloud.aio import storage as async_storage

from backend.util.data import get_data_path
from backend.util.exceptions import MissingConfigError
from backend.util.settings import Settings
from backend.util.virus_scanner import scan_content_safe

from . import exceptions as store_exceptions

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def get_local_media_root() -> str:
    settings = Settings()
    return os.path.realpath(
        settings.config.media_storage_dir or get_data_path() / "media"
    )


def get_local_media_path(storage_path: str) -> Path:
    root = get_local_media_root()
    path = os.path.realpath(os.path.join(root, storage_path))
    if not path.startswith(os.path.join(root, "")):
        raise ValueError("Invalid media path")
    return Path(path)


def _safe_path_segment(value: str) -> str:
    normalized = value.replace("\\", "/")
    safe_value = os.path.basename(normalized)
    if not safe_value or safe_value in {".", ".."} or safe_value != normalized:
        raise ValueError("Invalid media path segment")
    return safe_value


def _platform_path_prefix(platform_base_url: str) -> str:
    if not platform_base_url:
        return ""
    raw_path = urlsplit(platform_base_url).path.replace("\\", "/")
    parts: list[str] = []
    for raw_part in raw_path.split("/"):
        part = unquote(raw_part)
        if not part or part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(quote(part, safe=":@-._~!$&'()*+,;="))
    return f"/{'/'.join(parts)}" if parts else ""


def _url_path_segment(value: str) -> str:
    return quote(_safe_path_segment(value), safe="")


def _local_store_media_storage_path(
    organization_id: str,
    team_id: str | None,
    media_type: str,
    filename: str,
) -> str:
    safe_org_id = _safe_path_segment(organization_id)
    safe_media_type = _safe_path_segment(media_type)
    safe_filename = _safe_path_segment(filename)
    if safe_media_type not in {"images", "videos"}:
        raise ValueError("Invalid media type")
    if team_id:
        safe_team_id = _safe_path_segment(team_id)
        owner_path = f"orgs/{safe_org_id}/teams/{safe_team_id}"
    else:
        owner_path = f"orgs/{safe_org_id}/home"
    return f"store/{owner_path}/{safe_media_type}/{safe_filename}"


def get_local_store_media_path(
    organization_id: str,
    team_id: str | None,
    media_type: str,
    filename: str,
) -> Path:
    return get_local_media_path(
        _local_store_media_storage_path(organization_id, team_id, media_type, filename)
    )


def get_local_store_media_url(
    organization_id: str,
    team_id: str | None,
    media_type: str,
    filename: str,
) -> str:
    public_path_prefix = _platform_path_prefix(Settings().config.platform_base_url)
    api_path = f"{public_path_prefix}/api"
    safe_org_id = _url_path_segment(organization_id)
    safe_media_type = _url_path_segment(media_type)
    safe_filename = _url_path_segment(filename)
    if team_id:
        safe_team_id = _url_path_segment(team_id)
        return (
            f"{api_path}/store/media/orgs/{safe_org_id}/teams/{safe_team_id}/"
            f"{safe_media_type}/{safe_filename}"
        )
    return (
        f"{api_path}/store/media/orgs/{safe_org_id}/{safe_media_type}/{safe_filename}"
    )


def get_local_store_media_type(filename: str) -> str | None:
    content_type = mimetypes.guess_type(filename)[0]
    if content_type in ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES:
        return content_type
    return None


async def check_media_exists(
    user_id: str,
    filename: str,
    organization_id: str | None = None,
    team_id: str | None = None,
    local_store_media: bool = False,
) -> str | None:
    """
    Check if a media file exists in storage for the given user.
    Tries both images and videos directories.

    Args:
        user_id (str): ID of the user who uploaded the file
        filename (str): Name of the file to check

    Returns:
        str | None: URL of the blob if it exists, None otherwise
    """
    settings = Settings()
    if not settings.config.media_gcs_bucket_name:
        if local_store_media and organization_id:
            for media_type in ("images", "videos"):
                try:
                    path = get_local_store_media_path(
                        organization_id, team_id, media_type, filename
                    )
                except ValueError:
                    return None
                if path.is_file():
                    return get_local_store_media_url(
                        organization_id, team_id, media_type, filename
                    )
            return None
        raise MissingConfigError("GCS media bucket is not configured")

    async with async_storage.Storage() as async_client:
        bucket_name = settings.config.media_gcs_bucket_name

        # Check images
        owner_path = (
            f"orgs/{organization_id}" if organization_id else f"users/{user_id}"
        )
        image_path = f"{owner_path}/images/{filename}"
        try:
            await async_client.download_metadata(bucket_name, image_path)
            # If we get here, the file exists - construct public URL
            return f"https://storage.googleapis.com/{bucket_name}/{image_path}"
        except Exception:
            # File doesn't exist, continue to check videos
            pass

        # Check videos
        video_path = f"{owner_path}/videos/{filename}"
        try:
            await async_client.download_metadata(bucket_name, video_path)
            # If we get here, the file exists - construct public URL
            return f"https://storage.googleapis.com/{bucket_name}/{video_path}"
        except Exception:
            # File doesn't exist
            pass

        return None


async def upload_media(
    user_id: str,
    file: fastapi.UploadFile,
    use_file_name: bool = False,
    organization_id: str | None = None,
    team_id: str | None = None,
    local_store_media: bool = False,
) -> str:
    """Validate, virus-scan, and upload a media file to GCS.

    When ``organization_id`` is set the file is stored under the org-scoped
    path ``orgs/{organization_id}/...`` instead of ``users/{user_id}/...``.
    Both IDs come from the server-side auth context, never from the client.
    """
    # Get file content for deeper validation
    try:
        content = await file.read(1024)  # Read first 1KB for validation
        await file.seek(0)  # Reset file pointer
    except Exception as e:
        logger.error(f"Error reading file content: {str(e)}")
        raise store_exceptions.FileReadError("Failed to read file content") from e

    # Validate file signature/magic bytes
    if file.content_type in ALLOWED_IMAGE_TYPES:
        # Check image file signatures
        if content.startswith(b"\xff\xd8\xff"):  # JPEG
            if file.content_type != "image/jpeg":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            if file.content_type != "image/png":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):  # GIF
            if file.content_type != "image/gif":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"RIFF") and content[8:12] == b"WEBP":  # WebP
            if file.content_type != "image/webp":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        else:
            raise store_exceptions.InvalidFileTypeError("Invalid image file signature")

    elif file.content_type in ALLOWED_VIDEO_TYPES:
        # Check video file signatures
        if content.startswith(b"\x00\x00\x00") and (content[4:8] == b"ftyp"):  # MP4
            if file.content_type != "video/mp4":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"\x1a\x45\xdf\xa3"):  # WebM
            if file.content_type != "video/webm":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        else:
            raise store_exceptions.InvalidFileTypeError("Invalid video file signature")

    settings = Settings()
    if not settings.config.media_gcs_bucket_name and not organization_id:
        logger.error("Missing GCS bucket name setting")
        raise store_exceptions.StorageConfigError(
            "Missing storage bucket configuration"
        )

    try:
        # Validate file type
        content_type = file.content_type
        if content_type is None:
            content_type = "image/jpeg"

        if (
            content_type not in ALLOWED_IMAGE_TYPES
            and content_type not in ALLOWED_VIDEO_TYPES
        ):
            logger.warning(f"Invalid file type attempted: {content_type}")
            raise store_exceptions.InvalidFileTypeError(
                f"File type not supported. Must be jpeg, png, gif, webp, mp4 or webm. Content type: {content_type}"
            )

        # Validate file size
        file_size = 0
        chunk_size = 8192  # 8KB chunks

        try:
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    logger.warning(f"File size too large: {file_size} bytes")
                    raise store_exceptions.FileSizeTooLargeError(
                        "File too large. Maximum size is 50MB"
                    )
        except store_exceptions.FileSizeTooLargeError:
            raise
        except Exception as e:
            logger.error(f"Error reading file chunks: {str(e)}")
            raise store_exceptions.FileReadError("Failed to read uploaded file") from e

        # Reset file pointer
        await file.seek(0)

        # Generate unique filename
        filename = file.filename or ""
        if use_file_name:
            normalized_filename = filename.replace("\\", "/")
            safe_filename = os.path.basename(normalized_filename)
            if (
                not safe_filename
                or safe_filename in {".", ".."}
                or safe_filename != normalized_filename
            ):
                raise store_exceptions.InvalidFileTypeError("Invalid file name")
            unique_filename = safe_filename
        else:
            file_ext = os.path.splitext(filename)[1].lower()
            unique_filename = f"{uuid.uuid4()}{file_ext}"

        owner_id = organization_id or user_id
        try:
            safe_owner_id = _safe_path_segment(owner_id)
        except ValueError:
            raise store_exceptions.StorageUploadError("Invalid media owner path")

        media_type = "images" if content_type in ALLOWED_IMAGE_TYPES else "videos"
        if organization_id:
            storage_path = f"orgs/{safe_owner_id}/{media_type}/{unique_filename}"
        else:
            storage_path = f"users/{safe_owner_id}/{media_type}/{unique_filename}"

        file_bytes = await file.read()
        await scan_content_safe(file_bytes, filename=unique_filename)

        if not settings.config.media_gcs_bucket_name:
            if not organization_id:
                logger.error("Missing GCS bucket name setting")
                raise store_exceptions.StorageConfigError(
                    "Missing storage bucket configuration"
                )
            try:
                if local_store_media:
                    local_path = get_local_store_media_path(
                        organization_id, team_id, media_type, unique_filename
                    )
                else:
                    local_path = get_local_media_path(storage_path)
            except ValueError as error:
                raise store_exceptions.StorageUploadError(
                    "Invalid media storage path"
                ) from error
            local_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(local_path, "wb") as local_file:
                await local_file.write(file_bytes)
            logger.info("Successfully uploaded file to local storage: %s", local_path)
            if local_store_media:
                return get_local_store_media_url(
                    organization_id, team_id, media_type, unique_filename
                )
            return f"/api/orgs/{organization_id}/avatar/{unique_filename}"

        try:
            async with async_storage.Storage() as async_client:
                bucket_name = settings.config.media_gcs_bucket_name

                # Upload using pure async client
                await async_client.upload(
                    bucket_name, storage_path, file_bytes, content_type=content_type
                )

                # Construct public URL
                public_url = (
                    f"https://storage.googleapis.com/{bucket_name}/{storage_path}"
                )

                logger.info(f"Successfully uploaded file to: {storage_path}")
                return public_url

        except Exception as e:
            logger.error(f"GCS storage error: {str(e)}")
            raise store_exceptions.StorageUploadError(
                "Failed to upload file to storage"
            ) from e

    except store_exceptions.MediaUploadError:
        raise
    except Exception as e:
        logger.exception("Unexpected error in upload_media")
        raise store_exceptions.MediaUploadError(
            "Unexpected error during media upload"
        ) from e
