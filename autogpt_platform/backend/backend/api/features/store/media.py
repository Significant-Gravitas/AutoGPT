import logging
import os
import re
import uuid
from pathlib import Path

import aiofiles
import fastapi
from gcloud.aio import storage as async_storage

from backend.util.data import get_data_path
from backend.util.settings import Settings
from backend.util.virus_scanner import scan_content_safe

from . import exceptions as store_exceptions

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MEDIA_TYPES = frozenset({"images", "videos"})
LOCAL_MEDIA_DIR_NAME = "store-media"

_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpeg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}
EXTENSION_CONTENT_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


async def check_media_exists(user_id: str, filename: str) -> str | None:
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
        return _check_media_exists_locally(user_id, filename)

    async with async_storage.Storage() as async_client:
        bucket_name = settings.config.media_gcs_bucket_name

        # Check images
        image_path = f"users/{user_id}/images/{filename}"
        try:
            await async_client.download_metadata(bucket_name, image_path)
            # If we get here, the file exists - construct public URL
            return f"https://storage.googleapis.com/{bucket_name}/{image_path}"
        except Exception:
            # File doesn't exist, continue to check videos
            pass

        # Check videos
        video_path = f"users/{user_id}/videos/{filename}"
        try:
            await async_client.download_metadata(bucket_name, video_path)
            # If we get here, the file exists - construct public URL
            return f"https://storage.googleapis.com/{bucket_name}/{video_path}"
        except Exception:
            # File doesn't exist
            pass

        return None


async def upload_media(
    user_id: str, file: fastapi.UploadFile, use_file_name: bool = False
) -> str:
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
    use_local_storage = not settings.config.media_gcs_bucket_name

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
        file_ext = os.path.splitext(filename)[1].lower()
        if use_file_name:
            unique_filename = filename
        else:
            unique_filename = f"{uuid.uuid4()}{file_ext}"

        # Construct storage path
        media_type = "images" if content_type in ALLOWED_IMAGE_TYPES else "videos"

        if use_local_storage:
            unique_filename = _local_stored_filename(
                unique_filename, content_type, use_file_name
            )
            file_bytes = await file.read()
            await scan_content_safe(file_bytes, filename=unique_filename)
            return await _store_media_locally(
                user_id, media_type, unique_filename, file_bytes
            )

        storage_path = f"users/{user_id}/{media_type}/{unique_filename}"

        try:
            async with async_storage.Storage() as async_client:
                bucket_name = settings.config.media_gcs_bucket_name

                file_bytes = await file.read()
                await scan_content_safe(file_bytes, filename=unique_filename)

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


def content_type_for_filename(filename: str) -> str | None:
    return EXTENSION_CONTENT_TYPES.get(os.path.splitext(filename)[1].lower())


def get_local_media_path(user_id: str, media_type: str, filename: str) -> Path:
    if media_type not in MEDIA_TYPES:
        raise ValueError(f"Invalid media type: {media_type!r}")

    base_dir = local_media_root()
    candidate = os.path.join(
        str(base_dir),
        "users",
        _validate_path_component(user_id),
        media_type,
        _validate_path_component(filename),
    )
    real_base_dir = os.path.realpath(str(base_dir))
    real_candidate = os.path.realpath(candidate)
    if not real_candidate.startswith(real_base_dir + os.sep):
        raise ValueError("Invalid media path: path traversal detected")
    return Path(real_candidate)


def local_media_root() -> Path:
    storage_dir = Settings().config.workspace_storage_dir
    if storage_dir:
        return Path(storage_dir).resolve().parent / LOCAL_MEDIA_DIR_NAME
    return Path(get_data_path()) / LOCAL_MEDIA_DIR_NAME


def local_media_url(user_id: str, media_type: str, filename: str) -> str:
    path = (
        f"/api/store/media/{_validate_path_component(user_id)}"
        f"/{media_type}/{_validate_path_component(filename)}"
    )
    base_url = Settings().config.platform_base_url.rstrip("/")
    return f"{base_url}{path}" if base_url else path


def _validate_path_component(value: str) -> str:
    if not _SAFE_PATH_COMPONENT.fullmatch(value):
        raise ValueError(f"Invalid media path component: {value!r}")
    return value


def _local_stored_filename(
    filename: str, content_type: str, use_file_name: bool
) -> str:
    ext = CONTENT_TYPE_EXTENSIONS[content_type]
    if use_file_name:
        return _validate_path_component(f"{Path(filename).stem}{ext}")
    return f"{uuid.uuid4()}{ext}"


def _check_media_exists_locally(user_id: str, filename: str) -> str | None:
    try:
        safe_user_id = _validate_path_component(user_id)
        safe_filename = _validate_path_component(filename)
    except ValueError:
        return None

    for media_type in ("images", "videos"):
        if get_local_media_path(safe_user_id, media_type, safe_filename).is_file():
            return local_media_url(safe_user_id, media_type, safe_filename)
    return None


async def _store_media_locally(
    user_id: str, media_type: str, filename: str, content: bytes
) -> str:
    file_path = get_local_media_path(user_id, media_type, filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    logger.info(f"Successfully uploaded file to local storage: {file_path}")
    return local_media_url(user_id, media_type, filename)
