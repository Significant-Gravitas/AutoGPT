import logging

import fastapi
from gcloud.aio import storage as async_storage

from backend.util.gcs_utils import is_not_found_error
from backend.util.settings import Config, Settings
from backend.util.virus_scanner import scan_content_safe

from . import exceptions as store_exceptions
from . import local_media, submission_media

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PRIVATE_IMAGE_FILE_SIZE = submission_media.PRIVATE_MEDIA_PROXY_BUFFER_BYTES


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
    config = Settings().config
    bucket_name = config.resolved_private_user_data_bucket
    if not bucket_name:
        return await local_media.check_media_exists(user_id, filename)
    try:
        safe_user_id = local_media.validate_path_component(user_id)
        safe_filename = local_media.validate_path_component(filename)
    except ValueError:
        return None
    content_type = local_media.content_type_for_filename(safe_filename)
    if content_type is None:
        return None
    safe_filename = local_media.stored_filename(safe_filename, content_type, True)

    async with async_storage.Storage() as async_client:
        image_path = f"users/{safe_user_id}/images/{safe_filename}"
        try:
            await async_client.download_metadata(bucket_name, image_path)
            return _stored_media_url(
                _serves_private_urls(config),
                bucket_name,
                safe_user_id,
                "images",
                safe_filename,
            )
        except Exception as error:
            if not _is_missing_media_error(error):
                raise

        video_path = f"users/{safe_user_id}/videos/{safe_filename}"
        try:
            await async_client.download_metadata(bucket_name, video_path)
            return _stored_media_url(
                _serves_private_urls(config),
                bucket_name,
                safe_user_id,
                "videos",
                safe_filename,
            )
        except Exception as error:
            if not _is_missing_media_error(error):
                raise

        return None


async def upload_media(
    user_id: str,
    file: fastapi.UploadFile,
    use_file_name: bool = False,
    is_avatar: bool = False,
) -> str:
    # Get file content for deeper validation
    try:
        content = await file.read(1024)  # Read first 1KB for validation
        await file.seek(0)  # Reset file pointer
    except Exception as e:
        logger.error(f"Error reading file content: {str(e)}")
        raise store_exceptions.FileReadError("Failed to read file content") from e

    content_type = file.content_type
    if content_type is None:
        content_type = "image/jpeg"

    # Validate file signature/magic bytes
    if content_type in ALLOWED_IMAGE_TYPES:
        # Check image file signatures
        if content.startswith(b"\xff\xd8\xff"):  # JPEG
            if content_type != "image/jpeg":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            if content_type != "image/png":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):  # GIF
            if content_type != "image/gif":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"RIFF") and content[8:12] == b"WEBP":  # WebP
            if content_type != "image/webp":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        else:
            raise store_exceptions.InvalidFileTypeError("Invalid image file signature")

    elif content_type in ALLOWED_VIDEO_TYPES:
        # Check video file signatures
        if content.startswith(b"\x00\x00\x00") and (content[4:8] == b"ftyp"):  # MP4
            if content_type != "video/mp4":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"\x1a\x45\xdf\xa3"):  # WebM
            if content_type != "video/webm":
                raise store_exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        else:
            raise store_exceptions.InvalidFileTypeError("Invalid video file signature")

    config = Settings().config
    bucket_name = config.resolved_private_user_data_bucket
    use_local_storage = not bucket_name
    private_media_proxy_enabled = _serves_private_urls(config)

    try:
        # Validate file type
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
        max_file_size = (
            MAX_PRIVATE_IMAGE_FILE_SIZE
            if private_media_proxy_enabled and content_type in ALLOWED_IMAGE_TYPES
            else MAX_FILE_SIZE
        )
        max_file_size_mb = max_file_size // (1024 * 1024)

        try:
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                if file_size > max_file_size:
                    logger.warning(f"File size too large: {file_size} bytes")
                    raise store_exceptions.FileSizeTooLargeError(
                        f"File too large. Maximum size is {max_file_size_mb}MB"
                    )
        except store_exceptions.FileSizeTooLargeError:
            raise
        except Exception as e:
            logger.error(f"Error reading file chunks: {str(e)}")
            raise store_exceptions.FileReadError("Failed to read uploaded file") from e

        if is_avatar:
            if content_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise fastapi.HTTPException(
                    400, "Choose a PNG, JPEG, or WebP image for your appearance."
                )
            if file_size > 5 * 1024 * 1024:
                raise fastapi.HTTPException(
                    413, "Appearance images must be 5 MB or smaller."
                )

        # Reset file pointer
        await file.seek(0)

        media_type = "images" if content_type in ALLOWED_IMAGE_TYPES else "videos"
        unique_filename = local_media.stored_filename(
            file.filename or "", content_type, use_file_name and not is_avatar
        )

        if use_local_storage:
            file_bytes = await file.read()
            await scan_content_safe(file_bytes, filename=unique_filename)
            return await local_media.store_media(
                user_id, media_type, unique_filename, file_bytes
            )

        storage_path = submission_media.object_path(
            user_id, media_type, unique_filename
        )

        try:
            async with async_storage.Storage() as async_client:
                file_bytes = await file.read()
                await scan_content_safe(file_bytes, filename=unique_filename)

                # Upload using pure async client
                await async_client.upload(
                    bucket_name, storage_path, file_bytes, content_type=content_type
                )

                logger.info(f"Successfully uploaded file to: {storage_path}")
                return _stored_media_url(
                    _serves_private_urls(config),
                    bucket_name,
                    user_id,
                    media_type,
                    unique_filename,
                )

        except fastapi.HTTPException:
            raise
        except Exception as e:
            logger.error(f"GCS storage error: {str(e)}")
            raise store_exceptions.StorageUploadError(
                "Failed to upload file to storage"
            ) from e

    except (store_exceptions.MediaUploadError, fastapi.HTTPException):
        raise
    except Exception as e:
        logger.exception("Unexpected error in upload_media")
        raise store_exceptions.MediaUploadError(
            "Unexpected error during media upload"
        ) from e


def _serves_private_urls(config: Config) -> bool:
    """Only the legacy single-bucket setup still hands out direct GCS URLs."""
    private_bucket = config.resolved_private_user_data_bucket
    return bool(
        private_bucket and private_bucket != config.resolved_public_site_media_bucket
    )


def _stored_media_url(
    buckets_are_split: bool,
    bucket_name: str,
    user_id: str,
    media_type: str,
    filename: str,
) -> str:
    if buckets_are_split:
        return submission_media.url(user_id, media_type, filename)
    object_path = submission_media.object_path(user_id, media_type, filename)
    return f"https://storage.googleapis.com/{bucket_name}/{object_path}"


def _is_missing_media_error(error: Exception) -> bool:
    return isinstance(error, FileNotFoundError) or is_not_found_error(error)
