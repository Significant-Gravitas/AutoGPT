import logging
import os
import uuid

import fastapi
from gcloud.aio import storage as async_storage

from backend.util.exceptions import MissingConfigError
from backend.util.settings import Settings
from backend.util.virus_scanner import scan_content_safe

from . import exceptions as store_exceptions

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


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
        raise MissingConfigError("GCS media bucket is not configured")

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

    # Detect actual content type from file signature/magic bytes
    # Trust the file signature over the declared content-type header
    detected_content_type: str | None = None

    # Check image file signatures
    if content.startswith(b"\xff\xd8\xff"):  # JPEG
        detected_content_type = "image/jpeg"
    elif content.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
        detected_content_type = "image/png"
    elif content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):  # GIF
        detected_content_type = "image/gif"
    elif content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WEBP":  # WebP
        detected_content_type = "image/webp"
    # Check video file signatures
    elif content.startswith(b"\x00\x00\x00") and len(content) >= 8 and content[4:8] == b"ftyp":  # MP4
        detected_content_type = "video/mp4"
    elif content.startswith(b"\x1a\x45\xdf\xa3"):  # WebM
        detected_content_type = "video/webm"

    # If we detected a valid type, use it; otherwise reject the file
    if detected_content_type is None:
        raise store_exceptions.InvalidFileTypeError(
            "Could not detect a valid image or video file signature. "
            "Supported formats: JPEG, PNG, GIF, WebP, MP4, WebM"
        )

    # Log if we're auto-correcting a mismatched content-type
    if file.content_type != detected_content_type:
        logger.info(
            f"Auto-correcting content-type from '{file.content_type}' to "
            f"'{detected_content_type}' based on file signature"
        )

    # Use the detected content type going forward
    content_type = detected_content_type

    settings = Settings()

    # Check required settings first before doing any file processing
    if not settings.config.media_gcs_bucket_name:
        logger.error("Missing GCS bucket name setting")
        raise store_exceptions.StorageConfigError(
            "Missing storage bucket configuration"
        )

    try:
        # content_type is already validated from file signature detection above

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
