import logging
import os
import uuid

import fastapi
from google.cloud import storage

import backend.server.v2.store.exceptions
from backend.util.exceptions import MissingConfigError
from backend.util.settings import Settings

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

    storage_client = storage.Client()
    bucket = storage_client.bucket(settings.config.media_gcs_bucket_name)

    # Check images
    image_path = f"users/{user_id}/images/{filename}"
    image_blob = bucket.blob(image_path)
    if image_blob.exists():
        return image_blob.public_url

    # Check videos
    video_path = f"users/{user_id}/videos/{filename}"

    video_blob = bucket.blob(video_path)
    if video_blob.exists():
        return video_blob.public_url

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
        raise backend.server.v2.store.exceptions.FileReadError(
            "Failed to read file content"
        ) from e

    # Validate file signature/magic bytes
    if file.content_type in ALLOWED_IMAGE_TYPES:
        # Check image file signatures
        if content.startswith(b"\xFF\xD8\xFF"):  # JPEG
            if file.content_type != "image/jpeg":
                raise backend.server.v2.store.exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            if file.content_type != "image/png":
                raise backend.server.v2.store.exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):  # GIF
            if file.content_type != "image/gif":
                raise backend.server.v2.store.exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"RIFF") and content[8:12] == b"WEBP":  # WebP
            if file.content_type != "image/webp":
                raise backend.server.v2.store.exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        else:
            raise backend.server.v2.store.exceptions.InvalidFileTypeError(
                "Invalid image file signature"
            )

    elif file.content_type in ALLOWED_VIDEO_TYPES:
        # Check video file signatures
        if content.startswith(b"\x00\x00\x00") and (content[4:8] == b"ftyp"):  # MP4
            if file.content_type != "video/mp4":
                raise backend.server.v2.store.exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        elif content.startswith(b"\x1a\x45\xdf\xa3"):  # WebM
            if file.content_type != "video/webm":
                raise backend.server.v2.store.exceptions.InvalidFileTypeError(
                    "File signature does not match content type"
                )
        else:
            raise backend.server.v2.store.exceptions.InvalidFileTypeError(
                "Invalid video file signature"
            )

    settings = Settings()

    # Check required settings first before doing any file processing
    if not settings.config.media_gcs_bucket_name:
        logger.error("Missing GCS bucket name setting")
        raise backend.server.v2.store.exceptions.StorageConfigError(
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
            raise backend.server.v2.store.exceptions.InvalidFileTypeError(
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
                    raise backend.server.v2.store.exceptions.FileSizeTooLargeError(
                        "File too large. Maximum size is 50MB"
                    )
        except backend.server.v2.store.exceptions.FileSizeTooLargeError:
            raise
        except Exception as e:
            logger.error(f"Error reading file chunks: {str(e)}")
            raise backend.server.v2.store.exceptions.FileReadError(
                "Failed to read uploaded file"
            ) from e

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
            storage_client = storage.Client()
            bucket = storage_client.bucket(settings.config.media_gcs_bucket_name)
            blob = bucket.blob(storage_path)
            blob.content_type = content_type

            file_bytes = await file.read()
            blob.upload_from_string(file_bytes, content_type=content_type)

            public_url = blob.public_url

            logger.info(f"Successfully uploaded file to: {storage_path}")
            return public_url

        except Exception as e:
            logger.error(f"GCS storage error: {str(e)}")
            raise backend.server.v2.store.exceptions.StorageUploadError(
                "Failed to upload file to storage"
            ) from e

    except backend.server.v2.store.exceptions.MediaUploadError:
        raise
    except Exception as e:
        logger.exception("Unexpected error in upload_media")
        raise backend.server.v2.store.exceptions.MediaUploadError(
            "Unexpected error during media upload"
        ) from e
