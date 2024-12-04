import logging
import os
import uuid

import dotenv
import fastapi
from google.cloud import storage

import backend.server.v2.store.exceptions

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


async def upload_media(user_id: str, file: fastapi.UploadFile) -> str:
    # Check required environment variables first before doing any file processing
    if not os.environ.get("MEDIA_GCS_BUCKET_NAME") or not os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    ):
        logger.error("Missing required GCS environment variables")
        raise backend.server.v2.store.exceptions.StorageConfigError(
            "Missing storage configuration"
        )

    try:
        # Validate file type
        content_type = file.content_type
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
        unique_filename = f"{uuid.uuid4()}{file_ext}"

        # Construct storage path
        media_type = "images" if content_type in ALLOWED_IMAGE_TYPES else "videos"
        storage_path = f"users/{user_id}/{media_type}/{unique_filename}"

        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(os.environ["MEDIA_GCS_BUCKET_NAME"])
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
