import logging
from datetime import datetime
from typing import Sequence

import fastapi
import prisma.models
from google.cloud import storage
from pydantic import Field

from backend.data.db import BaseDbModel
from backend.util.exceptions import MissingConfigError, NotFoundError
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


# ---------------- MODEL ---------------- #


class File(BaseDbModel):
    user_id: str

    name: str
    size: int = Field(..., description="file size in bytes")
    content_type: str = Field(..., description="MIME content type of the file")
    created_at: datetime

    @classmethod
    def from_db(cls, _file: prisma.models.File) -> "File":
        return cls(
            id=_file.id,
            user_id=_file.userID,
            name=_file.name,
            size=_file.size,
            content_type=_file.contentType,
            created_at=_file.createdAt,
        )


# ---------------- CRUD functions ---------------- #


async def list_files(user_id: str) -> Sequence[File]:
    return [
        File.from_db(f)
        for f in await prisma.models.File.prisma().find_many(where={"userID": user_id})
    ]


async def get_file(file_id: str, user_id: str) -> File:
    file = await prisma.models.File.prisma().find_first(
        where={"id": file_id, "userID": user_id}
    )
    if not file:
        raise NotFoundError(f"File #{file_id} does not exist")
    return File.from_db(file)


async def get_file_content(file_id: str, user_id: str) -> tuple[File, storage.Blob]:
    file = await get_file(file_id=file_id, user_id=user_id)

    blob = _user_file_bucket().get_blob(file.id)
    if not (blob and blob.exists()):
        logger.error(f"File #{file_id} of user #{user_id} not found in bucket")
        raise NotFoundError(f"File #{file_id} not found in storage")
    return file, blob


async def create_file(
    user_id: str, content: bytes, content_type: str, name: str = ""
) -> File:
    file = await prisma.models.File.prisma().create(
        data={
            "userID": user_id,
            "name": name,
            "size": len(content),
            "contentType": content_type,
        }
    )
    _user_file_bucket().blob(file.id).upload_from_string(
        content, content_type=content_type
    )
    return File.from_db(file)


async def create_file_from_upload(
    user_id: str, uploaded_file: fastapi.UploadFile
) -> File:
    # Validate file type
    content_type = uploaded_file.content_type
    if content_type is None:
        raise ValueError(
            "File has no type"
        )  # FIXME: graceful fallback to type detection

    # Validate file size
    if uploaded_file.size is None:
        raise ValueError("File has no size")
    if uploaded_file.size > MAX_FILE_SIZE:
        raise ValueError("File is too large: maximum size is 50MiB")

    file = await prisma.models.File.prisma().create(
        data={
            "userID": user_id,
            "name": uploaded_file.filename or "",
            "size": uploaded_file.size,
            "contentType": content_type,
        }
    )
    _user_file_bucket().blob(file.id).upload_from_file(
        uploaded_file, content_type=content_type
    )
    return File.from_db(file)


# ---------------- UTILITIES ---------------- #


def _user_file_bucket() -> storage.Bucket:
    if not settings.secrets.user_file_gcs_bucket_name:
        raise MissingConfigError("Missing storage bucket configuration")

    # TODO: use S3 API instead to allow use of other cloud storage providers
    storage_client = storage.Client()
    return storage_client.bucket(settings.secrets.user_file_gcs_bucket_name)
