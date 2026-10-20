import asyncio
import os
import re
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from backend.util.data import get_data_path
from backend.util.settings import Settings

MEDIA_TYPES = frozenset({"images", "videos"})
CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpeg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}
EXTENSION_CONTENT_TYPES = {
    extension: content_type
    for content_type, extension in CONTENT_TYPE_EXTENSIONS.items()
} | {".jpg": "image/jpeg"}
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


async def store_media(
    user_id: str, media_type: str, filename: str, content: bytes
) -> str:
    """Publish fully validated media atomically so readers never see partial writes."""
    await asyncio.to_thread(_write_media, user_id, media_type, filename, content)
    return media_url(user_id, media_type, filename)


async def check_media_exists(user_id: str, filename: str) -> str | None:
    """Find named uploads using the same extension normalization as storage."""
    return await asyncio.to_thread(_check_media_exists, user_id, filename)


async def media_metadata(
    user_id: str, media_type: str, filename: str
) -> dict[str, str]:
    return await asyncio.to_thread(_media_metadata, user_id, media_type, filename)


async def stream_media(
    user_id: str,
    media_type: str,
    filename: str,
    byte_range: tuple[int, int] | None = None,
) -> AsyncIterator[bytes]:
    path = get_media_path(user_id, media_type, filename)
    if not path.is_file():
        raise FileNotFoundError("Media not found")

    file = await asyncio.to_thread(path.open, "rb")
    try:
        remaining: int | None = None
        if byte_range is not None:
            start, end = byte_range
            await asyncio.to_thread(file.seek, start)
            remaining = end - start + 1
        while remaining is None or remaining > 0:
            read_size = 64 * 1024 if remaining is None else min(64 * 1024, remaining)
            chunk = await asyncio.to_thread(file.read, read_size)
            if not chunk:
                break
            yield chunk
            if remaining is not None:
                remaining -= len(chunk)
    finally:
        await asyncio.to_thread(file.close)


def get_media_path(user_id: str, media_type: str, filename: str) -> Path:
    """Resolve a file inside the media root, rejecting traversal and escaping symlinks."""
    if media_type not in MEDIA_TYPES:
        raise ValueError("Invalid media type")

    base_dir = os.path.realpath(media_root())
    candidate = os.path.realpath(
        os.path.join(
            base_dir,
            "users",
            validate_path_component(user_id),
            media_type,
            validate_path_component(filename),
        )
    )
    if not candidate.startswith(base_dir + os.sep):
        raise ValueError("Invalid media path")
    return Path(candidate)


def content_type_for_filename(filename: str) -> str | None:
    """Only serve the image and video formats accepted by marketplace uploads."""
    return EXTENSION_CONTENT_TYPES.get(Path(filename).suffix.lower())


def stored_filename(filename: str, content_type: str, use_file_name: bool) -> str:
    """Derive the extension from validated content rather than a client-supplied name."""
    extension = CONTENT_TYPE_EXTENSIONS[content_type]
    if use_file_name:
        return f"{Path(validate_path_component(filename)).stem}{extension}"
    return f"{uuid.uuid4()}{extension}"


def media_root() -> Path:
    """Keep marketplace files beside workspaces, including persistent /data installs."""
    storage_dir = Settings().config.workspace_storage_dir
    if storage_dir:
        return Path(storage_dir).resolve().parent / "store-media"
    return get_data_path() / "store-media"


def media_url(user_id: str, media_type: str, filename: str) -> str:
    """Return a public backend URL or a same-origin path for local installations."""
    path = (
        f"/api/store/media/{validate_path_component(user_id)}"
        f"/{media_type}/{validate_path_component(filename)}"
    )
    return f"{Settings().config.platform_base_url.rstrip('/')}{path}"


def validate_path_component(value: str) -> str:
    if value in {".", ".."} or not _SAFE_PATH_COMPONENT.fullmatch(value):
        raise ValueError("Invalid media path component")
    return value


def _check_media_exists(user_id: str, filename: str) -> str | None:
    try:
        validate_path_component(user_id)
        validate_path_component(filename)
        content_type = content_type_for_filename(filename)
        if content_type is None:
            return None
        canonical_filename = stored_filename(filename, content_type, True)
        for media_type in ("images", "videos"):
            if get_media_path(user_id, media_type, canonical_filename).is_file():
                return media_url(user_id, media_type, canonical_filename)
    except ValueError:
        return None
    return None


def _media_metadata(user_id: str, media_type: str, filename: str) -> dict[str, str]:
    path = get_media_path(user_id, media_type, filename)
    try:
        return {"size": str(path.stat().st_size)}
    except FileNotFoundError as error:
        raise FileNotFoundError("Media not found") from error


def _write_media(user_id: str, media_type: str, filename: str, content: bytes) -> None:
    file_path = get_media_path(user_id, media_type, filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=file_path.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(content)
        os.replace(temporary_path, file_path)
    finally:
        temporary_path.unlink(missing_ok=True)
