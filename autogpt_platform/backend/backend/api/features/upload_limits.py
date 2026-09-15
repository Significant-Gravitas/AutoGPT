"""Reading an uploaded body without trusting its size.

An upload is the one request where the caller decides how many bytes the process
holds, so the cap is enforced as the body arrives rather than after all of it has
landed. Shared by every route that takes an archive, because a cap that only one
of them applies is not a cap.
"""

from fastapi import HTTPException, UploadFile

_CHUNK_BYTES = 64 * 1024


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload with an early abort, so a body over the cap is refused
    without ever being held whole."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Archive is larger than the {max_bytes}-byte limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)
