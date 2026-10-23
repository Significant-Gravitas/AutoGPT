from e2b import AsyncSandbox

from backend.copilot.tool_image import MAX_IMAGE_BYTES


async def read_sandbox_image(sandbox: AsyncSandbox, path: str) -> bytes:
    """Stop downloading after the image limit plus one byte, and close the stream."""
    data = bytearray()
    stream = await sandbox.files.read(path, format="stream")
    async with stream:
        async for chunk in stream:
            data.extend(chunk[: MAX_IMAGE_BYTES + 1 - len(data)])
            if len(data) > MAX_IMAGE_BYTES:
                break
    return bytes(data)
