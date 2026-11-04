from e2b import AsyncSandbox

from backend.copilot.tool_image import MAX_IMAGE_BYTES
from backend.util.sandbox_output import read_sandbox_file


async def read_sandbox_image(sandbox: AsyncSandbox, path: str) -> bytes:
    """Stop downloading after the image limit plus one byte, and close the stream."""
    return await read_sandbox_file(sandbox, path, MAX_IMAGE_BYTES)
