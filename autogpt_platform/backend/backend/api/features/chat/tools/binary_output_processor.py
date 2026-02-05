"""
Content-based detection and saving of binary data in block outputs.

This module post-processes block execution outputs to detect and save binary
content (images, PDFs) to the workspace, returning workspace:// references
instead of raw base64 data. This reduces LLM output token usage by ~97% for
file generation tasks.

Detection is content-based (not field-name based) because:
- Code execution blocks return base64 in stdout_logs, not structured fields
- The png/jpeg/pdf fields only populate from Jupyter display mechanisms
- Other blocks use various field names: image, result, output, response, etc.
"""

import base64
import binascii
import hashlib
import logging
import re
import uuid
from typing import Any, Optional

from backend.util.file import sanitize_filename
from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# Only process strings larger than this (filters out tokens, hashes, short strings)
SIZE_THRESHOLD = 1024  # 1KB

# Data URI pattern with mimetype extraction
DATA_URI_PATTERN = re.compile(
    r"^data:([a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+);base64,(.+)$",
    re.DOTALL,
)

# Only process these mimetypes from data URIs (avoid text/plain, etc.)
ALLOWED_MIMETYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",  # Non-standard but sometimes used
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "application/pdf",
    "application/octet-stream",
}

# Base64 character validation (strict - must be pure base64)
BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/\n\r]+=*$")

# Magic numbers for binary file detection
# Note: WebP requires two-step detection: RIFF prefix + WEBP at offset 8
MAGIC_SIGNATURES = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"%PDF-", "pdf"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"RIFF", "webp"),  # Special case: also check content[8:12] == b'WEBP'
]


async def process_binary_outputs(
    outputs: dict[str, list[Any]],
    workspace_manager: WorkspaceManager,
    block_name: str,
) -> dict[str, list[Any]]:
    """
    Scan all string values in outputs and replace detected binary content
    with workspace:// references.

    Uses content-based detection (data URIs, magic numbers) to find binary
    data regardless of field name. Deduplicates identical content within
    a single call using content hashing.

    Args:
        outputs: Block execution outputs (dict of output_name -> list of values)
        workspace_manager: WorkspaceManager instance with session scoping
        block_name: Name of the block (used in generated filenames)

    Returns:
        Processed outputs with binary data replaced by workspace references
    """
    cache: dict[str, str] = {}  # content_hash -> workspace_ref

    processed: dict[str, list[Any]] = {}
    for name, items in outputs.items():
        processed_items = []
        for item in items:
            processed_items.append(
                await _process_value(item, workspace_manager, block_name, cache)
            )
        processed[name] = processed_items
    return processed


async def _process_value(
    value: Any,
    wm: WorkspaceManager,
    block: str,
    cache: dict[str, str],
) -> Any:
    """Recursively process a value, detecting binary content in strings."""
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            result[k] = await _process_value(v, wm, block, cache)
        return result
    if isinstance(value, list):
        return [await _process_value(v, wm, block, cache) for v in value]
    if isinstance(value, str) and len(value) > SIZE_THRESHOLD:
        return await _try_detect_and_save(value, wm, block, cache)
    return value


async def _try_detect_and_save(
    value: str,
    wm: WorkspaceManager,
    block: str,
    cache: dict[str, str],
) -> str:
    """Attempt to detect binary content and save it. Returns original if not binary."""

    # Try data URI first (highest confidence - explicit mimetype)
    result = _detect_data_uri(value)
    if result:
        content, ext = result
        return await _save_binary(content, ext, wm, block, cache, value)

    # Try raw base64 with magic number detection
    result = _detect_raw_base64(value)
    if result:
        content, ext = result
        return await _save_binary(content, ext, wm, block, cache, value)

    return value  # Not binary, return unchanged


def _detect_data_uri(value: str) -> Optional[tuple[bytes, str]]:
    """
    Detect data URI with whitelisted mimetype.

    Returns (content, extension) or None.
    """
    match = DATA_URI_PATTERN.match(value)
    if not match:
        return None

    mimetype, b64_payload = match.groups()
    if mimetype not in ALLOWED_MIMETYPES:
        return None

    try:
        content = base64.b64decode(b64_payload, validate=True)
    except (ValueError, binascii.Error):
        return None

    ext = _mimetype_to_ext(mimetype)
    return content, ext


def _detect_raw_base64(value: str) -> Optional[tuple[bytes, str]]:
    """
    Detect raw base64 with magic number validation.

    Only processes strings that:
    1. Look like pure base64 (regex pre-filter)
    2. Successfully decode as base64
    3. Start with a known binary file magic number

    Returns (content, extension) or None.
    """
    # Pre-filter: must look like base64 (no spaces, punctuation, etc.)
    if not BASE64_PATTERN.match(value):
        return None

    try:
        content = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        return None

    # Check magic numbers
    for magic, ext in MAGIC_SIGNATURES:
        if content.startswith(magic):
            # Special case for WebP: RIFF container, verify "WEBP" at offset 8
            if magic == b"RIFF":
                if len(content) < 12 or content[8:12] != b"WEBP":
                    continue
            return content, ext

    return None  # No magic number match = not a recognized binary format


async def _save_binary(
    content: bytes,
    ext: str,
    wm: WorkspaceManager,
    block: str,
    cache: dict[str, str],
    original: str,
) -> str:
    """
    Save binary content to workspace with deduplication.

    Returns workspace://file-id reference, or original value on failure.
    """
    content_hash = hashlib.sha256(content).hexdigest()

    if content_hash in cache:
        return cache[content_hash]

    try:
        safe_block = sanitize_filename(block)[:20].lower()
        filename = f"{safe_block}_{ext}_{uuid.uuid4().hex[:12]}.{ext}"

        file = await wm.write_file(content, filename)
        ref = f"workspace://{file.id}"
        cache[content_hash] = ref
        return ref
    except Exception as e:
        logger.warning(f"Failed to save binary output: {e}")
        return original  # Graceful degradation


def _mimetype_to_ext(mimetype: str) -> str:
    """Convert mimetype to file extension."""
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/svg+xml": "svg",
        "application/pdf": "pdf",
        "application/octet-stream": "bin",
    }
    return mapping.get(mimetype, "bin")
