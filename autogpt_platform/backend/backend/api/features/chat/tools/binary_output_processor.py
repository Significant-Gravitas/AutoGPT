"""
Save binary block outputs to workspace, return references instead of base64.

This module post-processes block execution outputs to detect and save binary
content (from code execution results) to the workspace, returning workspace://
references instead of raw base64 data. This reduces LLM output token usage
by ~97% for file generation tasks.

Detection is field-name based, targeting the standard e2b CodeExecutionResult
fields: png, jpeg, pdf, svg. Other image-producing blocks already use
store_media_file() and don't need this post-processing.
"""

import base64
import binascii
import hashlib
import logging
import uuid
from typing import Any

from backend.util.file import sanitize_filename
from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# Field names that contain binary data (base64 encoded)
BINARY_FIELDS = {"png", "jpeg", "pdf"}

# Field names that contain large text data (not base64, save as-is)
TEXT_FIELDS = {"svg"}

# Combined set for quick lookup
SAVEABLE_FIELDS = BINARY_FIELDS | TEXT_FIELDS

# Only process content larger than this (string length, not decoded size)
SIZE_THRESHOLD = 1024  # 1KB


async def process_binary_outputs(
    outputs: dict[str, list[Any]],
    workspace_manager: WorkspaceManager,
    block_name: str,
) -> dict[str, list[Any]]:
    """
    Replace binary data in block outputs with workspace:// references.

    Scans outputs for known binary field names (png, jpeg, pdf, svg) and saves
    large content to the workspace. Returns processed outputs with base64 data
    replaced by workspace:// references.

    Deduplicates identical content within a single call using content hashing.

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
        processed_items: list[Any] = []
        for item in items:
            processed_items.append(
                await _process_item(item, workspace_manager, block_name, cache)
            )
        processed[name] = processed_items
    return processed


async def _process_item(
    item: Any,
    wm: WorkspaceManager,
    block: str,
    cache: dict[str, str],
) -> Any:
    """Recursively process an item, handling dicts and lists."""
    if isinstance(item, dict):
        return await _process_dict(item, wm, block, cache)
    if isinstance(item, list):
        processed: list[Any] = []
        for i in item:
            processed.append(await _process_item(i, wm, block, cache))
        return processed
    return item


async def _process_dict(
    data: dict[str, Any],
    wm: WorkspaceManager,
    block: str,
    cache: dict[str, str],
) -> dict[str, Any]:
    """Process a dict, saving binary fields and recursing into nested structures."""
    result: dict[str, Any] = {}

    for key, value in data.items():
        if (
            key in SAVEABLE_FIELDS
            and isinstance(value, str)
            and len(value) > SIZE_THRESHOLD
        ):
            # Determine content bytes based on field type
            if key in BINARY_FIELDS:
                content = _decode_base64(value)
                if content is None:
                    # Decode failed, keep original value
                    result[key] = value
                    continue
            else:
                # TEXT_FIELDS: encode as UTF-8
                content = value.encode("utf-8")

            # Hash decoded content for deduplication
            content_hash = hashlib.sha256(content).hexdigest()

            if content_hash in cache:
                # Reuse existing workspace reference
                result[key] = cache[content_hash]
            elif ref := await _save_content(content, key, wm, block):
                # Save succeeded, cache and use reference
                cache[content_hash] = ref
                result[key] = ref
            else:
                # Save failed, keep original value
                result[key] = value

        elif isinstance(value, dict):
            result[key] = await _process_dict(value, wm, block, cache)
        elif isinstance(value, list):
            processed: list[Any] = []
            for i in value:
                processed.append(await _process_item(i, wm, block, cache))
            result[key] = processed
        else:
            result[key] = value

    return result


async def _save_content(
    content: bytes,
    field: str,
    wm: WorkspaceManager,
    block: str,
) -> str | None:
    """
    Save content to workspace, return workspace:// reference.

    Args:
        content: Decoded binary content to save
        field: Field name (used for extension)
        wm: WorkspaceManager instance
        block: Block name (used in filename)

    Returns:
        workspace://file-id reference, or None if save failed
    """
    try:
        # Map field name to file extension
        ext = {"jpeg": "jpg"}.get(field, field)

        # Sanitize block name for safe filename
        safe_block = sanitize_filename(block.lower())[:20]
        filename = f"{safe_block}_{field}_{uuid.uuid4().hex[:12]}.{ext}"

        file = await wm.write_file(content=content, filename=filename)
        return f"workspace://{file.id}"

    except Exception as e:
        logger.error(f"Failed to save {field} to workspace for block '{block}': {e}")
        return None


def _decode_base64(value: str) -> bytes | None:
    """
    Decode base64 string, handling both raw base64 and data URI formats.

    Args:
        value: Base64 string or data URI (data:<mime>;base64,<payload>)

    Returns:
        Decoded bytes, or None if decoding failed
    """
    try:
        # Handle data URI format
        if value.startswith("data:"):
            if "," in value:
                value = value.split(",", 1)[1]
            else:
                # Malformed data URI, no comma separator
                return None

        # Normalize padding (handle missing = chars)
        padded = value + "=" * (-len(value) % 4)

        # Strict validation to prevent corrupted data
        return base64.b64decode(padded, validate=True)

    except (binascii.Error, ValueError):
        return None
