"""Save binary block outputs to workspace, return references instead of base64."""

import base64
import binascii
import hashlib
import logging
import uuid
from typing import Any

from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

BINARY_FIELDS = {"png", "jpeg", "pdf"}  # Base64 encoded
TEXT_FIELDS = {"svg"}  # Large text, save raw
SAVEABLE_FIELDS = BINARY_FIELDS | TEXT_FIELDS
SIZE_THRESHOLD = 1024  # Only process content > 1KB (string length, not decoded size)


async def process_binary_outputs(
    outputs: dict[str, list[Any]],
    workspace_manager: WorkspaceManager,
    block_name: str,
) -> dict[str, list[Any]]:
    """
    Replace binary data in block outputs with workspace:// references.

    Deduplicates identical content within a single call (e.g., same PDF
    appearing in both main_result and results).
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
    item: Any, wm: WorkspaceManager, block: str, cache: dict
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
    data: dict, wm: WorkspaceManager, block: str, cache: dict
) -> dict:
    """Process a dict, saving binary fields and recursing into nested structures."""
    result: dict[str, Any] = {}

    for key, value in data.items():
        if (
            key in SAVEABLE_FIELDS
            and isinstance(value, str)
            and len(value) > SIZE_THRESHOLD
        ):
            content_hash = hashlib.sha256(value.encode()).hexdigest()

            if content_hash in cache:
                result[key] = cache[content_hash]
            elif ref := await _save(value, key, wm, block):
                cache[content_hash] = ref
                result[key] = ref
            else:
                result[key] = value  # Save failed, keep original

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


async def _save(value: str, field: str, wm: WorkspaceManager, block: str) -> str | None:
    """Save content to workspace, return workspace:// reference or None on failure."""
    try:
        if field in BINARY_FIELDS:
            content = _decode_base64(value)
            if content is None:
                return None
        else:
            content = value.encode("utf-8")

        ext = {"jpeg": "jpg"}.get(field, field)
        filename = f"{block.lower().replace(' ', '_')[:20]}_{field}_{uuid.uuid4().hex[:12]}.{ext}"

        file = await wm.write_file(content=content, filename=filename)
        return f"workspace://{file.id}"

    except Exception as e:
        logger.error(f"Failed to save {field} to workspace for block '{block}': {e}")
        return None


def _decode_base64(value: str) -> bytes | None:
    """Decode base64, handling data URI format. Returns None on failure."""
    try:
        if value.startswith("data:"):
            value = value.split(",", 1)[1] if "," in value else value
        # Normalize padding and use strict validation to prevent corrupted data
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
