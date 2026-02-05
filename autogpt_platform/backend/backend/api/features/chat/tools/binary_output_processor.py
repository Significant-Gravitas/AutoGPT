"""
Detect and save embedded binary data in block outputs.

Scans stdout_logs and other string outputs for embedded base64 patterns,
saves detected binary content to workspace, and replaces the base64 with
workspace:// references. This reduces LLM output token usage by ~97% for
file generation tasks.

Primary use case: ExecuteCodeBlock prints base64 to stdout, which appears
in stdout_logs. Without this processor, the LLM would re-type the entire
base64 string when saving files.
"""

import base64
import binascii
import hashlib
import logging
import re
import uuid
from typing import Any, Optional

from backend.util.file import sanitize_filename
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# Minimum decoded size to process (filters out small base64 strings)
MIN_DECODED_SIZE = 1024  # 1KB

# Pattern to find base64 chunks in text (at least 100 chars to be worth checking)
# Matches continuous base64 characters (with optional whitespace for line wrapping),
# optionally ending with = padding
EMBEDDED_BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/\s]{100,}={0,2}")

# Magic numbers for binary file detection
MAGIC_SIGNATURES = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"%PDF-", "pdf"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"RIFF", "webp"),  # Also check content[8:12] == b'WEBP'
]


async def process_binary_outputs(
    outputs: dict[str, list[Any]],
    workspace_manager: WorkspaceManager,
    block_name: str,
) -> dict[str, list[Any]]:
    """
    Scan all string values in outputs for embedded base64 binary content.
    Save detected binaries to workspace and replace with references.

    Args:
        outputs: Block execution outputs (dict of output_name -> list of values)
        workspace_manager: WorkspaceManager instance with session scoping
        block_name: Name of the block (used in generated filenames)

    Returns:
        Processed outputs with embedded base64 replaced by workspace references
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
    """Recursively process a value, detecting embedded base64 in strings."""
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            result[k] = await _process_value(v, wm, block, cache)
        return result
    if isinstance(value, list):
        return [await _process_value(v, wm, block, cache) for v in value]
    if isinstance(value, str) and len(value) > MIN_DECODED_SIZE:
        return await _extract_and_replace_base64(value, wm, block, cache)
    return value


async def _extract_and_replace_base64(
    text: str,
    wm: WorkspaceManager,
    block: str,
    cache: dict[str, str],
) -> str:
    """
    Find embedded base64 in text, save binaries, replace with references.

    Scans for base64 patterns, validates each as binary via magic numbers,
    saves valid binaries to workspace, and replaces the base64 portion
    (plus any surrounding markers) with the workspace reference.
    """
    result = text
    offset = 0

    for match in EMBEDDED_BASE64_PATTERN.finditer(text):
        b64_str = match.group(0)

        # Try to decode and validate
        detection = _decode_and_validate(b64_str)
        if detection is None:
            continue

        content, ext = detection

        # Save to workspace
        ref = await _save_binary(content, ext, wm, block, cache)
        if ref is None:
            continue

        # Calculate replacement bounds (include surrounding markers if present)
        start, end = match.start(), match.end()
        start, end = _expand_to_markers(text, start, end)

        # Apply replacement with offset adjustment
        adj_start = start + offset
        adj_end = end + offset
        result = result[:adj_start] + ref + result[adj_end:]
        offset += len(ref) - (end - start)

    return result


def _decode_and_validate(b64_str: str) -> Optional[tuple[bytes, str]]:
    """
    Decode base64 and validate it's a known binary format.

    Tries multiple 4-byte aligned offsets to handle cases where marker text
    (e.g., "START" from "PDF_BASE64_START") bleeds into the regex match.
    Base64 works in 4-char chunks, so we only check aligned offsets.

    Returns (content, extension) if valid binary, None otherwise.
    """
    # Strip whitespace for RFC 2045 line-wrapped base64
    normalized = re.sub(r"\s+", "", b64_str)

    # Try offsets 0, 4, 8, ... up to 32 chars (handles markers up to ~24 chars)
    # This handles cases like "STARTJVBERi0..." where "START" bleeds into match
    for char_offset in range(0, min(33, len(normalized)), 4):
        candidate = normalized[char_offset:]

        try:
            content = base64.b64decode(candidate, validate=True)
        except (ValueError, binascii.Error):
            continue

        # Must meet minimum size
        if len(content) < MIN_DECODED_SIZE:
            continue

        # Check magic numbers
        for magic, ext in MAGIC_SIGNATURES:
            if content.startswith(magic):
                # Special case for WebP: RIFF container, verify "WEBP" at offset 8
                if magic == b"RIFF":
                    if len(content) < 12 or content[8:12] != b"WEBP":
                        continue
                return content, ext

    return None


def _expand_to_markers(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Expand replacement bounds to include surrounding markers if present.

    Handles patterns like:
    - ---BASE64_START---\\n{base64}\\n---BASE64_END---
    - [BASE64]{base64}[/BASE64]
    - Or just the raw base64
    """
    # Common marker patterns to strip (order matters - check longer patterns first)
    start_markers = [
        "PDF_BASE64_START",
        "---BASE64_START---\n",
        "---BASE64_START---",
        "[BASE64]\n",
        "[BASE64]",
    ]
    end_markers = [
        "PDF_BASE64_END",
        "\n---BASE64_END---",
        "---BASE64_END---",
        "\n[/BASE64]",
        "[/BASE64]",
    ]

    # Check for start markers
    for marker in start_markers:
        marker_start = start - len(marker)
        if marker_start >= 0 and text[marker_start:start] == marker:
            start = marker_start
            break

    # Check for end markers
    for marker in end_markers:
        marker_end = end + len(marker)
        if marker_end <= len(text) and text[end:marker_end] == marker:
            end = marker_end
            break

    return start, end


async def _save_binary(
    content: bytes,
    ext: str,
    wm: WorkspaceManager,
    block: str,
    cache: dict[str, str],
) -> Optional[str]:
    """
    Save binary content to workspace with deduplication.

    Returns workspace://file-id reference, or None on failure.
    """
    content_hash = hashlib.sha256(content).hexdigest()

    if content_hash in cache:
        return cache[content_hash]

    try:
        safe_block = sanitize_filename(block)[:20].lower()
        filename = f"{safe_block}_{uuid.uuid4().hex[:12]}.{ext}"

        # Scan for viruses before saving
        await scan_content_safe(content, filename=filename)

        file = await wm.write_file(content, filename)
        ref = f"workspace://{file.id}"
        cache[content_hash] = ref
        return ref
    except Exception as e:
        logger.warning("Failed to save binary output: %s", e)
        return None
