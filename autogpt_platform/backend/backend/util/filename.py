import re
import uuid

# Maximum filename length (conservative limit for most filesystems)
MAX_FILENAME_LENGTH = 200
# Filesystems cap a name at 255 *bytes* (NAME_MAX), not characters: 200 CJK or
# emoji characters can be 600-800 UTF-8 bytes. Leave headroom under 255.
MAX_FILENAME_BYTES = 240
# A longer "extension" is most likely just text after a dot.
_MAX_EXTENSION_LENGTH = 20


def sanitize_filename(filename: str) -> str:
    """
    Sanitize and truncate filename to prevent filesystem errors.

    The result is at most ``MAX_FILENAME_LENGTH`` characters and
    ``MAX_FILENAME_BYTES`` UTF-8 bytes, keeping a short extension when present.
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", filename)

    # Truncate if too long
    if not _fits(sanitized):
        name, dot, ext = sanitized.rpartition(".")
        # Keep the extension if possible, but only if it's reasonable length
        if dot and name and len(ext) <= _MAX_EXTENSION_LENGTH:
            suffix = "." + ext
            max_name_chars = MAX_FILENAME_LENGTH - len(suffix)
            max_name_bytes = MAX_FILENAME_BYTES - len(suffix.encode("utf-8"))
            sanitized = _truncate_utf8(name[:max_name_chars], max_name_bytes) + suffix
        else:
            sanitized = _truncate_utf8(
                sanitized[:MAX_FILENAME_LENGTH], MAX_FILENAME_BYTES
            )

    # Ensure it's not empty or just dots
    if not sanitized or sanitized.strip(".") == "":
        sanitized = f"file_{uuid.uuid4().hex[:8]}"

    return sanitized


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Cut ``text`` to at most ``max_bytes`` UTF-8 bytes on a character boundary."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def _fits(text: str) -> bool:
    return (
        len(text) <= MAX_FILENAME_LENGTH
        and len(text.encode("utf-8")) <= MAX_FILENAME_BYTES
    )
