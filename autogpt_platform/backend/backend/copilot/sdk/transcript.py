"""Re-export public API from shared ``backend.copilot.transcript``.

The canonical implementation now lives at ``backend.copilot.transcript``
so both the SDK and baseline paths can import without cross-package
dependencies.  Public symbols are re-exported here so existing ``from
.transcript import ...`` statements within the ``sdk`` package continue
to work without modification.
"""

from backend.copilot.transcript import (
    COMPACT_MSG_ID_PREFIX,
    ENTRY_TYPE_MESSAGE,
    STOP_REASON_END_TURN,
    STRIPPABLE_TYPES,
    TRANSCRIPT_STORAGE_PREFIX,
    TranscriptDownload,
    cleanup_stale_project_dirs,
    compact_transcript,
    delete_transcript,
    download_transcript,
    read_compacted_entries,
    strip_for_upload,
    strip_progress_entries,
    strip_stale_thinking_blocks,
    upload_transcript,
    validate_transcript,
    write_transcript_to_tempfile,
)

__all__ = [
    "COMPACT_MSG_ID_PREFIX",
    "ENTRY_TYPE_MESSAGE",
    "STOP_REASON_END_TURN",
    "STRIPPABLE_TYPES",
    "TRANSCRIPT_STORAGE_PREFIX",
    "TranscriptDownload",
    "cleanup_stale_project_dirs",
    "compact_transcript",
    "delete_transcript",
    "download_transcript",
    "read_compacted_entries",
    "strip_for_upload",
    "strip_progress_entries",
    "strip_stale_thinking_blocks",
    "upload_transcript",
    "validate_transcript",
    "write_transcript_to_tempfile",
]
