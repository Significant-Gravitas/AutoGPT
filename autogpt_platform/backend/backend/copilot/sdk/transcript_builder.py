"""Re-export from shared ``backend.copilot.transcript_builder`` for backward compat.

The canonical implementation now lives at ``backend.copilot.transcript_builder``
so both the SDK and baseline paths can import without cross-package
dependencies.
"""

from backend.copilot.transcript_builder import TranscriptBuilder, TranscriptEntry

__all__ = ["TranscriptBuilder", "TranscriptEntry"]
