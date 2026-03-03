"""Shared constants for the CoPilot module."""

# Special message prefixes for text-based markers (parsed by frontend).
# The hex suffix makes accidental LLM generation of these strings virtually
# impossible, avoiding false-positive marker detection in normal conversation.
COPILOT_ERROR_PREFIX = "[__COPILOT_ERROR_f7a1__]"  # Renders as ErrorCard
COPILOT_SYSTEM_PREFIX = "[__COPILOT_SYSTEM_e3b0__]"  # Renders as system info message

# Compaction notice messages shown to users.
COMPACTION_DONE_MSG = "Earlier messages were summarized to fit within context limits."
COMPACTION_TOOL_NAME = "context_compaction"
