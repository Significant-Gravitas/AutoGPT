"""The "pai" copilot engine: the copilot turn on a Pydantic AI agent core.

A third engine beside the Claude Agent SDK and the baseline OpenAI-compat
loop, off by default (see :mod:`backend.copilot.pai.selection`). It reuses the
tool registry, the auto-mode gate, the stream wire format, the pending-message
buffer, the chat rows and the usage/cost recorders the other two engines use;
only the agent loop is Pydantic AI's.

Kept import-light: the processor imports
``backend.copilot.pai.service.stream_chat_completion_pai`` lazily, only on a
turn that selected this engine, so ``pydantic_ai`` never loads otherwise.
"""
