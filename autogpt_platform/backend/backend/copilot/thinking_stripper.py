"""Streaming tag stripper for model reasoning blocks.

Different LLMs wrap internal chain-of-thought in different XML-style tags
(Claude uses ``<thinking>``, Gemini uses ``<internal_reasoning>``, etc.).
When extended thinking is **not** enabled, these tags may appear as plain text
in the response stream and must be stripped before the content reaches the
user.

The :class:`ThinkingStripper` handles chunk-boundary splitting so it can be
plugged into any delta-based streaming pipeline.
"""

from __future__ import annotations

# Tag pairs to strip.  Each entry is (open_tag, close_tag).
_REASONING_TAG_PAIRS: list[tuple[str, str]] = [
    ("<thinking>", "</thinking>"),
    ("<internal_reasoning>", "</internal_reasoning>"),
]

# Longest opener — used to size the partial-tag buffer.
_MAX_OPEN_TAG_LEN = max(len(o) for o, _ in _REASONING_TAG_PAIRS)


class ThinkingStripper:
    """Strip reasoning blocks from a stream of text deltas.

    Handles multiple tag patterns (``<thinking>``, ``<internal_reasoning>``,
    etc.) so the same stripper works across Claude, Gemini, and other models.

    Buffers just enough characters to detect a tag that may be split
    across chunks; emits text immediately when no tag is in-flight.
    Robust to single chunks that open and close a block, multiple
    blocks per stream, and tags that straddle chunk boundaries.
    Handles nested same-type tags via a per-tag depth counter so that
    ``<thinking><thinking>inner</thinking>after</thinking>`` correctly
    strips both levels and does not leak ``after``.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._in_thinking: bool = False
        self._close_tag: str = ""  # closing tag for the currently open block
        self._open_tag: str = ""  # opening tag for the currently open block
        self._depth: int = 0  # nesting depth for the current tag type

    def _find_open_tag(self) -> tuple[int, str, str]:
        """Find the earliest opening tag in the buffer.

        Returns (position, open_tag, close_tag) or (-1, "", "") if none.
        """
        best_pos = -1
        best_open = ""
        best_close = ""
        for open_tag, close_tag in _REASONING_TAG_PAIRS:
            pos = self._buffer.find(open_tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
                best_open = open_tag
                best_close = close_tag
        return best_pos, best_open, best_close

    def process(self, chunk: str) -> str:
        """Feed a chunk and return the text that is safe to emit now."""
        self._buffer += chunk
        out: list[str] = []
        while self._buffer:
            if self._in_thinking:
                # Search for both the open and close tags to track nesting.
                open_pos = self._buffer.find(self._open_tag)
                close_pos = self._buffer.find(self._close_tag)
                if close_pos == -1:
                    # No closing tag yet.  Consume any complete nested open
                    # tags first so depth stays accurate even when open and
                    # close tags straddle a chunk boundary.
                    if open_pos != -1:
                        self._depth += 1
                        self._buffer = self._buffer[open_pos + len(self._open_tag) :]
                        continue
                    # No complete close or open tag — keep a tail that could
                    # be the start of either tag.
                    keep = max(len(self._open_tag), len(self._close_tag)) - 1
                    self._buffer = self._buffer[-keep:] if keep else ""
                    return "".join(out)
                if open_pos != -1 and open_pos < close_pos:
                    # A nested open tag appears before the close tag — increase
                    # depth and skip past the nested opener.
                    self._depth += 1
                    self._buffer = self._buffer[open_pos + len(self._open_tag) :]
                else:
                    # Close tag is next; decrease depth.
                    self._buffer = self._buffer[close_pos + len(self._close_tag) :]
                    self._depth -= 1
                    if self._depth == 0:
                        self._in_thinking = False
                        self._open_tag = ""
                        self._close_tag = ""
            else:
                start, open_tag, close_tag = self._find_open_tag()
                if start == -1:
                    # No opening tag; emit everything except a tail that
                    # could start a partial opener on the next chunk.
                    safe_end = len(self._buffer)
                    for keep in range(
                        min(_MAX_OPEN_TAG_LEN - 1, len(self._buffer)), 0, -1
                    ):
                        tail = self._buffer[-keep:]
                        if any(o[:keep] == tail for o, _ in _REASONING_TAG_PAIRS):
                            safe_end = len(self._buffer) - keep
                            break
                    out.append(self._buffer[:safe_end])
                    self._buffer = self._buffer[safe_end:]
                    return "".join(out)
                out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(open_tag) :]
                self._in_thinking = True
                self._open_tag = open_tag
                self._close_tag = close_tag
                self._depth = 1
        return "".join(out)

    def flush(self) -> str:
        """Return any remaining emittable text when the stream ends."""
        if self._in_thinking:
            # Unclosed thinking block — discard the buffered reasoning.
            self._buffer = ""
            return ""
        out = self._buffer
        self._buffer = ""
        return out
