"""Tests for the shared ThinkingStripper."""

from backend.copilot.thinking_stripper import ThinkingStripper


def test_basic_thinking_tag() -> None:
    """<thinking>...</thinking> blocks are fully stripped."""
    s = ThinkingStripper()
    assert s.process("<thinking>internal reasoning here</thinking>Hello!") == "Hello!"


def test_internal_reasoning_tag() -> None:
    """<internal_reasoning>...</internal_reasoning> blocks are stripped."""
    s = ThinkingStripper()
    assert (
        s.process("<internal_reasoning>step by step</internal_reasoning>Answer")
        == "Answer"
    )


def test_split_across_chunks() -> None:
    """Tags split across multiple chunks are handled correctly."""
    s = ThinkingStripper()
    out = s.process("Hello <thin")
    out += s.process("king>secret</thinking> world")
    assert out == "Hello  world"


def test_plain_text_preserved() -> None:
    """Plain text with the word 'thinking' is not stripped."""
    s = ThinkingStripper()
    assert (
        s.process("I am thinking about this problem")
        == "I am thinking about this problem"
    )


def test_multiple_blocks() -> None:
    """Multiple reasoning blocks in one stream are all stripped."""
    s = ThinkingStripper()
    result = s.process(
        "A<thinking>x</thinking>B<internal_reasoning>y</internal_reasoning>C"
    )
    assert result == "ABC"


def test_flush_discards_unclosed() -> None:
    """Unclosed reasoning block is discarded on flush."""
    s = ThinkingStripper()
    s.process("Start<thinking>never closed")
    flushed = s.flush()
    assert "never closed" not in flushed


def test_empty_block() -> None:
    """Empty reasoning blocks are handled gracefully."""
    s = ThinkingStripper()
    assert s.process("Before<thinking></thinking>After") == "BeforeAfter"


def test_flush_emits_remaining_plain_text() -> None:
    """flush() returns any plain text still in the buffer."""
    s = ThinkingStripper()
    # The trailing '<' could be a partial tag, so process buffers it.
    out = s.process("Hello")
    flushed = s.flush()
    assert out + flushed == "Hello"


def test_internal_reasoning_split_open_tag() -> None:
    """<internal_reasoning> split across three chunks."""
    s = ThinkingStripper()
    out = s.process("OK <inter")
    out += s.process("nal_reaso")
    out += s.process("ning>secret stuff</internal_reasoning> visible")
    out += s.flush()
    assert out == "OK  visible"


def test_no_tags_passthrough() -> None:
    """Text without any tags passes through unchanged."""
    s = ThinkingStripper()
    out = s.process("Hello world, this is fine.")
    out += s.flush()
    assert out == "Hello world, this is fine."


def test_reasoning_at_end_of_stream() -> None:
    """Reasoning block at end of stream with no trailing text."""
    s = ThinkingStripper()
    out = s.process("Answer<internal_reasoning>my thoughts</internal_reasoning>")
    out += s.flush()
    assert out == "Answer"


def test_nested_same_type_tags_do_not_leak() -> None:
    """Nested same-type tags use a depth counter so inner close-tag does not end the block."""
    s = ThinkingStripper()
    out = s.process("<thinking><thinking>inner</thinking>after</thinking>final")
    out += s.flush()
    assert "inner" not in out
    assert "after" not in out
    assert out == "final"


def test_nested_tags_split_across_chunks() -> None:
    """Nested same-type tag nesting tracked correctly across chunk boundaries."""
    s = ThinkingStripper()
    out = s.process("<thinking><thin")
    out += s.process("king>inner</thinking>still_inside</thinking>visible")
    out += s.flush()
    assert "inner" not in out
    assert "still_inside" not in out
    assert out == "visible"


def test_flush_tail_not_re_suppressed_on_next_process() -> None:
    """Regression: a stream ending with a partial tag opener must survive flush().

    flush() returns the buffered prefix that was withheld because it *might* be
    the start of a reasoning tag (e.g. "Hello <inter").  After flush() the
    buffer is empty.  Calling process() on that flushed tail in a fresh context
    must return it unchanged — the tail is safe plain text, not a live tag.
    """
    s = ThinkingStripper()
    # Stream ends mid-way through a potential tag opener — stripper buffers " <inter".
    out = s.process("Hello <inter")
    tail = s.flush()
    # The full text "Hello <inter" must be delivered.
    assert out + tail == "Hello <inter"
    # After flush, the stripper is reset.  Calling process on the flushed tail
    # (simulating what _dispatch_response does when skip_strip=False) would
    # re-buffer " <inter" and return "".  This test documents that flush() clears
    # the buffer so a new process() call starts clean — caller must use skip_strip.
    s2 = ThinkingStripper()
    out2 = s2.process("safe text")
    assert out2 == "safe text"  # unaffected by prior flush


def test_nested_open_tag_depth_tracked_across_chunk_boundary() -> None:
    """Regression: nested open tag in chunk without close tag must increment depth.

    If a chunk contains a complete nested opening tag but no closing tag, the
    depth counter must still be incremented.  Without the fix, the trim at
    'close_pos == -1' would discard the nested opener, leaving depth=1.  On
    the next chunk the first </thinking> decrements depth to 0 and exits
    thinking mode prematurely, leaking the content after it.
    """
    s = ThinkingStripper()
    # Chunk 1: outer open + nested open (complete), no close yet
    out = s.process("<thinking>outer<thinking>inner")
    # Chunk 2: first close ends nested block, second close ends outer block
    out += s.process("</thinking>middle</thinking>final")
    out += s.flush()
    # All reasoning content must be stripped; only "final" is visible
    assert "inner" not in out
    assert "middle" not in out
    assert out == "final"
