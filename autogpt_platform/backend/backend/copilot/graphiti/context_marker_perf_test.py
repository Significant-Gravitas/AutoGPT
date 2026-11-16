import time

from .context_marker import (
    INJECTED_MEMORY_BLOCK_RE,
    INJECTED_MEMORY_MARKER,
    mark_injected_memory_block,
)


def test_many_unclosed_marked_tags_are_scrubbed_promptly() -> None:
    text = (f"<temporal_context {INJECTED_MEMORY_MARKER}") * 1000

    started = time.perf_counter()
    scrubbed = INJECTED_MEMORY_BLOCK_RE.sub("", text)
    elapsed = time.perf_counter() - started

    assert scrubbed == text
    assert elapsed < 1.0


def test_scrub_preserves_malformed_user_tags_before_the_real_block() -> None:
    user_text = "<temporal_context " * 20
    block = mark_injected_memory_block(
        '<temporal_context version="2" >fresh memory</temporal_context>'
    )
    assert block is not None

    assert INJECTED_MEMORY_BLOCK_RE.sub("", f"{user_text}\n\n{block}") == user_text
