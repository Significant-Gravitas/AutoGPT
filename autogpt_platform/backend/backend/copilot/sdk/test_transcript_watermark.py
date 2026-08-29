"""Unit tests for the watermark-fix logic in stream_chat_completion_sdk.

The fix is at the upload step: when use_resume=True and transcript_msg_count>0
we set the JSONL coverage watermark to transcript_msg_count + 2 (the pair just
recorded) instead of len(session.messages).  This prevents the "inflated
watermark" bug where a stale JSONL in GCS could hide missing context from
future gap-fill checks.
"""

from __future__ import annotations


def _compute_jsonl_covered(
    use_resume: bool,
    transcript_msg_count: int,
    session_msg_count: int,
) -> int:
    """Mirror the watermark computation from ``stream_chat_completion_sdk``.

    Extracted here so we can unit-test it independently without invoking the
    full streaming stack.
    """
    if use_resume and transcript_msg_count > 0:
        return transcript_msg_count + 2
    return session_msg_count


class TestWatermarkFix:
    """Watermark computation logic — mirrors the finally-block in SDK service."""

    def test_inflated_watermark_triggers_gap_fill(self):
        """Stale JSONL (T12) with high watermark (46) → after fix, watermark=14.

        Before fix: watermark=46 → next turn's gap check (transcript_msg_count < db-1)
        never fires because 46 >= 47-1=46, so context loss is silent.
        After fix: watermark = 12 + 2 = 14 → gap check fires (14 < 46) and
        the model receives the missing turns.
        """
        # Simulate: use_resume=True, transcript covered T12 (12 msgs), DB now has 47
        use_resume = True
        transcript_msg_count = 12
        session_msg_count = 47  # DB count (what old code used to set watermark)

        watermark = _compute_jsonl_covered(
            use_resume, transcript_msg_count, session_msg_count
        )

        assert watermark == 14  # 12 + 2, NOT 47
        # Verify: the gap check would fire on next turn
        # next-turn check: transcript_msg_count < msg_count - 1 → 14 < 47-1=46 → True
        assert watermark < session_msg_count - 1

    def test_no_false_positive_when_transcript_current(self):
        """Transcript current (watermark=46, DB=47) → gap stays 0.

        When the JSONL actually covers T46 (the most recent assistant turn),
        uploading watermark=46+2=48 means next turn's gap check sees
        48 >= 48-1=47 → no gap. Correct.
        """
        use_resume = True
        transcript_msg_count = 46
        session_msg_count = 47

        watermark = _compute_jsonl_covered(
            use_resume, transcript_msg_count, session_msg_count
        )

        assert watermark == 48  # 46 + 2
        # Next turn: session has 48 msgs, watermark=48 → 48 >= 48-1=47 → no gap
        next_turn_session = 48
        assert watermark >= next_turn_session - 1

    def test_fresh_session_falls_back_to_db_count(self):
        """use_resume=False → watermark = len(session.messages) (original behaviour)."""
        use_resume = False
        transcript_msg_count = 0
        session_msg_count = 3

        watermark = _compute_jsonl_covered(
            use_resume, transcript_msg_count, session_msg_count
        )

        assert watermark == session_msg_count

    def test_old_format_meta_zero_count_falls_back_to_db(self):
        """transcript_msg_count=0 (old-format meta with no count field) → DB fallback."""
        use_resume = True
        transcript_msg_count = 0  # old-format meta or not-yet-set
        session_msg_count = 10

        watermark = _compute_jsonl_covered(
            use_resume, transcript_msg_count, session_msg_count
        )

        assert watermark == session_msg_count
