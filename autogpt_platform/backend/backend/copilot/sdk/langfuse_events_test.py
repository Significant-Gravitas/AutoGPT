import logging

from backend.copilot.sdk.compaction import CompactionStats
from backend.copilot.sdk.langfuse_events import emit_compaction_event


def test_compaction_measurement_is_logged_locally(caplog):
    stats = CompactionStats(
        tokens_before=200000, tokens_after=95000, messages_before=90, messages_after=40
    )
    with caplog.at_level(logging.INFO):
        emit_compaction_event(path="sdk_internal", stats=stats, after_source="file")
    assert len(caplog.records) == 1
    assert caplog.records[0].args[-1]["tokensAfter"] == 95000
    assert caplog.records[0].args[-1]["after_source"] == "file"


def test_missing_compaction_counts_remain_observable(caplog):
    with caplog.at_level(logging.INFO):
        emit_compaction_event(path="sdk_internal", stats=None)
    assert caplog.records[0].args[-1]["path"] == "sdk_internal"
