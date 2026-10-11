"""A context-size retry after a building-mode restart keeps the guide.

The restart puts the agent-building guide in the system prompt and sets
``guide_in_system_prompt``. When the restarted launch then hits a context-size
error, the retry must send the prompt the restart installed, whether it keeps
``--resume`` or starts a fresh CLI session, and the turn summary must describe
what that last launch sent.
"""

import logging
from unittest.mock import AsyncMock

import pytest

from backend.copilot.sdk.service import ReducedContext
from backend.copilot.sdk.turn_summary_model import short_hash
from backend.copilot.sdk.turn_summary_wiring_test import (
    _GUIDE,
    _LIVE_PROMPT,
    _SUMMARY_LOGGER,
    _SVC,
    _answer,
    _assert_line_reads_the_query,
    _client,
    _launch_patches,
    _prompt_text,
    _run_turn,
    _sent_query,
    _session,
    _summaries,
)
from backend.copilot.transcript_builder import TranscriptBuilder


async def _builder_suffix(_session, force=False):
    return _GUIDE if force else ""


def _recovery(kind: str) -> list:
    """How the context-size retry recovers: it keeps ``--resume`` on the CLI
    session, or drops the transcript and starts a fresh one."""
    if kind == "fresh":
        return []
    resumed = ReducedContext(TranscriptBuilder(), True, "test-session-id", False, True)
    return [(f"{_SVC}._reduce_context", dict(new=AsyncMock(return_value=resumed)))]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("recovery", "retry_path"),
    [("resumed", "rebuilt_resume"), ("fresh", "rebuilt_no_resume_history")],
)
async def test_a_context_retry_after_the_restart_keeps_the_guide(
    recovery, retry_path, tmp_path, caplog
):
    session = _session()
    session.building_mode_requested = True
    scripts = [
        lambda: _client(_answer()[:1]),
        lambda: _client(
            [], query_error=Exception("prompt is too long (context_length_exceeded)")
        ),
        lambda: _client(_answer()),
    ]
    # Per launch: its options, the guide flag when it launched, its client.
    launches = []

    def factory(*args, **kwargs):
        client = scripts[len(launches)]()
        launches.append((kwargs["options"], session.guide_in_system_prompt, client))
        return client

    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(
            tmp_path,
            factory,
            session=session,
            extra_patches=_launch_patches()
            + [
                (
                    f"{_SVC}.build_builder_system_prompt_suffix",
                    dict(side_effect=_builder_suffix),
                ),
                (
                    f"{_SVC}._rebuild_first_turn_prefix",
                    dict(new_callable=AsyncMock, return_value=""),
                ),
                *_recovery(recovery),
            ],
            config_overrides=_LIVE_PROMPT,
        )

    assert [
        (_GUIDE in _prompt_text(options), flag) for options, flag, _ in launches
    ] == [(False, False), (True, True), (True, True)]
    (_, _, _), (restart, _, _), (retry, _, retry_client) = launches
    assert _prompt_text(retry) == _prompt_text(restart)
    if recovery == "resumed":
        assert (retry.resume, retry.session_id) == ("test-session-id", None)
    else:
        assert (retry.resume, retry.session_id) == (None, "test-session-id")

    [summary] = _summaries(caplog)
    assert summary["outcome"] == "ok"
    assert (summary["cli_launches"], summary["context_rebuilds"]) == (3, 1)
    assert summary["building_guide_in_prompt"] is True
    assert summary["system_prompt_sha"] == short_hash(_prompt_text(retry))
    _assert_line_reads_the_query(summary, _sent_query(retry_client), retry_path)
