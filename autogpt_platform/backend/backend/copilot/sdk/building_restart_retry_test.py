"""A context-size retry after a building-mode restart keeps the guide.

The restart puts the agent-building guide in the system prompt and sets
``guide_in_system_prompt``. When the restarted launch then hits a context-size
error, the retry must send the prompt the restart installed on the fresh CLI
session it starts, and the turn summary must describe what that last launch
sent.
"""

import logging
from unittest.mock import AsyncMock

import pytest

from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.retry_scenarios_test import scenario_profile
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


async def _builder_suffix(_session, force=False):
    return _GUIDE if force else ""


@pytest.mark.asyncio
async def test_a_context_retry_after_the_restart_keeps_the_guide(tmp_path, caplog):
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
    # Each query's budget.
    builds: list[int | None] = []
    real_build = sdk_service._build_query_message

    async def _record_build(*args, **kwargs):
        builds.append(kwargs.get("target_tokens"))
        return await real_build(*args, **kwargs)

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
                    f"{_SVC}._rebuild_first_turn_context",
                    dict(new_callable=AsyncMock, return_value=[]),
                ),
                (f"{_SVC}._build_query_message", dict(new=_record_build)),
            ],
            config_overrides=_LIVE_PROMPT,
        )

    assert [
        (_GUIDE in _prompt_text(options), flag) for options, flag, _ in launches
    ] == [(False, False), (True, True), (True, True)]
    (_, _, _), (restart, _, _), (retry, _, retry_client) = launches
    assert _prompt_text(retry) == _prompt_text(restart)
    assert (retry.resume, retry.session_id) == (None, "test-session-id")
    # The retry rebuilds its history to the route's first retry budget.
    assert builds[-1] == scenario_profile().retry_target_tokens()[0]

    [summary] = _summaries(caplog)
    assert summary["outcome"] == "ok"
    assert (summary["cli_launches"], summary["context_rebuilds"]) == (3, 1)
    assert summary["building_guide_in_prompt"] is True
    assert summary["system_prompt_sha"] == short_hash(_prompt_text(retry))
    _assert_line_reads_the_query(
        summary, _sent_query(retry_client), "rebuilt_no_resume_history"
    )
