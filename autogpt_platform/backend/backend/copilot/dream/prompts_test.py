"""Prompt builder smoke tests.

These ensure the three-phase prompt builders produce well-formed
``[system, user]`` message lists with the right per-phase content
shape — no LLM call.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .fetch import DreamInput, EpisodeRow, FactRow, SessionRow
from .prompts import (
    MAX_DEMOTIONS_PER_PASS,
    MAX_PROPOSALS_PER_PASS,
    MAX_WRITES_PER_PASS,
    build_consolidate_prompt,
    build_recombine_prompt,
    build_sanitize_prompt,
)


def _build_bundle() -> DreamInput:
    return DreamInput(
        user_id="u-1",
        group_id="g-1",
        window_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 14, tzinfo=timezone.utc),
        episodes=[
            EpisodeRow(
                uuid="ep-1",
                name="conversation",
                content="User: I love Belgian chocolate",
                source_description="msg",
                valid_at="2026-05-10",
                created_at="2026-05-10",
            ),
        ],
        facts=[
            FactRow(
                uuid="f-1",
                source="Nick",
                target="Belgian chocolate",
                name="loves",
                fact="Nick loves Belgian chocolate",
                scope="real:global",
                confidence=0.9,
                status="active",
                created_at="2026-05-10",
            ),
            FactRow(
                uuid="f-2",
                source="Nick",
                target="ProjectX",
                name="works_on",
                fact="Nick works on ProjectX",
                scope="project:x",
                confidence=0.8,
                status="active",
                created_at="2026-05-10",
            ),
        ],
        recent_sessions=[
            SessionRow(
                session_id="s-1",
                title="Chat about food",
                created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
                body="user: I love Belgian chocolate\nassistant: noted",
            ),
        ],
        known_fact_uuids={"f-1", "f-2"},
        known_episode_uuids={"ep-1"},
    )


def test_consolidate_prompt_has_system_and_user_messages():
    msgs = build_consolidate_prompt(_build_bundle())
    assert [m["role"] for m in msgs] == ["system", "user"]
    user_body = msgs[1]["content"]
    # The input bundle's episode body, facts, and session content should all
    # appear so the model has everything to consolidate from.
    assert "ep-1" in user_body
    assert "f-1" in user_body
    assert "Belgian chocolate" in user_body
    # Per-scope grouping in the facts section
    assert "[scope=real:global]" in user_body
    assert "[scope=project:x]" in user_body


def test_recombine_prompt_includes_consolidated_output_and_active_facts():
    bundle = _build_bundle()
    consolidated_json = '{"facts": [{"content": "consolidated"}]}'
    msgs = build_recombine_prompt(bundle, consolidated_json)
    assert msgs[0]["role"] == "system"
    assert "recombination" in msgs[0]["content"].lower()
    assert consolidated_json in msgs[1]["content"]
    assert "f-1" in msgs[1]["content"]


def test_sanitize_prompt_includes_known_fact_uuids_for_demotion_guard():
    """Sanitizer prompt MUST contain the allowlist so the sanitizer can
    reject invented uuids in demotions. Regression guard: if this drops
    out of the prompt, the runaway-demotion bug per p0-spec §3 comes back."""
    bundle = _build_bundle()
    msgs = build_sanitize_prompt(bundle, '{"facts": []}', '{"proposals": []}')
    assert msgs[0]["role"] == "system"
    user_body = msgs[1]["content"]
    # Both known fact uuids surfaced as the demotion allowlist
    assert "f-1" in user_body
    assert "f-2" in user_body
    # Hard caps mentioned in the system prompt (so the model can self-trim)
    sys = msgs[0]["content"]
    assert str(MAX_DEMOTIONS_PER_PASS) in sys
    assert str(MAX_PROPOSALS_PER_PASS) in sys
    assert str(MAX_WRITES_PER_PASS) in sys


def test_prompts_tolerate_empty_inputs():
    """A bundle with no episodes / facts / sessions still produces valid
    prompts — the model should answer with an empty list."""
    bundle = DreamInput(
        user_id="u",
        group_id="g",
        window_start=datetime.now(timezone.utc),
        window_end=datetime.now(timezone.utc),
    )
    consolidate_msgs = build_consolidate_prompt(bundle)
    recombine_msgs = build_recombine_prompt(bundle, "{}")
    sanitize_msgs = build_sanitize_prompt(bundle, "{}", "{}")
    assert len(consolidate_msgs) == 2
    assert len(recombine_msgs) == 2
    assert len(sanitize_msgs) == 2
    assert "(no recent episodes in window)" in consolidate_msgs[1]["content"]
    assert "(no active facts)" in consolidate_msgs[1]["content"]
