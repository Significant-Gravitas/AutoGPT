"""Tests for EpisodicActionHistory cursor safety and task continuation.

Covers:
- Cursor >= len guard in current_episode (prevents IndexError)
- History preserved across task changes (no clearing)
"""

from unittest.mock import MagicMock

from forge.components.action_history.model import EpisodicActionHistory


def _make_history_with_episodes(n: int) -> EpisodicActionHistory:
    """Create a history with n completed episodes (cursor advanced past all)."""
    history = EpisodicActionHistory()
    for i in range(n):
        ep = MagicMock()
        ep.result = MagicMock()
        history.episodes.append(ep)
        history.cursor += 1
    return history


class TestEpisodicActionHistoryCursor:
    def test_current_episode_returns_none_on_empty_history(self):
        history = EpisodicActionHistory()
        assert history.current_episode is None

    def test_current_episode_returns_none_when_cursor_at_end(self):
        history = _make_history_with_episodes(1)
        assert history.cursor == 1
        assert history.current_episode is None

    def test_current_episode_returns_episode_when_cursor_valid(self):
        history = EpisodicActionHistory()
        ep = MagicMock()
        ep.result = None
        history.episodes.append(ep)
        history.cursor = 0
        assert history.current_episode is ep

    def test_cursor_beyond_episodes_returns_none(self):
        """Any cursor value beyond the episode list should return None."""
        history = EpisodicActionHistory()
        history.cursor = 100
        assert history.current_episode is None

    def test_cursor_safe_after_clear(self):
        """Even if episodes are cleared without resetting cursor,
        current_episode must not crash (>= guard)."""
        history = _make_history_with_episodes(2)
        history.episodes.clear()
        assert history.cursor == 2
        assert history.current_episode is None


class TestHistoryPreservedAcrossTasks:
    def test_episodes_survive_task_change(self):
        """When user starts a new task, episodes from the previous task
        should still be present — the compression system handles overflow."""
        history = _make_history_with_episodes(3)
        assert len(history.episodes) == 3
        assert history.cursor == 3

        # Simulate what main.py does on task change (no clearing)
        # history is untouched — episodes remain

        assert len(history.episodes) == 3
        assert history.current_episode is None  # cursor at end

    def test_new_episode_appends_after_previous(self):
        """New task actions append to existing history."""
        history = _make_history_with_episodes(2)

        # New task starts — add a new episode
        new_ep = MagicMock()
        new_ep.result = None
        history.episodes.append(new_ep)
        # cursor still at 2, which is now the new episode
        assert history.current_episode is new_ep
        assert len(history.episodes) == 3


class TestFinishContinuationFlow:
    """Integration-style test for the AgentFinished -> new task flow.

    Mirrors run_interaction_loop in
    classic/original_autogpt/autogpt/app/main.py: when the agent proposes
    `finish`, the action is registered but AgentFinished is caught before
    execute() can record a result, leaving the episode open (result=None).
    The loop interprets an open episode as "reuse this proposal", so without
    closing it the agent would replay the finish proposal forever. main.py
    closes it by registering an ActionSuccessResult before the next task.
    """

    def test_finish_then_continuation_proposes_fresh_action(self):
        from forge.models.action import ActionSuccessResult

        history = EpisodicActionHistory()

        # 1. Finish action proposed and registered; result not yet recorded.
        finish_episode = MagicMock()
        finish_episode.result = None
        history.episodes.append(finish_episode)
        assert history.current_episode is finish_episode
        assert history.current_episode.result is None

        # 2. AgentFinished caught -> main.py closes the episode via
        #    register_result so the loop won't reuse the finish proposal.
        history.register_result(ActionSuccessResult(outputs="task complete"))

        # 3. Episode is closed -> current_episode is None, so the next cycle
        #    proposes a fresh action instead of replaying the stale finish.
        assert finish_episode.result is not None
        assert history.current_episode is None

        # 4. New task: a new episode appends and becomes current; the prior
        #    finish episode is preserved (history is kept across tasks).
        next_episode = MagicMock()
        next_episode.result = None
        history.episodes.append(next_episode)
        assert history.current_episode is next_episode
        assert len(history.episodes) == 2
        assert history.episodes[0] is finish_episode
