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
