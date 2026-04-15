"""Test for cursor reset bug when clearing episode history between tasks.

Reproduces: IndexError in EpisodicActionHistory.current_episode when
episodes.clear() is called without resetting cursor to 0.

This is the exact crash from run_interaction_loop when the user starts a
second task after finishing the first one.
"""

from unittest.mock import MagicMock

from forge.components.action_history.model import EpisodicActionHistory


def _make_history_with_episodes(n: int) -> EpisodicActionHistory:
    """Create a history with n completed episodes (cursor advanced past all)."""
    history = EpisodicActionHistory()
    for i in range(n):
        # Directly append mock episodes and advance cursor,
        # simulating what register_action + register_result does
        ep = MagicMock()
        ep.result = MagicMock()  # has a result = completed
        history.episodes.append(ep)
        history.cursor += 1
    return history


class TestEpisodicActionHistoryCursorReset:
    def test_current_episode_after_clear_without_cursor_reset_crashes(self):
        """REPRODUCER: This is the exact bug.

        After completing a task, the interaction loop clears episodes but
        doesn't reset cursor. On the next task, current_episode does
        `self[self.cursor]` where cursor > len(episodes) -> IndexError.
        """
        history = _make_history_with_episodes(2)
        assert history.cursor == 2
        assert len(history.episodes) == 2

        # This is what main.py line 759 does between tasks:
        history.episodes.clear()

        # cursor is still 2, but episodes is empty
        assert history.cursor == 2
        assert len(history.episodes) == 0

        # This is what main.py line 687 calls at the start of the next task.
        # BUG: cursor (2) != len(episodes) (0), so it falls through to
        # self.episodes[2] on an empty list -> IndexError
        #
        # After the fix, this should return None (no current episode).
        result = history.current_episode
        assert result is None

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
        ep.result = None  # not yet completed
        history.episodes.append(ep)
        history.cursor = 0
        assert history.current_episode is ep

    def test_clear_and_reset_allows_new_task(self):
        """After properly clearing episodes AND resetting cursor,
        the history should work correctly for a new task."""
        history = _make_history_with_episodes(3)

        # Clean reset between tasks
        history.episodes.clear()
        history.cursor = 0

        assert history.current_episode is None
        assert len(history) == 0

    def test_cursor_beyond_episodes_returns_none(self):
        """Any cursor value beyond the episode list should return None,
        not raise IndexError."""
        history = EpisodicActionHistory()
        history.cursor = 100  # way past empty list
        assert history.current_episode is None
