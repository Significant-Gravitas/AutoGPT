"""Test that permission denials are registered as feedback, not raw errors.

Bug: When a command is denied by the permission system, the agent received
a bare ActionErrorResult with no feedback in its action history. This meant
the agent had no memory of being denied and would propose the exact same
command again, creating an infinite loop.

Fix: Use do_not_execute() instead of returning ActionErrorResult, so the
denial is registered as ActionInterruptedByHuman with feedback text that
the agent sees in its next prompt.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from autogpt.agents.agent import Agent

from forge.models.action import ActionInterruptedByHuman


@pytest.fixture
def mock_agent():
    """Create a minimal mock Agent with permission manager."""
    agent = MagicMock(spec=Agent)
    agent.permission_manager = MagicMock()
    agent.event_history = MagicMock()

    # Wire up do_not_execute to return ActionInterruptedByHuman
    async def fake_do_not_execute(proposal, feedback):
        result = ActionInterruptedByHuman(feedback=feedback)
        return result

    agent.do_not_execute = AsyncMock(side_effect=fake_do_not_execute)
    return agent


class TestPermissionDenialFeedback:
    @pytest.mark.asyncio
    async def test_denied_command_calls_do_not_execute(self, mock_agent):
        """When permission is denied, execute() must call do_not_execute()
        so the denial is registered in action history as feedback.

        Previously this returned a bare ActionErrorResult which the agent
        couldn't learn from, causing infinite retry loops.
        """
        # Set up permission denial
        perm_result = MagicMock()
        perm_result.allowed = False
        perm_result.feedback = None  # No user feedback — the common case
        mock_agent.permission_manager.check_command.return_value = perm_result

        # Create a mock proposal with a tool
        proposal = MagicMock()
        tool = MagicMock()
        tool.name = "finish"
        tool.arguments = {"reason": "done"}
        proposal.get_tools.return_value = [tool]

        # Call the real execute method
        result = await Agent.execute(mock_agent, proposal)

        # Must have called do_not_execute, NOT returned ActionErrorResult
        mock_agent.do_not_execute.assert_called_once()
        call_args = mock_agent.do_not_execute.call_args
        assert call_args[0][0] is proposal
        assert "Permission denied" in call_args[0][1]
        assert "different approach" in call_args[0][1]

        # Result should be ActionInterruptedByHuman, not ActionErrorResult
        assert isinstance(result, ActionInterruptedByHuman)

    @pytest.mark.asyncio
    async def test_denied_command_with_user_feedback_passes_it_through(
        self, mock_agent
    ):
        """When permission is denied WITH user feedback, that feedback
        should be passed to do_not_execute."""
        perm_result = MagicMock()
        perm_result.allowed = False
        perm_result.feedback = "Not yet, keep working on the task."
        mock_agent.permission_manager.check_command.return_value = perm_result

        proposal = MagicMock()
        tool = MagicMock()
        tool.name = "finish"
        tool.arguments = {"reason": "done"}
        proposal.get_tools.return_value = [tool]

        await Agent.execute(mock_agent, proposal)

        mock_agent.do_not_execute.assert_called_once()
        # User's explicit feedback should be used, not the generic message
        assert mock_agent.do_not_execute.call_args[0][1] == (
            "Not yet, keep working on the task."
        )

    @pytest.mark.asyncio
    async def test_allowed_command_does_not_call_do_not_execute(self, mock_agent):
        """Sanity check: allowed commands should NOT trigger do_not_execute."""
        perm_result = MagicMock()
        perm_result.allowed = True
        perm_result.feedback = None
        mock_agent.permission_manager.check_command.return_value = perm_result

        proposal = MagicMock()
        tool = MagicMock()
        tool.name = "web_search"
        tool.arguments = {"query": "test"}
        proposal.get_tools.return_value = [tool]

        # Need to mock the actual command execution path
        mock_agent.commands = []
        mock_agent.run_pipeline = AsyncMock(return_value=[])
        mock_agent._remove_disabled_commands = MagicMock()

        # This will fail because we haven't mocked the full execution chain,
        # but do_not_execute should NOT have been called
        try:
            await Agent.execute(mock_agent, proposal)
        except Exception:
            pass  # Expected — we only care that do_not_execute wasn't called

        mock_agent.do_not_execute.assert_not_called()
