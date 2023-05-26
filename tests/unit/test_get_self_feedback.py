from datetime import datetime

from autogpt.agent.agent import Agent
from autogpt.config import AIConfig
from autogpt.llm.chat import create_chat_completion
from autogpt.log_cycle.log_cycle import LogCycleHandler


def test_get_self_feedback(mocker):
    # Define a sample thoughts dictionary
    thoughts = {
        "reasoning": "Sample reasoning.",
        "plan": "Sample plan.",
        "thoughts": "Sample thoughts.",
    }

    # Define a fake response for the create_chat_completion function
    fake_response = (
        "The AI Agent has demonstrated a reasonable thought process, but there is room for improvement. "
        "For example, the reasoning could be elaborated to better justify the plan, and the plan itself "
        "could be more detailed to ensure its effectiveness. In addition, the AI Agent should focus more "
        "on its core role and prioritize thoughts that align with that role."
    )

    # Mock the create_chat_completion function
    mock_create_chat_completion = mocker.patch(
        "autogpt.agent.agent.create_chat_completion", wraps=create_chat_completion
    )
    mock_create_chat_completion.return_value = fake_response

    # Create a MagicMock object to replace the Agent instance
    agent_mock = mocker.MagicMock(spec=Agent)

    # Mock the config attribute of the Agent instance
    agent_mock.config = AIConfig()

    # Mock the log_cycle_handler attribute of the Agent instance
    agent_mock.log_cycle_handler = LogCycleHandler()

    # Mock the create_nested_directory method of the LogCycleHandler instance
    agent_mock.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Mock the cycle_count attribute of the Agent instance
    agent_mock.cycle_count = 0

    # Call the get_self_feedback method
    feedback = Agent.get_self_feedback(
        agent_mock,
        thoughts,
        "gpt-3.5-turbo",
    )

    # Check if the response is a non-empty string
    assert isinstance(feedback, str) and len(feedback) > 0

    # Check if certain keywords from input thoughts are present in the feedback response
    for keyword in ["reasoning", "plan", "thoughts"]:
        assert keyword in feedback
