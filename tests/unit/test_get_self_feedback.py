from autogpt.agent.agent import Agent
from autogpt.config import AIConfig
from autogpt.llm import create_chat_completion


def test_get_self_feedback(mocker):
    # Define a sample thoughts dictionary
    thoughts = {
        "reasoning": "Sample reasoning.",
        "plan": "Sample plan.",
        "thoughts": "Sample thoughts.",
        "criticism": "Sample criticism.",
    }

    # Define a fake response for the create_chat_completion function
    fake_response = (
        "Y The provided information is suitable for achieving the role's objectives."
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

    # Call the get_self_feedback method
    feedback = Agent.get_self_feedback(
        agent_mock,
        thoughts,
        "gpt-3.5-turbo",
    )

    # Check if the response is correct
    assert feedback == fake_response
