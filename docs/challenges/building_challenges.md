# Creating Challenges for AutoGPT

🏹 We're on the hunt for talented Challenge Creators! 🎯

Join us in shaping the future of Auto-GPT by designing challenges that test its limits. Your input will be invaluable in guiding our progress and ensuring that we're on the right track. We're seeking individuals with a diverse skill set, including:

🎨 UX Design: Your expertise will enhance the user experience for those attempting to conquer our challenges. With your help, we'll develop a dedicated section in our wiki, and potentially even launch a standalone website.

💻 Coding Skills: Proficiency in Python, pytest, and VCR (a library that records OpenAI calls and stores them) will be essential for creating engaging and robust challenges.

⚙️ DevOps Skills: Experience with CI pipelines in GitHub and possibly Google Cloud Platform will be instrumental in streamlining our operations.

Are you ready to play a pivotal role in Auto-GPT's journey? Apply now to become a Challenge Creator by opening a PR! 🚀


# Getting Started
Clone the original AutoGPT repo and checkout to master branch


The challenges are not written using a specific framework. They try to be very agnostic
The challenges are acting like a user that wants something done: 
INPUT:
- User desire
- Files, other inputs

Output => Artifact (files, image, code, etc, etc...)

## Defining your Agent

Go to https://github.com/Significant-Gravitas/Auto-GPT/blob/master/tests/integration/agent_factory.py

Create your agent fixture.

```python
def kubernetes_agent(
    agent_test_config, memory_json_file, workspace: Workspace
):
    # Please choose the commands your agent will need to beat the challenges, the full list is available in the main.py
    # (we 're working on a better way to design this, for now you have to look at main.py)
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.app")

    # Define all the settings of our challenged agent
    ai_config = AIConfig(
        ai_name="Kubernetes",
        ai_role="an autonomous agent that specializes in creating Kubernetes deployment templates.",
        ai_goals=[
            "Write a simple kubernetes deployment file and save it as a kube.yaml.",
        ],
    )
    ai_config.command_registry = command_registry

    system_prompt = ai_config.construct_full_prompt()
    Config().set_continuous_mode(False)
    agent = Agent(
        # We also give the AI a name 
        ai_name="Kubernetes-Demo",
        memory=memory_json_file,
        full_message_history=[],
        command_registry=command_registry,
        config=ai_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent
```

## Creating your challenge
Go to `tests/integration/challenges`and create a file that is called `test_your_test_description.py` and add it to the appropriate folder. If no category exists you can create a new one.

Your test could look something like this 

```python
import contextlib
from functools import wraps
from typing import Generator

import pytest
import yaml

from autogpt.commands.file_operations import read_file, write_to_file
from tests.integration.agent_utils import run_interaction_loop
from tests.integration.challenges.utils import run_multiple_times
from tests.utils import requires_api_key


def input_generator(input_sequence: list) -> Generator[str, None, None]:
    """
    Creates a generator that yields input strings from the given sequence.

    :param input_sequence: A list of input strings.
    :return: A generator that yields input strings.
    """
    yield from input_sequence


@pytest.mark.skip("This challenge hasn't been beaten yet.")
@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@run_multiple_times(3)
def test_information_retrieval_challenge_a(kubernetes_agent, monkeypatch) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs
    and checking the output file content.

    :param get_company_revenue_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """
    input_sequence = ["s", "s", "s", "s", "s", "EXIT"]
    gen = input_generator(input_sequence)
    monkeypatch.setattr("builtins.input", lambda _: next(gen))

    with contextlib.suppress(SystemExit):
        run_interaction_loop(kubernetes_agent, None)

    # here we load the output file
    file_path = str(kubernetes_agent.workspace.get_path("kube.yaml"))
    content = read_file(file_path)

    # then we check if it's including keywords from the kubernetes deployment config
    for word in ["apiVersion", "kind", "metadata", "spec"]:
        assert word in content, f"Expected the file to contain {word}"

    content = yaml.safe_load(content)
    for word in ["Service", "Deployment", "Pod"]:
        assert word in content["kind"], f"Expected the file to contain {word}"


```
