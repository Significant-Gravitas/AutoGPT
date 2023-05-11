# Creating Challenges for AutoGPT

🏹 We're on the hunt for talented Challenge Creators! 🎯

Join us in shaping the future of Auto-GPT by designing challenges that test its limits. Your input will be invaluable in guiding our progress and ensuring that we're on the right track. We're seeking individuals with a diverse skill set, including:

🎨 UX Design: Your expertise will enhance the user experience for those attempting to conquer our challenges. With your help, we'll develop a dedicated section in our wiki, and potentially even launch a standalone website.

💻 Coding Skills: Proficiency in Python, pytest, and VCR (a library that records OpenAI calls and stores them) will be essential for creating engaging and robust challenges.

⚙️ DevOps Skills: Experience with CI pipelines in GitHub and possibly Google Cloud Platform will be instrumental in streamlining our operations.

Are you ready to play a pivotal role in Auto-GPT's journey? Apply now to become a Challenge Creator! 🚀


# Getting Started
Clone the original AutoGPT repo and checkout to master branch

## Defining your Agent

Go to https://github.com/Significant-Gravitas/Auto-GPT/blob/master/tests/integration/agent_factory.py

Create your agent fixture, in line 8 through 13 we define all the settings of our challenged agent. We also give it a name in line 20.

Please choose the commands your agent will need to beat the challenges, the full list is available in the main.py See lines 4 to 6. (we 're working on a better way to design this, for now you have to look at main.py)

```python=
def kubernetes_agent(
    agent_test_config, memory_local_cache, workspace: Workspace
):
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.app")

    ai_config = AIConfig(
        ai_name="Kubernetes",
        ai_role="an autonomous agent that specializes in creating Kubernetes deployment templates.",
        ai_goals=[
            "Write the a simple kubernetes deployment file and save it as kube.yaml. You should make a simple nginx web server that uses docker and exposes the port 80.",
        ],
    )
    ai_config.command_registry = command_registry

    system_prompt = ai_config.construct_full_prompt()
    Config().set_continuous_mode(False)
    agent = Agent(
        ai_name="Kubernetes-Demo",
        memory=memory_local_cache,
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
Go to `tests/integration/challenges`and create a file that is called `test_your_test_description.py` and add it to the appropriate folder.

