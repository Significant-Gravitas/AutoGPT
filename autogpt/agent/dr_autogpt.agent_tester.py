from autogpt.commands.command import CommandRegistry
from autogpt.config.ai_config import AIConfig
from autogpt.agent.agent import Agent
from autogpt.memory.vector import VectorMemory

# Create a new VectorMemory instance for testing.
memory = VectorMemory()

# Create a new CommandRegistry instance.
command_registry = CommandRegistry()

# Add the `print` command to the command registry.
command_registry.add_command("print", lambda args: f"Printing: {args}")

# Create a new AIConfig instance.
config = AIConfig()

# Create a new Agent instance with the VectorMemory instance and other required parameters.
agent = Agent(
    ai_name="Test AI",
    memory=memory,
    next_action_count=1,
    command_registry=command_registry,
    config=config,
    system_prompt="",
    triggering_prompt="",
    workspace_directory="",
)

# Execute the `print` command through the Agent instance.
result = agent.execute_command("print('Hello, world!')")

# Print the result of the command execution.
print(result)
