from autogpt.agents import Agent, AgentThoughts, CommandArgs, CommandName
# https://api.github.com/repos/jmikedupont2/ai-ticket/issues/comments/1735879231

command_name = "request_assistance"

command_args = {
    'ticket_url': 'https://api.github.com/repos/jmikedupont2/ai-ticket/issues/comments/1735879231',
    'next_action': 'poll_url'
}

user_input = ""
memory = None #get_memory(config)
#memory.clear()
from autogpt.commands import COMMAND_CATEGORIES

from autogpt.models.command_registry import CommandRegistry
working_directory = "."
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.config import AIConfig, Config, ConfigBuilder, check_openai_api_key
config = ConfigBuilder.build_config_from_env(workdir=working_directory)
config.workspace_path = "."
config.file_logger_path="log.txt"
command_registry = CommandRegistry.with_command_modules(COMMAND_CATEGORIES, config)
ai_config = AIConfig.load("./ai_settings.yaml")
agent = Agent(
    memory=memory,
    command_registry=command_registry,
    triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
    ai_config=ai_config,
    config=config,
)

class Foo :
    content = """```{"command": {"name": "request_assistance", "args": {"ticket_url": "https://api.github.com/repos/jmikedupont2/ai-ticket/issues/comments/1735910623", "next_action": "poll_url"}}, "thoughts": {"plan": "Initiated a request for assistance.", "speak": "TODO:cv.safe_generate(json.dumps(data))", "criticism": "todo", "reasoning": "todo", "text": "I encountered an issue with our application, and I need assistance. I\'ve created a ticket for it. Here\'s the URL to the ticket: https://api.github.com/repos/jmikedupont2/ai-ticket/issues/comments/1735910623. My next action is to poll that URL for updates."}}```"""
    
data = agent.parse_and_process_response(Foo())
#print(data)
result = agent.execute(command_name, command_args, user_input)

print(result)
