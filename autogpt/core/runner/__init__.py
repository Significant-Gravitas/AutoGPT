"""
This module contains the runner for the v2 agent server and client.
"""
import click
from autogpt.core.status import Status

# Temporary status and handover notes for the re-architecture activity
status = Status.IN_PROGRESS
handover_notes = "A basic client and server is being sketched out. It is at the idea stage."


@click.group()
def runner() -> None:
    """Auto-GPT Runner commands"""
    pass

@runner.command()
def client() -> None:
    """Run the Auto-GPT runner client."""
    import autogpt.core.runner.client
    print("Running Auto-GPT runner client...")
    autogpt.core.runner.client.run()

@runner.command()
def server() -> None:
    """Run the Auto-GPT runner server."""
    import autogpt.core.runner.server
    import autogpt.core.messaging.base
    print("Running Auto-GPT runner server...")
    msg = autogpt.core.messaging.base.Message({
            "message": "Translated user input into objective prompt.",
            "objective_prompt": "test auto-gpt",
        },
        None)
    autogpt.core.runner.server.launch_agent("msg")