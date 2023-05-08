import click

# from autogpt.core.agent import Agent
# from autogpt.core.budget import BudgetManager
# from autogpt.core.command import Command, CommandRegistry
# from autogpt.core.configuration import Configuration
# from autogpt.core.llm import LanguageModel
# from autogpt.core.logging import Logger
# from autogpt.core.memory import MemoryBackend
# from autogpt.core.messaging import Message, MessageBroker
# from autogpt.core.planning import Planner
# from autogpt.core.plugin import Plugin, PluginManager
# from autogpt.core.workspace import Workspace


@click.group()
def v2():
    """Temporary command group for v2 commands."""
    pass

@v2.command()
@click.option("-a", "--is-async", is_flag=True, help="Run the agent asynchronously.")
def run(is_async: bool):
    print("Running v2 agent...")
    print(f"Is async: {is_async}")


@v2.command()
@click.option("-d", "--detailed", is_flag=True, help="Show detailed status.")
def status(detailed: bool):
    import autogpt.core.agent
    import autogpt.core.budget
    import autogpt.core.command
    import autogpt.core.configuration
    import autogpt.core.llm
    import autogpt.core.logging
    import autogpt.core.memory
    import autogpt.core.messaging
    import autogpt.core.planning
    import autogpt.core.plugin
    import autogpt.core.workspace
    import autogpt.core.runner

    modules = [
        ("Runner", autogpt.core.runner.status.name, autogpt.core.runner.handover_notes),
        ("Agent", autogpt.core.agent.status.name, autogpt.core.agent.handover_notes),
        ("Budget", autogpt.core.budget.status.name, autogpt.core.budget.handover_notes),
        ("Command", autogpt.core.command.status.name, autogpt.core.command.handover_notes),
        ("Configuration", autogpt.core.configuration.status.name, autogpt.core.configuration.handover_notes),
        ("LLM", autogpt.core.llm.status.name, autogpt.core.llm.handover_notes),
        ("Logging", autogpt.core.logging.status.name, autogpt.core.logging.handover_notes),
        ("Memory", autogpt.core.memory.status.name, autogpt.core.memory.handover_notes),
        ("Messaging", autogpt.core.messaging.status.name, autogpt.core.messaging.handover_notes),
        ("Planning", autogpt.core.planning.status.name, autogpt.core.planning.handover_notes),
        ("Plugin", autogpt.core.plugin.status.name, autogpt.core.plugin.handover_notes),
        ("Workspace", autogpt.core.workspace.status.name, autogpt.core.workspace.handover_notes),
    ]


    print("Getting v2 agent status...")
    if not detailed:
        # print list of module status in the format module name | status. Make sure columns are standard width and have a header
        print("|{:-<17}|{:-<17}|".format("", ""))
        print("| {:<15} | {:<15} |".format("Name", "Status"))
        print("|{:-<17}|{:-<17}|".format("", ""))


        for module_name, status, notes in modules:
            print("| {:<15} | {:<15} |".format(module_name, status))
        print("|{:-<17}|{:-<17}|".format("", ""))
    else:
        print("\nHere are some handover notes from the last contributor to work on the system. ")
        print("These are not necessarily up to date, but should give you a good idea of where to jump in.\n")
        for module_name, status, notes in modules:
            print(f"{module_name}:")
            print(f"\t {notes}\n")

