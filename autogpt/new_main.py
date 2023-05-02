from autogpt.core import (
    Agent,
    BudgetManager,
    CommandRegistry,
    Configuration,
    LanguageModel,
    Logger,
    MemoryBackend,
    MessageBroker,
    Planner,
    PluginManager,
    Workspace,
)


def run_auto_gpt(
    *args,
    **kwargs,
):
    # Configure logging before we do anything else.
    # Right here, I think we only configure it to log to stdout.
    logger = Logger(*args, **kwargs)

    # Configuration should, I propose, do all the input validation.
    # Pass in the logger so we can log warnings and errors.
    # Probably don't need the config to hold a long term reference to the logger
    configuration = Configuration(logger, *args, **kwargs)

    # Setup the workspace next so we can use it in all the other components.
    # First do the on disk operations to make a workspace if we need to and fill
    # it with default files like a serialized version of the configuration,
    # the ai settings, etc.
    workspace_path = Workspace.setup_workspace(configuration, logger, *args, **kwargs)
    # Then make the abstraction (could return the abstraction directly from the above
    workspace = Workspace(workspace_path, configuration, logger, *args, **kwargs)

    # Setup the plugin manager next so we can use plugins in all the other components.
    plugin_manager = PluginManager(configuration, logger, workspace, *args, **kwargs)

    # Remaining systems should be (I think at this point) independent of each other,
    # so we can instantiate them in any order.
    budget_manager = BudgetManager(
        configuration,
        logger,
        workspace,
        plugin_manager,
        *args,
        **kwargs,
    )
    command_registry = CommandRegistry(
        configuration,
        logger,
        workspace,
        plugin_manager,
        *args,
        **kwargs,
    )
    language_model = LanguageModel(
        configuration,
        logger,
        workspace,
        plugin_manager,
        *args,
        **kwargs,
    )
    memory_backend = MemoryBackend(
        configuration,
        logger,
        workspace,
        plugin_manager,
        *args,
        **kwargs,
    )
    message_broker = MessageBroker(
        configuration,
        logger,
        workspace,
        plugin_manager,
        *args,
        **kwargs,
    )
    planner = Planner(
        configuration,
        logger,
        workspace,
        plugin_manager,
        *args,
        **kwargs,
    )

    # Finally, we can instantiate the agent with all subsystems
    agent = Agent(
        configuration=configuration,
        logger=logger,
        budget_manager=budget_manager,
        command_registry=command_registry,
        language_model=language_model,
        memory_backend=memory_backend,
        message_broker=message_broker,
        planner=planner,
        plugin_manager=plugin_manager,
        workspace=workspace,
    )

    agent.run()
