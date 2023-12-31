from __future__ import annotations

from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy


def load_all_strategies() -> list[AbstractPromptStrategy]:
    import AFAAS.prompts.planner as planner
    import AFAAS.prompts.routing as routing
    import AFAAS.prompts.usercontext as usercontext

    return (
        planner.StrategiesSet.get_strategies()
        + routing.StrategiesSet.get_strategies()
        + usercontext.StrategiesSet.get_strategies()
    )


def load_all_strategiesv2() -> list[AbstractPromptStrategy]:
    import importlib
    import pkgutil

    import AFAAS.prompts

    strategies = []
    # Iterate through all modules in the AFAAS.prompts package
    for _, module_name, _ in pkgutil.iter_modules(
        AFAAS.prompts.__path__, AFAAS.prompts.__name__ + "."
    ):
        # Dynamically import the module
        module = importlib.import_module(module_name)
        # Check if the module has a 'StrategiesSet' attribute
        if hasattr(module, "StrategiesSet"):
            # Get the strategies from the StrategiesSet
            module_strategies = getattr(module, "StrategiesSet").get_strategies()
            strategies.extend(module_strategies)
    return strategies
