from __future__ import annotations

from autogpt.interaction.base import InteractionProviderSingleton
from autogpt.logs import Logger
from autogpt.utils import clean_input


## TODO: MAKE THIS USEFUL, STRUCTURE IS NOT WELL THOUGHT OUT FOR OUR IMPLEMENTATION
class Console(InteractionProviderSingleton):
    """A class that provides interaction via console"""

    def __init__(self, logger: Logger):
        """Inits the provider"""
        self._logger = logger

    def input(self, message: str):
        """Gets from input-er"""
        return clean_input(message)

    def output(self, context, text: str):
        """Outputs to output-er"""
        self._logger.typewriter_log(text)

    @property
    def logger(self):
        return self._logger
