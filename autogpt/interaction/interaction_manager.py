from __future__ import annotations

from autogpt.singleton import Singleton
from autogpt.interaction.base import InteractionProviderSingleton

from typing import List


## TODO: MAKE THIS USEFUL, STRUCTURE IS NOT WELL THOUGHT OUT FOR IMPLEMENTATION
class InteractionManager(metaclass=Singleton):
    def __init__(self, cfg):
        self.registered: List(InteractionProviderSingleton) = []

    def register(self, interactor: InteractionProviderSingleton):
        self.registered.append(interactor)

    def remove_item(self, item: InteractionProviderSingleton):
        pass
