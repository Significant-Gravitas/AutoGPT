import abc
from autogpt.core import Logger


class Configuration(abc.ABC):
    @abc.abstractmethod
    def __init__(self, logger: Logger, *arg, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def command_registry(self):
        pass

    @property
    @abc.abstractmethod
    def language_model(self):
        pass

    @property
    @abc.abstractmethod
    def memory_backend(self):
        pass

    @property
    @abc.abstractmethod
    def message_broker(self):
        pass

    @property
    @abc.abstractmethod
    def planner(self):
        pass

    @property
    @abc.abstractmethod
    def plugin_manager(self):
        pass

    @property
    @abc.abstractmethod
    def workspace(self):
        pass

    @abc.abstractmethod
    def set_defaults(self, system_classes: list):
        pass
