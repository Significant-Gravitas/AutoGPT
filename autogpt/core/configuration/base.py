import abc


class Configuration(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *arg, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def budget_manager(self):
        pass

    @property
    @abc.abstractmethod
    def command_registry(self):
        pass

    @property
    @abc.abstractmethod
    def credentials(self):
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
    def to_dict(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass
