import abc


class Workspace(abc.ABC):
    @abc.abstractmethod
    def setup_workspace(self, *arg, **kwargs):
        pass
